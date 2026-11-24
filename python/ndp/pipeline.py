"""Overlap sequential TP stages across independent requests using thread-owned Sessions."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
from dataclasses import dataclass
import threading
import time

import torch

from . import Session, DeviceGroup
from .nn import TensorParallelLlamaDecodeStack, _ResidentRMSNorm, NdpLinear
from .serving import GenerationRequest, GenerationStep


@dataclass
class _StageResult:
    value: object
    error: str | None
    started: float
    finished: float


class _Stage:
    """Created, called and closed only on its executor's single thread."""

    def __init__(self, index, layers, options, capacity, final_norm=None, head=None):
        self.index = index
        self.sessions = []
        self.requests = {}
        try:
            for option in options:
                self.sessions.append(Session(**option))
            self.stack = TensorParallelLlamaDecodeStack(
                layers, devices=DeviceGroup(self.sessions), max_tokens=capacity)
            self.norm = _ResidentRMSNorm(final_norm, self.sessions[0]) if final_norm is not None else None
            self.head = copy.deepcopy(head).requires_grad_(False) if head is not None else None
        except BaseException:
            self.close()
            raise

    @torch.inference_mode()
    def run(self, request_id, capacity, hidden):
        started = time.perf_counter()
        try:
            request = self.requests.get(request_id)
            if request is None:
                request = self.stack.new_request(max_tokens=capacity)
                self.requests[request_id] = request
            if self.norm is None:
                value = self.stack(hidden, request=request)
            else:
                root = self.sessions[0]
                resident = self.stack.forward_ndp(root.upload(hidden.reshape(-1, hidden.shape[-1])), request=request)
                normalized = root.download(self.norm(resident)).reshape_as(hidden)
                value = self.head(normalized).float()
            return _StageResult(value, None, started, time.perf_counter())
        except Exception as error:
            # No exception traceback crosses the thread boundary holding native tensors.
            return _StageResult(None, f'{type(error).__name__}: {error}', started, time.perf_counter())

    def release(self, request_id):
        request = self.requests.pop(request_id, None)
        if request is not None:
            request.close()

    def memory_stats(self):
        result = []
        for session in self.sessions:
            session.synchronize()
            stats = session.memory_stats()
            result.append(dict(live_bytes=stats.live_bytes, peak_bytes=stats.peak_bytes))
        return result

    def close(self):
        for request_id in list(self.requests):
            self.release(request_id)
        # Drop all device handles on their owning thread before closing connections.
        self.stack = self.norm = self.head = None
        errors = []
        for session in self.sessions:
            try:
                session.close()
            except Exception as error:
                errors.append(f'{type(error).__name__}: {error}')
        self.sessions.clear()
        return errors


class PipelineGenerationScheduler:
    """One in-flight token per request, with different requests overlapping across stages."""

    def __init__(self, source, *, device_groups, placement, max_tokens,
                 max_active_requests=2, record_timing=False, prefill_chunk_size=1):
        if type(max_active_requests) is not int or max_active_requests < 1:
            raise ValueError('max_active_requests must be a positive integer')
        options = tuple(tuple(dict(option) if isinstance(option, dict) else () for option in group)
                        for group in device_groups)
        placement = tuple(placement)
        count = len(options)
        if count < 2 or any(not group or any(not isinstance(o, dict) for o in group) for group in options):
            raise ValueError('Provide at least two nonempty groups of backend option dictionaries')
        # Stages are contiguous and run in increasing ordinal order; no cycles.
        if (len(placement) != len(source.model.layers)
                or any(type(i) is not int for i in placement)
                or sorted(set(placement)) != list(range(count))
                or list(placement) != sorted(placement)):
            raise ValueError('Placement must cover contiguous stages 0, 1, ... in layer order')
        if source.training or source.config.model_type != 'llama':
            raise ValueError('Expected an eval-mode Llama model')
        NdpLinear._validate(source.model.embed_tokens.weight, 'embedding')
        NdpLinear._validate(source.lm_head.weight, 'LM head')
        if type(max_tokens) is not int or not 0 < max_tokens <= source.config.max_position_embeddings:
            raise ValueError('Invalid model capacity')
        if (source.model.embed_tokens.weight.shape != source.lm_head.weight.shape
                or source.lm_head.weight.shape[1] != source.config.hidden_size
                or source.model.norm.weight.shape != (source.config.hidden_size,)):
            raise ValueError('Embedding, LM head and final norm dimensions must agree')
        self._embedding = copy.deepcopy(source.model.embed_tokens).requires_grad_(False)
        self._eos = source.config.eos_token_id
        if type(prefill_chunk_size) is not int or prefill_chunk_size < 1:
            raise ValueError('prefill_chunk_size must be a positive integer')
        self._chunk_size = prefill_chunk_size
        self._capacity = max_tokens
        self._limit = max_active_requests
        self._thread = threading.get_ident()
        self._waiting = deque()
        self._live = {}
        self._flights = {}
        self._next_id = 0
        self._closed = self._stepping = False
        self._timings = [] if record_timing else None
        self._executors, self._stages = [], []
        try:
            for i, group in enumerate(options):
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'ndp-stage-{i}')
                self._executors.append(executor)
                layers = [layer for layer, ordinal in zip(source.model.layers, placement) if ordinal == i]
                stage = executor.submit(_Stage, i, layers, group, max_tokens,
                    source.model.norm if i == count-1 else None,
                    source.lm_head if i == count-1 else None).result()
                self._stages.append(stage)
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                if hasattr(error, 'add_note'):
                    error.add_note(f'Pipeline cleanup failed: {cleanup_error}')
            raise

    def _check(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError('Use the scheduler on its creating thread')
        if self._closed or self._stepping:
            raise RuntimeError('Scheduler is closed or a call is reentrant')

    @property
    def max_active_requests(self):
        return self._limit

    @property
    def pending(self):
        return len(self._waiting)+len(self._live)

    @property
    def timings(self):
        return tuple(dict(item) for item in (self._timings or ()))

    def submit(self, input_ids, *, max_new_tokens=16, eos_token_id=None):
        self._check()
        vocab = self._embedding.num_embeddings
        if (not isinstance(input_ids, torch.Tensor) or input_ids.device.type != 'cpu'
                or input_ids.dtype != torch.int64 or input_ids.ndim != 2
                or input_ids.shape[0] != 1 or input_ids.shape[1] < 1
                or (input_ids < 0).any() or (input_ids >= vocab).any()):
            raise ValueError('Expected nonempty CPU int64 token IDs [1, sequence] in vocabulary')
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError('max_new_tokens must be a nonnegative integer')
        if max_new_tokens and input_ids.shape[1]+max_new_tokens-1 > self._capacity:
            raise ValueError('Prompt and decode steps exceed model capacity')
        eos = self._eos if eos_token_id is None else eos_token_id
        eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        if any(type(t) is not int or not 0 <= t < vocab for t in eos):
            raise ValueError('Invalid EOS token IDs')
        request = GenerationRequest(self, self._next_id, input_ids, max_new_tokens, eos)
        self._next_id += 1
        if max_new_tokens == 0:
            request._finish('completed')
        else:
            self._waiting.append(request)
        return request

    @torch.inference_mode()
    def _dispatch(self, request, stage, value=None):
        if stage == 0:
            count = min(self._chunk_size, request._prompt_length-request._processed) if request.status == 'prefill' else 1
            request._inflight_tokens = count
            token = torch.tensor([request._tokens[request._processed:request._processed+count]], dtype=torch.int64)
            value = self._embedding(token)
        future = self._executors[stage].submit(self._stages[stage].run, request.request_id,
            request._prompt_length+request._limit-1, value)
        self._flights[request.request_id] = (stage, future)

    def _retire(self, request, status, error=None):
        self._flights.pop(request.request_id, None)
        # FIFO executor queues ensure release runs after any already submitted stage call.
        releases = [executor.submit(stage.release, request.request_id)
                    for executor, stage in zip(self._executors, self._stages)]
        for release in releases:
            release.result()
        self._live.pop(request.request_id, None)
        request._finish(status, error)

    def step(self):
        """Advance one completed stage; different requests may be running in other stages."""
        self._check()
        self._stepping = True
        try:
            while self._waiting and len(self._live) < self._limit:
                request = self._waiting.popleft()
                request._status = 'prefill'
                self._live[request.request_id] = request
                try:
                    self._dispatch(request, 0)
                except Exception as error:
                    self._retire(request, 'failed', f'{type(error).__name__}: {error}')
                    return GenerationStep(request.request_id, 'admission', None, request.status, request.error)
            if not self._flights:
                return None
            completed, _ = wait([future for _, future in self._flights.values()], return_when=FIRST_COMPLETED)
            request_id = next(key for key, (_, future) in self._flights.items() if future in completed)
            stage_index, future = self._flights.pop(request_id)
            request = self._live[request_id]
            phase, token_id = request.status, None
            count = request._inflight_tokens
            try:
                result = future.result()
                if self._timings is not None:
                    self._timings.append(dict(request_id=request_id, token_index=request._processed,
                        stage=stage_index, input_tokens=request._inflight_tokens, started=result.started, finished=result.finished))
                if result.error is not None:
                    self._retire(request, 'failed', result.error)
                elif stage_index+1 < len(self._stages):
                    self._dispatch(request, stage_index+1, result.value)
                else:
                    request._processed += request._inflight_tokens
                    if request._processed >= request._prompt_length:
                        token_id = int(result.value[0, -1].argmax())
                        request._tokens.append(token_id)
                        if token_id in request._eos or request.generated_tokens >= request._limit:
                            self._retire(request, 'completed')
                        else:
                            request._status = 'decode'
                    if not request.done:
                        self._dispatch(request, 0)
            except Exception as error:
                self._retire(request, 'failed', f'{type(error).__name__}: {error}')
            return GenerationStep(request_id, phase, token_id, request.status, request.error, stage_index, count)
        except BaseException as error:
            # An interrupted coordinator must not leave a live request without a future.
            self._stepping = False
            try:
                self.close()
            except Exception as cleanup_error:
                if hasattr(error, 'add_note'):
                    error.add_note(f'Pipeline cleanup failed: {cleanup_error}')
            raise
        finally:
            self._stepping = False

    def run(self):
        self._check()
        while self.pending:
            self.step()

    def cancel(self, request):
        self._check()
        if not isinstance(request, GenerationRequest) or request._owner() is not self:
            raise ValueError('Request belongs to another scheduler')
        if request.done:
            return
        if request.status == 'queued':
            self._waiting.remove(request)
            request._finish('cancelled')
        else:
            self._retire(request, 'cancelled')

    def memory_stats(self):
        self._check()
        if self._live:
            raise RuntimeError('Inspect memory only with no active requests')
        return [executor.submit(stage.memory_stats).result()
                for executor, stage in zip(self._executors, self._stages)]

    def close(self):
        if self._closed:
            return
        self._check()
        for request in tuple(self._waiting):
            request._finish('cancelled')
        self._waiting.clear()
        errors = []
        for request in tuple(self._live.values()):
            try:
                self._retire(request, 'cancelled')
            except Exception as error:
                errors.append(f'{type(error).__name__}: {error}')
                self._flights.pop(request.request_id, None)
                self._live.pop(request.request_id, None)
                request._finish('cancelled')
        for executor, stage in zip(self._executors, self._stages):
            try:
                errors.extend(executor.submit(stage.close).result())
            except Exception as error:
                errors.append(f'{type(error).__name__}: {error}')
        for executor in self._executors:
            executor.shutdown(wait=True)
        self._stages.clear()
        self._embedding = None
        self._closed = True
        if errors:
            raise RuntimeError('Pipeline shutdown failed: ' + '; '.join(errors))

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
