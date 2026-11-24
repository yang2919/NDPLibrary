"""Sequential round-robin generation with bounded active request histories."""
from collections import deque
from dataclasses import dataclass
import threading
import weakref

import torch


@dataclass(frozen=True)
class GenerationStep:
    """One processed input token; token_id is set only when a new token is emitted."""
    request_id: int
    phase: str
    token_id: int | None
    status: str
    error: str | None = None
    stage: int | None = None
    input_tokens: int = 1


class GenerationRequest:
    """Generation result and status; device history is released on termination."""

    def __init__(self, scheduler, request_id, prompt, limit, eos):
        self._owner = weakref.ref(scheduler)
        self._id = request_id
        self._tokens = prompt.flatten().tolist()
        self._prompt_length = len(self._tokens)
        self._limit = limit
        self._eos = frozenset(eos)
        self._processed = 0
        self._history = None
        self._status = 'queued'
        self._error = None

    @property
    def request_id(self):
        return self._id

    @property
    def status(self):
        return self._status

    @property
    def error(self):
        return self._error

    @property
    def generated_tokens(self):
        return len(self._tokens)-self._prompt_length

    @property
    def token_ids(self):
        """Return a new CPU int64 tensor containing the prompt and emitted tokens."""
        return torch.tensor([self._tokens], dtype=torch.int64)

    @property
    def done(self):
        return self._status in ('completed', 'cancelled', 'failed')

    def _finish(self, status, error=None):
        if self._history is not None:
            self._history.close()
            self._history = None
        self._status, self._error = status, error


class GenerationScheduler:
    """Alternate one input token per active request on the owning Python thread."""

    def __init__(self, model, *, max_active_requests=1, prefill_chunk_size=1):
        from .nn import TensorParallelLlamaForCausalLM
        if not isinstance(model, TensorParallelLlamaForCausalLM):
            raise TypeError('Expected TensorParallelLlamaForCausalLM')
        if type(max_active_requests) is not int or max_active_requests < 1:
            raise ValueError('max_active_requests must be a positive integer')
        if type(prefill_chunk_size) is not int or prefill_chunk_size < 1:
            raise ValueError('prefill_chunk_size must be a positive integer')
        self.prefill_chunk_size = prefill_chunk_size
        self.model = model
        self._max_active_requests = max_active_requests
        self._thread = threading.get_ident()
        self._waiting, self._active = deque(), deque()
        self._next_id = 0
        self._closed = False
        self._stepping = False

    def _check(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError('Use the scheduler on its creating thread')
        if self._closed:
            raise RuntimeError('Scheduler is closed')
        if self._stepping:
            raise RuntimeError('Scheduler calls cannot be reentrant')

    @property
    def max_active_requests(self):
        return self._max_active_requests

    @property
    def pending(self):
        return len(self._waiting)+len(self._active)

    def submit(self, input_ids, *, max_new_tokens=16, eos_token_id=None):
        self._check()
        self.model._validate_ids(input_ids)
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError('max_new_tokens must be a nonnegative integer')
        if max_new_tokens and input_ids.shape[1]+max_new_tokens-1 > self.model.model.max_tokens:
            raise ValueError('Prompt and decode steps exceed model capacity')
        eos = self.model.eos_token_id if eos_token_id is None else eos_token_id
        eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        if any(type(t) is not int or not 0 <= t < self.model.model.embed_tokens.num_embeddings for t in eos):
            raise ValueError('EOS IDs must belong to the vocabulary')
        request = GenerationRequest(self, self._next_id, input_ids, max_new_tokens, eos)
        self._next_id += 1
        if max_new_tokens == 0:
            request._finish('completed')
        else:
            self._waiting.append(request)
        return request

    def cancel(self, request):
        """Cancel a queued or active request between step calls; repeated cancel is harmless."""
        self._check()
        if not isinstance(request, GenerationRequest) or request._owner() is not self:
            raise ValueError('Request belongs to another scheduler')
        if request.done:
            return
        queue = self._waiting if request._status == 'queued' else self._active
        queue.remove(request)
        request._finish('cancelled')

    @torch.inference_mode()
    def step(self):
        """Execute at most one model call and return its event, or None when idle."""
        self._check()
        self._stepping = True
        try:
            return self._step()
        finally:
            self._stepping = False

    def _step(self):
        while self._waiting and len(self._active) < self.max_active_requests:
            request = self._waiting.popleft()
            try:
                capacity = request._prompt_length+request._limit-1
                request._history = self.model.new_request(max_tokens=capacity)
                request._status = 'prefill'
                self._active.append(request)
            except Exception as error:
                request._finish('failed', f'{type(error).__name__}: {error}')
                return GenerationStep(request.request_id, 'admission', None, request.status, request.error)
            except BaseException:
                request._finish('cancelled')
                raise
        if not self._active:
            return None
        request = self._active.popleft()
        phase = request._status
        token_id = None
        count = 0
        try:
            count = min(self.prefill_chunk_size, request._prompt_length-request._processed) if phase == 'prefill' else 1
            token = torch.tensor([request._tokens[request._processed:request._processed+count]], dtype=torch.int64)
            logits = self.model.prefill(token, request=request._history) if count > 1 else \
                self.model(token, request=request._history)
            request._processed += count
            if request._processed >= request._prompt_length:
                token_id = int(logits[0, -1].argmax())
                request._tokens.append(token_id)
                if token_id in request._eos or request.generated_tokens >= request._limit:
                    request._finish('completed')
                else:
                    request._status = 'decode'
            if not request.done:
                self._active.append(request)
        except Exception as error:
            # Keep text only: retaining a traceback can retain device tensors and caches.
            request._finish('failed', f'{type(error).__name__}: {error}')
        except BaseException:
            request._finish('cancelled')
            raise
        return GenerationStep(request.request_id, phase, token_id, request.status, request.error, input_tokens=count)

    def run(self):
        """Drain currently pending work; inspect request handles for results and failures."""
        self._check()
        while self.pending:
            self.step()

    def close(self):
        if self._closed:
            return
        self._check()
        for request in (*self._waiting, *self._active):
            request._finish('cancelled')
        self._waiting.clear()
        self._active.clear()
        self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
