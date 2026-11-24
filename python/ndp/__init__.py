"""Run explicit, inference-only NDP operations with resident device handles."""
import torch as _torch  # Load Torch shared libraries before importing the extension.
from ._C import (
    _Session, _DeviceGroup, NdpError, NdpTensor, DeviceBuffer, TensorCopy, Event, HostTransfer,
    Engine, DType, LayoutKind, LayoutPolicy, EventStatus, ErrorCode, OpKind,
    OpOptions, OpAttributes, OperationSpec, TensorDesc, LayoutDesc, KernelPlan,
    MemoryStats, DeviceCaps, RepresentationInfo,
)


class Session(_Session):
    """A single-device runtime owned by its creating Python thread."""

    @classmethod
    def connect(cls, host, port):
        """Open an exclusive FP16 worker session over TCP."""
        import socket
        with socket.create_connection((host, port), timeout=15) as stream:
            stream.setblocking(True)
            return cls(backend="gem5_pim", worker_fd=stream.fileno())

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as shutdown_error:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(f"NDP shutdown also failed: {shutdown_error}")
            else:
                exc.ndp_shutdown_error = shutdown_error
        return False


class DeviceGroup:
    """Independent devices in ordinal order; create one session per backend option dict."""

    def __init__(self, devices):
        self._sessions = tuple(d if isinstance(d, Session) else Session(**d) for d in devices)
        self._group = _DeviceGroup(self._sessions)

    def __len__(self):
        return len(self._sessions)

    def all_gather_rows(self, shards):
        """Concatenate matrix feature shards in device order and replicate the result."""
        return self._group._all_gather_rows(shards)

    def broadcast(self, tensor):
        """Replicate a dense tensor with one download shared by all destination uploads."""
        return self._group._broadcast(tensor, self._sessions)

    def reduce_partials(self, partials, destination=0):
        return self._group._reduce_partials(partials, destination, self.device(destination))

    def device(self, ordinal):
        if type(ordinal) is not int or not 0 <= ordinal < len(self._sessions):
            raise ValueError("Invalid device ordinal")
        return self._sessions[ordinal]

    def copy_to(self, tensor, destination):
        return self._group._copy_to(tensor, destination, self.device(destination))


from . import _dispatch, llm, nn, serving, pipeline
