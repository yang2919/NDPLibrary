"""Verify GEMV rounding before download and before another resident GEMV."""
import argparse
from pathlib import Path

from client import Session


def verify(worker):
    with Session(worker, 'fp16') as s:
        w = s.create('W', 3, 3)
        x = s.create('X', 1, 3)
        y = s.create('Y', 1, 3)
        half_ulp = 2 ** -11
        # Row 0 ties to even; row 1 accumulates two half-ULPs before rounding.
        # Row 2 is representable in binary16 but not bfloat16.
        s.upload(w, [1, half_ulp, 0, 1, half_ulp, half_ulp, 1 + 2 ** -10, 0, 0])
        s.upload(x, [1, 1, 1])
        s.gemv(y, w, x)
        diff = s.create('W', 1, 3)
        z = s.create('Y', 1, 1)
        s.upload(diff, [1, -1, 0])
        s.gemv(z, diff, y)  # No download/re-upload of y.
        if s.download(z) != (-2 ** -10,) or s.download(y) != (1, 1 + 2 ** -10, 1 + 2 ** -10):
            raise AssertionError('FP16 output or resident chaining mismatch')
        s.upload(x, [2, 2, 2])
        s.gemv(y, w, x)
        s.gemv(z, diff, y)
        if s.download(z) != (-2 ** -9,):
            raise AssertionError('stale result after reusing output handle')
    try:
        with Session(worker, 'bf16'):
            pass
    except RuntimeError as error:
        if 'mismatch' not in str(error):
            raise
    else:
        raise AssertionError('mismatched client profile accepted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('worker', type=Path)
    verify(parser.parse_args().worker.resolve())
    print('PASS: FP16 operands, FP32 accumulation, FP16 resident outputs and profile check')
