"""Generate small decoder arithmetic requests with independent dense references."""
import math
import struct


def decoder_requests(emit):
    handles = []

    def half(value):
        return struct.unpack('<e', struct.pack('<e', value))[0]

    def create(size, role=b'X', rows=1, cols=None):
        handle = len(handles) + 1
        handles.append(handle)
        emit(struct.pack('<ccII', b'C', role, rows, size if cols is None else cols),
             struct.pack('<I', handle))
        return handle

    def tensor(values):
        handle = create(len(values))
        emit(struct.pack(f'<cI{len(values)}e', b'U', handle, *values))
        return handle

    def check(handle, values, fp32=False):
        fmt = 'f' if fp32 else 'e'
        emit(struct.pack('<cI', b'D', handle), struct.pack(f'<{len(values)}{fmt}', *values))

    def dense(op, m, n, k, left, right=0, bias=0, factor=1, fp32=False):
        out = create(m*n, b'P' if fp32 else b'X')
        emit(struct.pack('<c8If', b'A', out, left, right, bias, op, m, n, k, factor))
        return out

    a = tensor([1, 2, 3, -1, 0, 2])
    b = tensor([1, 0, 0, 1, 1, -1])
    product = dense(1, 2, 2, 3, a, b, factor=0.5)
    check(product, [2, -0.5, 0.5, -1])

    weights = tensor([1, 0, 1, 0, 1, -1])
    bias = tensor([0.25, -0.5])
    linear = dense(2, 2, 2, 3, a, weights, bias, factor=0.5)
    check(linear, [2.25, -1, 0.75, -1.5])
    # An FP32 partial must retain a value that rounds away in FP16.
    partial = dense(2, 1, 1, 2, tensor([1, 2**-11]), tensor([1, 1]), fp32=True)
    check(partial, [1 + 2**-11], fp32=True)

    transposed = dense(3, 2, 3, 0, a)
    check(transposed, [1, -1, 2, 0, 3, 2])
    scores = [0, 1, -1, 2, 2, 2]
    softmax = dense(4, 2, 3, 0, tensor(scores))
    exp = [math.exp(x-1) for x in scores[:3]]
    check(softmax, [x/sum(exp) for x in exp] + [1/3]*3)

    # mean(x^2)+epsilon = 1 gives a hand-checkable nontrivial weighted RMSNorm.
    norm = dense(5, 2, 2, 0, tensor([1, 0, -1, 0]), tensor([0.5, 2]), factor=0.5)
    check(norm, [0.5, 0, -0.5, 0])
    added = dense(6, 2, 2, 0, product, norm)
    check(added, [2.5, -0.5, 0, -1])

    values = [-2, -0.5, 0, 1]
    activated = create(4)
    emit(struct.pack('<cII', b'S', activated, tensor(values)))
    activated_ref = [half(x/(1+math.exp(-x))) for x in values]
    check(activated, activated_ref)
    gated = create(4)
    emit(struct.pack('<cIII', b'M', gated, activated, tensor([1, -2, 0.5, 2])))
    check(gated, [x*y for x, y in zip(activated_ref, [1, -2, 0.5, 2])])

    # Split-half RoPE via gather, multiply and add; no dedicated RoPE opcode.
    q = tensor([1, 2, 3, 4])
    rotated = create(4)
    emit(struct.pack('<c6I', b'K', rotated, q, 2, 3, 0, 1))
    signed = create(4)
    emit(struct.pack('<cIII', b'M', signed, rotated, tensor([-1, -1, 1, 1])))
    cosine = half(math.sqrt(0.5))
    left, right = create(4), create(4)
    trig = tensor([cosine]*4)
    emit(struct.pack('<cIII', b'M', left, q, trig))
    emit(struct.pack('<cIII', b'M', right, signed, trig))
    rope = dense(6, 1, 4, 0, left, right)
    rope_ref = [half(half(x*cosine)+half(y*cosine))
                for x, y in zip([1, 2, 3, 4], [-3, -4, 1, 2])]
    check(rope, rope_ref)

    # Consume the resident CPU-computed RoPE output with one real PIM GEMV.
    w = create(4, b'W', rows=1, cols=4)
    emit(struct.pack('<cI4e', b'U', w, 1, 0, 0, 1))
    result = create(1, b'Y')
    emit(struct.pack('<cIII', b'G', result, w, rope))
    check(result, [half(rope_ref[0]+rope_ref[3])])
    for handle in reversed(handles):
        emit(struct.pack('<cI', b'R', handle))
