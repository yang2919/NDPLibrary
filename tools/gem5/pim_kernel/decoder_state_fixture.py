"""Exercise a tiny stateful decoder through generic worker operations and KV regions."""
import math
import struct


def decoder_state_requests(emit):
    values = {}
    live = []
    checks = 0

    def rounded(xs):
        return [struct.unpack('<e', struct.pack('<e', x))[0] for x in xs]

    def alloc(xs, role=b'X', rows=1, cols=None, upload=False):
        handle = len(values) + 1
        values[handle] = rounded(xs)
        live.append(handle)
        emit(struct.pack('<ccII', b'C', role, rows, len(xs) if cols is None else cols),
             struct.pack('<I', handle))
        if upload:
            emit(struct.pack(f'<cI{len(xs)}e', b'U', handle, *values[handle]))
        return handle

    def tensor(xs):
        return alloc(xs, upload=True)

    def check(handle):
        nonlocal checks
        checks += 1
        emit(struct.pack('<cI', b'D', handle),
             struct.pack(f'<{len(values[handle])}e', *values[handle]))

    def gather(src, indices):
        out = alloc([values[src][i] for i in indices])
        emit(struct.pack(f'<c{2+len(indices)}I', b'K', out, src, *indices))
        return out

    def dense(op, left, right, m, n, k, expected, factor=1):
        out = alloc(expected)
        emit(struct.pack('<c8If', b'A', out, left, right, 0, op, m, n, k, factor))
        return out

    scale = tensor([1, 1])

    def norm(src, rows):
        x = values[src]
        out = []
        for i in range(rows):
            row = x[i*2:i*2+2]
            inv = 1/math.sqrt(sum(t*t for t in row)/2 + 0.5)
            out.extend(t*inv for t in row)
        return dense(5, src, scale, rows, 2, 0, out, 0.5)

    def add(a, b):
        return dense(6, a, b, 1, 2, 0, [x+y for x, y in zip(values[a], values[b])])

    def project(src, weight, rows):
        x, w = values[src], values[weight]
        result = [sum(x[i*2+k]*w[j*2+k] for k in range(2))
                  for i in range(rows) for j in range(2)]
        if rows > 1:
            return dense(2, src, weight, rows, 2, 2, result)
        out = alloc(result, b'Y')
        emit(struct.pack('<cIII', b'G', out, weight, src))
        return out

    def multiply(a, b):
        out = alloc([x*y for x, y in zip(values[a], values[b])])
        emit(struct.pack('<cIII', b'M', out, a, b))
        return out

    def rope(src, position):
        # Positions use 0, pi/2 and pi: exact rotations isolate cache/order errors.
        if position % 4 == 0:
            return src
        if position % 4 == 1:
            return multiply(gather(src, [1, 0]), tensor([-1, 1]))
        return multiply(src, tensor([-1, -1]))

    weights = [alloc(w, b'W', 2, 2, True) for w in (
        [0.5, 0.25, -0.25, 0.5], [0.25, 0.5, 0.5, -0.25],
        [0.5, -0.5, 0.25, 0.5], [0.5, 0.25, 0, 0.5],
        [0.25, 0.5, -0.5, 0.25], [0.5, 0, 0.25, 0.5],
        [0.5, 0.25, 0.25, -0.5])]
    key_cache, value_cache = alloc([0]*6, b'B'), alloc([0]*6, b'B')
    persistent_count = len(live)
    tokens = [[1, -0.5], [0.25, 1], [-1, 0.5]]
    baseline = None
    for schedule in ((2, 1), (1, 1, 1), (1,)):
        for cache in (key_cache, value_cache):
            emit(struct.pack('<cII', b'F', cache, 0))
            values[cache] = [0]*6
        position = 0
        outputs = []
        for rows in schedule:
            source = tensor([x for row in tokens[position:position+rows] for x in row])
            normalized = norm(source, rows)
            projected = [project(normalized, w, rows) for w in weights[:3]]
            # Materialize every token of this chunk before attention. Each query
            # reads only its active prefix, so the first cannot attend to the future.
            queries = []
            for offset in range(rows):
                q, k, v = [gather(h, [offset*2, offset*2+1]) for h in projected]
                q, k = rope(q, position+offset), rope(k, position+offset)
                queries.append(q)
                for cache, src in ((key_cache, k), (value_cache, v)):
                    at = 2*(position+offset)
                    emit(struct.pack('<cIIIII', b'B', cache, src, 0, at, 2))
                    values[cache][at:at+2] = values[src]
            for offset, q in enumerate(queries):
                length = position+offset+1
                active = []
                for cache in (key_cache, value_cache):
                    h = alloc(values[cache][:length*2])
                    emit(struct.pack('<cIIIII', b'B', h, cache, 0, 0, length*2))
                    active.append(h)
                k, v = active
                transposed = dense(3, k, 0, length, 2, 0,
                                   [values[k][r*2+c] for c in range(2) for r in range(length)])
                scores = dense(1, q, transposed, 1, length, 2,
                               [sum(values[q][c]*values[k][r*2+c] for c in range(2))*0.5
                                for r in range(length)], 0.5)
                exps = [math.exp(x-max(values[scores])) for x in values[scores]]
                probabilities = dense(4, scores, 0, 1, length, 0, [x/sum(exps) for x in exps])
                attention = dense(1, probabilities, v, 1, 2, length,
                                  [sum(values[probabilities][r]*values[v][r*2+c]
                                       for r in range(length)) for c in range(2)])
                check(scores)
                check(probabilities)
                residual = add(gather(source, [offset*2, offset*2+1]), project(attention, weights[3], 1))
                ff_input = norm(residual, 1)
                gate, up = [project(ff_input, w, 1) for w in weights[4:6]]
                activated = alloc([x/(1+math.exp(-x)) for x in values[gate]])
                emit(struct.pack('<cII', b'S', activated, gate))
                result = add(residual, project(multiply(activated, up), weights[6], 1))
                check(result)
                outputs.append(values[result])
            position += rows
            # Read the entire cache, including unused slots, without re-upload.
            for cache in (key_cache, value_cache):
                snapshot = alloc(values[cache])
                emit(struct.pack('<cIIIII', b'B', snapshot, cache, 0, 0, 6))
                check(snapshot)
        if baseline is None:
            baseline = outputs
        elif outputs != baseline[:len(outputs)]:
            raise AssertionError('independent chunk/token/reset references differ')
        for handle in reversed(live[persistent_count:]):
            emit(struct.pack('<cI', b'R', handle))
        del live[persistent_count:]
    for handle in reversed(live):
        emit(struct.pack('<cI', b'R', handle))
    return checks
