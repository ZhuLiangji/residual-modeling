import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import pickle
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool

import numpy as np
import zstandard as zstd


FORMAT = "lbrc_v1"


def nrmse(a, b, scale):
    return float(np.sqrt(np.mean(((a - b) / scale) ** 2)))


def block_slices(shape, bt, bh, bw):
    B, C, T, H, W = shape
    for b in range(B):
        for c in range(C):
            for t0 in range(0, T, bt):
                for h0 in range(0, H, bh):
                    for w0 in range(0, W, bw):
                        yield b, c, t0, min(t0 + bt, T), h0, min(h0 + bh, H), w0, min(w0 + bw, W)


def load_data(path, latent_bit_arg=0):
    data = np.load(path)
    x = data["original_data"].astype(np.float32, copy=False)
    x0 = data["recons_data"].astype(np.float32, copy=False)
    raw_shape = tuple(x.shape)

    if x.ndim == 4:
        x = x[:, None]
        x0 = x0[:, None]
    if x.shape != x0.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {x0.shape}")

    latent_bit = int(data["latent_bit"]) if "latent_bit" in data.files else int(latent_bit_arg)
    return x, x0, raw_shape, latent_bit


def make_residual(x, x0):
    x_mean = float(x.mean())
    scale = float(x.max() - x.min())
    if scale == 0:
        raise ValueError("zero data range")

    x_mean32 = np.float32(x_mean)
    scale32 = np.float32(scale)
    x_n = ((x - x_mean32) / scale32).astype(np.float32)
    x0_n = ((x0 - x_mean32) / scale32).astype(np.float32)
    residual = (x_n - x0_n).astype(np.float32)
    return residual, x0_n, x_mean, scale


def decode_path_sse(x, x0_n, residual, step, qbuf, ybuf, ebuf, x_mean, scale):
    step32 = np.float32(step)
    x_mean32 = np.float32(x_mean)
    scale32 = np.float32(scale)

    np.divide(residual, step32, out=qbuf)
    np.rint(qbuf, out=qbuf)

    np.multiply(qbuf, step32, out=ybuf)
    np.add(x0_n, ybuf, out=ybuf)
    np.multiply(ybuf, scale32, out=ybuf)
    np.add(ybuf, x_mean32, out=ybuf)

    np.subtract(x, ybuf, out=ebuf)
    np.divide(ebuf, scale32, out=ebuf)
    return float(np.sum(ebuf * ebuf))


def zero_sse(x, x0_n, ybuf, ebuf, x_mean, scale):
    x_mean32 = np.float32(x_mean)
    scale32 = np.float32(scale)

    np.multiply(x0_n, scale32, out=ybuf)
    np.add(ybuf, x_mean32, out=ybuf)
    np.subtract(x, ybuf, out=ebuf)
    np.divide(ebuf, scale32, out=ebuf)
    return float(np.sum(ebuf * ebuf))


def quantize_block(x, x0_n, residual, target, iters, x_mean, scale):
    qbuf = np.empty_like(residual, dtype=np.float32)
    ybuf = np.empty_like(residual, dtype=np.float32)
    ebuf = np.empty_like(residual, dtype=np.float32)
    target_sse = float(target) * float(target) * residual.size

    sse0 = zero_sse(x, x0_n, ybuf, ebuf, x_mean, scale)
    if sse0 <= target_sse:
        return 1.0, np.zeros(residual.shape, dtype=np.int32), sse0

    low = 0.0
    high = max(float(target) * np.sqrt(12.0), 1e-12)

    while decode_path_sse(x, x0_n, residual, high, qbuf, ybuf, ebuf, x_mean, scale) <= target_sse:
        low = high
        high *= 2.0

    for _ in range(iters):
        mid = 0.5 * (low + high)
        if decode_path_sse(x, x0_n, residual, mid, qbuf, ybuf, ebuf, x_mean, scale) <= target_sse:
            low = mid
        else:
            high = mid

    step = max(low, 1e-12)
    sse = decode_path_sse(x, x0_n, residual, step, qbuf, ybuf, ebuf, x_mean, scale)
    return float(step), qbuf.astype(np.int32), sse


def lorenzo_3d(q):
    q = np.ascontiguousarray(q)
    d = q.copy()
    d[1:, :, :] -= q[:-1, :, :]
    d[:, 1:, :] -= q[:, :-1, :]
    d[:, :, 1:] -= q[:, :, :-1]
    d[1:, 1:, :] += q[:-1, :-1, :]
    d[1:, :, 1:] += q[:-1, :, :-1]
    d[:, 1:, 1:] += q[:, :-1, :-1]
    d[1:, 1:, 1:] -= q[:-1, :-1, :-1]
    return d


def inverse_lorenzo_3d(d):
    q = np.cumsum(d, axis=-1)
    q = np.cumsum(q, axis=-2)
    q = np.cumsum(q, axis=-3)
    return q.astype(np.int32, copy=False)


def zigzag_encode(a):
    a = a.astype(np.int64, copy=False)
    u = np.where(a >= 0, a * 2, -2 * a - 1)
    if u.size and int(u.max()) > np.iinfo(np.uint32).max:
        raise OverflowError("zigzag overflow")
    return u.astype(np.uint32, copy=False)


def zigzag_decode(u):
    u = u.astype(np.uint64, copy=False)
    a = np.where((u & 1) == 0, u >> 1, -((u + 1) >> 1))
    return a.astype(np.int32, copy=False)


def bitplane_encode(u, level):
    flat = np.ascontiguousarray(u).reshape(-1)
    max_val = int(flat.max()) if flat.size else 0
    bit_count = max(1, max_val.bit_length())
    dtype = np.uint16 if max_val <= np.iinfo(np.uint16).max else np.uint32
    flat = flat.astype(dtype, copy=False)
    one = np.array(1, dtype=dtype)
    cctx = zstd.ZstdCompressor(level=level)
    streams = []

    for b in range(bit_count):
        bits = ((flat >> b) & one).astype(np.uint8, copy=False)
        streams.append(cctx.compress(np.packbits(bits, bitorder="little").tobytes()))

    return streams, bit_count


def bitplane_decode(streams, bit_count, shape):
    n = int(np.prod(shape))
    out = np.zeros(n, dtype=np.uint32)
    dctx = zstd.ZstdDecompressor()

    for b in range(bit_count):
        packed = np.frombuffer(dctx.decompress(streams[b]), dtype=np.uint8)
        bits = np.unpackbits(packed, bitorder="little")[:n].astype(np.uint32, copy=False)
        out |= bits << np.uint32(b)

    return out.reshape(shape)


def encode_block(x, residual, x0_n, sl, target, level, iters, x_mean, scale):
    b, c, t0, t1, h0, h1, w0, w1 = sl
    xb = np.ascontiguousarray(x[b, c, t0:t1, h0:h1, w0:w1])
    rb = np.ascontiguousarray(residual[b, c, t0:t1, h0:h1, w0:w1])
    x0b = np.ascontiguousarray(x0_n[b, c, t0:t1, h0:h1, w0:w1])

    step, q, sse = quantize_block(xb, x0b, rb, target, iters, x_mean, scale)
    d = lorenzo_3d(q)
    streams, bit_count = bitplane_encode(zigzag_encode(d), level)

    return {
        "slice": sl,
        "shape": tuple(d.shape),
        "step": step,
        "bit_count": bit_count,
        "streams": streams,
        "sse": sse,
        "num": int(d.size),
    }


def encode_residual(x, residual, x0_n, target, bt, bh, bw, level, iters, workers, x_mean, scale):
    slices = list(block_slices(residual.shape, bt, bh, bw))
    workers = cpu_count() if workers <= 0 else workers

    def job(sl):
        return encode_block(x, residual, x0_n, sl, target, level, iters, x_mean, scale)

    if workers == 1:
        blocks = [job(sl) for sl in slices]
    else:
        with ThreadPool(workers) as pool:
            blocks = list(pool.imap(job, slices))

    sse = float(sum(b["sse"] for b in blocks))
    num = int(sum(b["num"] for b in blocks))
    return blocks, float(np.sqrt(sse / max(num, 1)))


def decode_block(block, x0_n, x_mean, scale):
    b, c, t0, t1, h0, h1, w0, w1 = block["slice"]
    zz = bitplane_decode(block["streams"], block["bit_count"], block["shape"])
    q = inverse_lorenzo_3d(zigzag_decode(zz))
    y_n = x0_n[b, c, t0:t1, h0:h1, w0:w1] + q.astype(np.float32) * np.float32(block["step"])
    y = y_n * np.float32(scale) + np.float32(x_mean)
    return block["slice"], y.astype(np.float32, copy=False)


def decode_package(package, x0, workers):
    if package["format"] != FORMAT:
        raise ValueError("unsupported stream format")
    if tuple(x0.shape) != tuple(package["shape"]):
        raise ValueError(f"shape mismatch: {x0.shape} vs {package['shape']}")

    x_mean = float(package["x_mean"])
    scale = float(package["scale"])
    x0_n = ((x0 - np.float32(x_mean)) / np.float32(scale)).astype(np.float32)
    out = np.empty(tuple(package["shape"]), dtype=np.float32)
    workers = cpu_count() if workers <= 0 else workers

    def job(block):
        return decode_block(block, x0_n, x_mean, scale)

    iterator = map(job, package["blocks"]) if workers == 1 else ThreadPool(workers).imap(job, package["blocks"])
    for sl, y in iterator:
        b, c, t0, t1, h0, h1, w0, w1 = sl
        out[b, c, t0:t1, h0:h1, w0:w1] = y

    return out


def make_package(x, residual, x0_n, raw_shape, latent_bit, target, bt, bh, bw, level, iters, workers, x_mean, scale):
    blocks, encoded_nrmse = encode_residual(
        x, residual, x0_n, target, bt, bh, bw, level, iters, workers, x_mean, scale
    )

    return {
        "format": FORMAT,
        "target": float(target),
        "shape": tuple(x.shape),
        "raw_shape": tuple(raw_shape),
        "x_mean": float(x_mean),
        "scale": float(scale),
        "latent_bit": int(latent_bit),
        "original_bytes": int(x.nbytes),
        "block_size": (int(bt), int(bh), int(bw)),
        "encoded_nrmse": float(encoded_nrmse),
        "blocks": blocks,
    }


def package_bytes(package):
    return pickle.dumps(package, protocol=pickle.HIGHEST_PROTOCOL)


def save_package(package, path):
    with open(path, "wb") as f:
        f.write(package_bytes(package))


def load_package(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def total_cr(original_bytes, latent_bit, correction_bytes):
    latent_bytes = int(latent_bit) / 8.0
    total_bytes = latent_bytes + int(correction_bytes)
    return original_bytes / total_bytes if total_bytes > 0 else 0.0


def run_full(args):
    x, x0, raw_shape, latent_bit = load_data(args.path, args.latent_bit)
    residual, x0_n, x_mean, scale = make_residual(x, x0)
    package = make_package(
        x, residual, x0_n, raw_shape, latent_bit, args.nrmse,
        args.block_t, args.block_h, args.block_w, args.level, args.quant_iter, args.workers, x_mean, scale
    )
    correction_bytes = len(package_bytes(package))
    y = decode_package(package, x0, args.workers)
    print(f"Target NRMSE: {args.nrmse:.3e}")
    print(f"Final NRMSE: {nrmse(x, y, scale):.3e}")
    print(f"CR: {total_cr(x.nbytes, latent_bit, correction_bytes):.3f}")


def run_encode(args):
    x, x0, raw_shape, latent_bit = load_data(args.path, args.latent_bit)
    residual, x0_n, x_mean, scale = make_residual(x, x0)
    package = make_package(
        x, residual, x0_n, raw_shape, latent_bit, args.nrmse,
        args.block_t, args.block_h, args.block_w, args.level, args.quant_iter, args.workers, x_mean, scale
    )
    save_package(package, args.stream)
    correction_bytes = os.path.getsize(args.stream)
    print(f"stream: {args.stream}")
    print(f"shape: {package['raw_shape']}")
    print(f"blocks: {len(package['blocks'])}")
    print(f"target NRMSE: {args.nrmse:.3e}")
    print(f"encoded NRMSE: {package['encoded_nrmse']:.3e}")
    print(f"correction bytes: {correction_bytes}")
    print(f"latent bytes: {latent_bit / 8.0:.1f}")
    print(f"CR: {total_cr(x.nbytes, latent_bit, correction_bytes):.3f}")


def run_decode(args):
    package = load_package(args.stream)
    x, x0, _, _ = load_data(args.path, package.get("latent_bit", args.latent_bit))
    y = decode_package(package, x0, args.workers)
    print(f"NRMSE: {nrmse(x, y, float(package['scale'])):.3e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["evl", "encode", "decode"], required=True)
    p.add_argument("--path", required=True)
    p.add_argument("--stream", default="correction.lbrc")
    p.add_argument("--nrmse", type=float, default=1e-5)
    p.add_argument("--latent_bit", type=int, default=0)
    p.add_argument("--level", type=int, default=21)
    p.add_argument("--block_t", type=int, default=60)
    p.add_argument("--block_h", type=int, default=120)
    p.add_argument("--block_w", type=int, default=120)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--quant_iter", type=int, default=16)
    args = p.parse_args()

    if args.mode == "evl":
        run_full(args)
    elif args.mode == "encode":
        run_encode(args)
    else:
        run_decode(args)


if __name__ == "__main__":
    main()
