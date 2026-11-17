"""Post-training model optimisation: pruning and quantisation (Section 4.1-4.2)."""

import copy

import numpy as np


def magnitude_prune(model, sparsity):
    """Unstructured pruning: zero the smallest |w| across all hidden layers.

    The output layer is left alone, which is common practice since it's small
    and very sensitive. Returns a pruned copy with masks set, so fine-tuning
    keeps the zeros at zero.
    """
    m = copy.deepcopy(model)
    hidden = m.W[:-1]
    all_w = np.concatenate([np.abs(w).ravel() for w in hidden])
    threshold = np.quantile(all_w, sparsity)
    m.masks = [(np.abs(w) > threshold).astype(w.dtype) for w in hidden] + [np.ones_like(m.W[-1])]
    m.W = [w * mask for w, mask in zip(m.W, m.masks)]
    return m


def structured_prune(model, keep_fraction):
    """Structured pruning: drop whole hidden neurons with the smallest
    incoming-weight norm. Unlike unstructured pruning this really does make
    the matrices smaller, so dense BLAS gets faster too."""
    m = copy.deepcopy(model)
    for i in range(len(m.W) - 1):
        norms = np.linalg.norm(m.W[i], axis=0)
        k = max(1, int(round(len(norms) * keep_fraction)))
        keep = np.sort(np.argsort(norms)[-k:])
        m.W[i] = m.W[i][:, keep]
        m.b[i] = m.b[i][keep]
        m.W[i + 1] = m.W[i + 1][keep, :]
        m.sizes[i + 1] = k
    m.masks = None
    return m


def sparse_size_bytes(model, index_bytes=4):
    """Storage if pruned matrices were kept in CSR format."""
    total = 0
    for w, b in zip(model.W, model.b):
        nnz = int(np.count_nonzero(w))
        csr = nnz * (w.itemsize + index_bytes) + (w.shape[0] + 1) * index_bytes
        total += min(csr, w.nbytes) + b.nbytes
    return total


class QuantisedMLP:
    """Weight-only symmetric int8 quantisation, one scale per output channel.

    Weights are stored as int8 (4x smaller than float32) and dequantised on
    the fly. On a GPU or a CPU with int8 kernels the matmul itself would run
    in int8 too; NumPy has no fast int8 GEMM, so here the win is memory and
    model size, not speed.
    """

    def __init__(self, model, bits=8):
        qmax = 2 ** (bits - 1) - 1
        self.dtype = model.dtype
        self.scales, self.qW = [], []
        for w in model.W:
            scale = np.abs(w).max(axis=0) / qmax
            scale[scale == 0] = 1.0
            self.scales.append(scale.astype(np.float32))
            self.qW.append(np.clip(np.round(w / scale), -qmax, qmax).astype(np.int8))
        self.b = [b.copy() for b in model.b]
        self.bits = bits

    def forward(self, X):
        h = X.astype(np.float32, copy=False)
        last = len(self.qW) - 1
        for i, (q, s, b) in enumerate(zip(self.qW, self.scales, self.b)):
            z = (h @ q.astype(np.float32)) * s + b
            h = z if i == last else np.maximum(z, 0)
        return h

    def predict(self, X, batch=1024):
        return np.concatenate([self.forward(X[i:i + batch]).argmax(axis=1)
                               for i in range(0, len(X), batch)])

    def accuracy(self, X, y):
        return float((self.predict(X) == y).mean())

    def size_bytes(self):
        return sum(q.size * self.bits // 8 + s.nbytes + b.nbytes
                   for q, s, b in zip(self.qW, self.scales, self.b))

    def flops_per_sample(self):
        return sum(2 * q.size for q in self.qW)
