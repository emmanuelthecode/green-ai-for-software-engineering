"""A small multi-layer perceptron written in plain NumPy.

I wrote this from scratch rather than using PyTorch so that every technique
from the survey is visible in a few lines of code: pruning is a mask,
quantisation is a rescale-and-round, LoRA is two extra matrices, and so on.
It also means the whole project runs on a laptop CPU with only NumPy.
"""

import numpy as np


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes, seed=0, dtype=np.float32):
        self.sizes = list(sizes)
        self.dtype = dtype
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            self.W.append((rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(dtype))
            self.b.append(np.zeros(fan_out, dtype=dtype))

        self.masks = None          # pruning masks, one per weight matrix
        self.lora = {}             # layer index -> (A, B)
        self.lora_scale = 1.0
        self.frozen = False        # True while only LoRA + head are trained

    # ---- bookkeeping ---------------------------------------------------

    def n_params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def n_nonzero(self):
        return sum(int(np.count_nonzero(w)) + b.size for w, b in zip(self.W, self.b))

    def n_trainable(self):
        return sum(p.size for p in self._trainable())

    def flops_per_sample(self):
        """Multiply-adds x2 for one forward pass, counting only non-zero weights."""
        f = sum(2 * int(np.count_nonzero(w)) for w in self.W)
        for A, B in self.lora.values():
            f += 2 * (A.size + B.size)
        return f

    def size_bytes(self):
        return sum(w.nbytes + b.nbytes for w, b in zip(self.W, self.b))

    def astype(self, dtype):
        self.dtype = dtype
        self.W = [w.astype(dtype) for w in self.W]
        self.b = [b.astype(dtype) for b in self.b]
        return self

    # ---- forward / backward ----------------------------------------------

    def forward(self, X, cache=False):
        h = X.astype(self.dtype, copy=False)
        hs, zs = [h], []
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i in self.lora:
                A, B = self.lora[i]
                z = z + (h @ A) @ B * self.lora_scale
            zs.append(z)
            h = z if i == last else np.maximum(z, 0)
            hs.append(h)
        return (h, hs, zs) if cache else h

    def predict(self, X, batch=1024):
        out = [self.forward(X[i:i + batch]).argmax(axis=1) for i in range(0, len(X), batch)]
        return np.concatenate(out)

    def accuracy(self, X, y):
        return float((self.predict(X) == y).mean())

    def _trainable(self):
        if not self.frozen:
            return self.W + self.b
        params = []
        for A, B in self.lora.values():
            params += [A, B]
        return params + [self.W[-1], self.b[-1]]      # new task head is always trained

    def gradients(self, X, y, teacher_probs=None, T=4.0, alpha=0.7):
        """Cross-entropy gradients, optionally mixed with a distillation loss."""
        logits, hs, zs = self.forward(X, cache=True)
        n = len(y)
        onehot = np.zeros_like(logits)
        onehot[np.arange(n), y] = 1

        p = softmax(logits)
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        dz = (p - onehot) / n

        if teacher_probs is not None:
            # Hinton et al. (2015): match the teacher's softened distribution.
            # The T^2 factor keeps gradient magnitudes comparable across T.
            ps = softmax(logits, T)
            kd = (ps - teacher_probs) * T / n
            dz = (1 - alpha) * dz + alpha * kd
            loss = (1 - alpha) * loss + alpha * T * T * float(
                (teacher_probs * (np.log(teacher_probs + 1e-12) - np.log(ps + 1e-12))).sum(axis=1).mean())

        gW, gb, gA, gB = [None] * len(self.W), [None] * len(self.W), {}, {}
        for i in reversed(range(len(self.W))):
            h = hs[i]
            if not self.frozen or i == len(self.W) - 1:
                # frozen base weights don't need gradients - that's the
                # whole point of LoRA, so don't pay for them
                gW[i] = h.T @ dz
                gb[i] = dz.sum(axis=0)
            dh = dz @ self.W[i].T
            if i in self.lora:
                A, B = self.lora[i]
                hA = h @ A
                gB[i] = hA.T @ dz * self.lora_scale
                dhA = dz @ B.T * self.lora_scale
                gA[i] = h.T @ dhA
                dh = dh + dhA @ A.T
            if i > 0:
                dz = dh * (zs[i - 1] > 0)

        if self.masks is not None and not self.frozen:
            gW = [g * m for g, m in zip(gW, self.masks)]

        if not self.frozen:
            grads = gW + gb
        else:
            grads = []
            for i in self.lora:
                grads += [gA[i], gB[i]]
            grads += [gW[-1], gb[-1]]
        return float(loss), grads

    # ---- techniques that change the model ----------------------------------

    def add_lora(self, rank=8, alpha=16, seed=0, layers=None):
        """Freeze the base weights and attach low-rank adapters (Hu et al., 2021)."""
        rng = np.random.default_rng(seed)
        layers = range(len(self.W) - 1) if layers is None else layers
        for i in layers:
            fan_in, fan_out = self.W[i].shape
            A = (rng.standard_normal((fan_in, rank)) / np.sqrt(fan_in)).astype(self.dtype)
            B = np.zeros((rank, fan_out), dtype=self.dtype)     # starts as a no-op
            self.lora[i] = (A, B)
        self.lora_scale = alpha / rank
        self.frozen = True

    def replace_head(self, n_classes, seed=0):
        rng = np.random.default_rng(seed)
        fan_in = self.W[-1].shape[0]
        self.W[-1] = (rng.standard_normal((fan_in, n_classes)) * np.sqrt(2.0 / fan_in)).astype(self.dtype)
        self.b[-1] = np.zeros(n_classes, dtype=self.dtype)
        self.sizes[-1] = n_classes


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mhat = m / (1 - self.b1 ** self.t)
            vhat = v / (1 - self.b2 ** self.t)
            p -= (self.lr * mhat / (np.sqrt(vhat) + self.eps)).astype(p.dtype)


def train(model, X, y, epochs=30, batch=64, lr=1e-3, seed=0,
          val=None, patience=None, teacher_probs=None, T=4.0, alpha=0.7):
    """Mini-batch Adam. With `val` and `patience` set, stops early and rolls
    back to the best weights seen on the validation set.

    Returns the number of epochs actually run.
    """
    rng = np.random.default_rng(seed)
    opt = Adam(model._trainable(), lr=lr)
    best_acc, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y))
        for i in range(0, len(y), batch):
            idx = order[i:i + batch]
            tp = None if teacher_probs is None else teacher_probs[idx]
            _, grads = model.gradients(X[idx], y[idx], tp, T, alpha)
            opt.step(grads)
            if model.masks is not None:
                for w, m in zip(model.W, model.masks):
                    w *= m

        if val is not None and patience is not None:
            acc = model.accuracy(*val)
            if acc > best_acc:
                best_acc, bad_epochs = acc, 0
                best_state = ([w.copy() for w in model.W], [b.copy() for b in model.b],
                              {k: (a.copy(), b.copy()) for k, (a, b) in model.lora.items()})
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if best_state is not None:
        W, b, lora = best_state
        for i in range(len(W)):
            model.W[i][...] = W[i]
            model.b[i][...] = b[i]
        for k, (a, bb) in lora.items():
            model.lora[k][0][...] = a
            model.lora[k][1][...] = bb
    return epoch
