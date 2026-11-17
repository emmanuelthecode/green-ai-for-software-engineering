"""A small, self-contained code-classification dataset.

Task: given the source of a Python function, predict which standard-library
component it came from (asyncio, email, http, ...). This stands in for the
"which module / which owner / which component" style of classification that
shows up in real SE tooling (bug triage, code search, ownership routing).

Everything comes from the Python install on your machine, so there's nothing
to download. Exact counts vary a little between Python versions.
"""

import ast
import os
import re
import sysconfig
import zlib

import numpy as np

# Two groups so the LoRA experiment can pretrain on one and adapt to the other.
PRETRAIN_PACKAGES = ["asyncio", "email", "http", "xml", "unittest"]
ADAPT_PACKAGES = ["multiprocessing", "importlib", "urllib", "logging"]

_SPLIT_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def stdlib_dir():
    return sysconfig.get_paths()["stdlib"]


def tokens(source):
    """Identifiers and keywords, with snake_case and camelCase split up."""
    out = []
    for ident in _IDENT.findall(source):
        out.append(ident.lower())
        for part in _SPLIT_CAMEL.sub("_", ident).lower().split("_"):
            if part and part != ident.lower():
                out.append(part)
    return out


def functions_in(path):
    try:
        with open(path, encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src)
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return []

    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seg = ast.get_source_segment(src, node)
            if seg and seg.count("\n") >= 2:     # skip one-liners
                funcs.append(seg)
    return funcs


def collect(packages, root=None):
    root = root or stdlib_dir()
    samples, labels = [], []
    for label, pkg in enumerate(packages):
        pkg_dir = os.path.join(root, pkg)
        for dirpath, dirnames, filenames in os.walk(pkg_dir):
            dirnames[:] = [d for d in dirnames if d not in ("test", "tests", "__pycache__")]
            for fn in sorted(filenames):
                if fn.endswith(".py"):
                    for src in functions_in(os.path.join(dirpath, fn)):
                        samples.append(src)
                        labels.append(label)
    return samples, np.array(labels)


def vectorise(samples, dim=2048):
    """Hashed bag-of-tokens with log scaling and L2 normalisation."""
    X = np.zeros((len(samples), dim), dtype=np.float64)
    for i, src in enumerate(samples):
        for tok in tokens(src):
            X[i, zlib.crc32(tok.encode()) % dim] += 1.0
    X = np.log1p(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)


def split(X, y, seed=0, val=0.15, test=0.15):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test, n_val = int(len(y) * test), int(len(y) * val)
    te, va, tr = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def load(packages=PRETRAIN_PACKAGES, dim=2048, seed=0):
    samples, y = collect(packages)
    if len(samples) == 0:
        raise RuntimeError(f"no Python sources found under {stdlib_dir()}")
    return split(vectorise(samples, dim), y, seed)
