#!/usr/bin/env python3
"""Measure the energy/accuracy trade-off of techniques from the survey.

    python run_experiments.py              # full run (a few minutes on a laptop)
    python run_experiments.py --quick      # smaller model, fewer epochs

Writes results/results.md, results/results.json and results/tradeoff.png.
"""

import argparse
import copy
import json
import os
import platform

import numpy as np

from greenai import carbon, data
from greenai.compress import QuantisedMLP, magnitude_prune, sparse_size_bytes, structured_prune
from greenai.energy import EnergyTracker, DEFAULT_INTENSITY
from greenai.mlp import MLP, softmax, train

OUT = "results"


def inference_cost(model, X, n_predictions):
    reps = max(1, n_predictions // len(X))
    Xr = np.tile(X, (reps, 1)).astype(np.float32 if model.dtype != np.float64 else np.float64)
    with EnergyTracker() as m:
        model.predict(Xr)
    return 1000 * m.energy_j / len(Xr) * 1000, m     # mJ per 1k predictions


def row(name, stage, model, test, n_inf, train_m=None, size=None, trainable=None, note=""):
    acc = model.accuracy(*test)
    inf_mj, inf_m = inference_cost(model, test[0], n_inf)
    r = {
        "technique": name,
        "stage": stage,
        "accuracy": round(acc, 4),
        "train_energy_j": round(train_m.energy_j, 2) if train_m else None,
        "train_wall_s": round(train_m.wall_s, 2) if train_m else None,
        "infer_mj_per_1k": round(inf_mj, 2),
        "size_kb": round((size if size is not None else model.size_bytes()) / 1024, 1),
        "flops_per_sample": model.flops_per_sample(),
        "trainable_params": trainable,
        "method": (train_m or inf_m).method,
        "note": note,
    }
    print(f"  {name:38s} acc {acc:.3f} | train {r['train_energy_j'] or 0:8.1f} J "
          f"| infer {inf_mj:8.2f} mJ/1k | {r['size_kb']:8.1f} KB")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    hidden = [256, 128] if args.quick else [512, 256]
    epochs = 15 if args.quick else 40
    n_inf = 20_000 if args.quick else 100_000
    seed = args.seed

    print("building dataset from the Python standard library ...")
    tr, va, te = data.load(data.PRETRAIN_PACKAGES, seed=seed)
    print(f"  {len(tr[1])} train / {len(va[1])} val / {len(te[1])} test functions, "
          f"{len(data.PRETRAIN_PACKAGES)} classes")
    sizes = [tr[0].shape[1], *hidden, len(data.PRETRAIN_PACKAGES)]
    results = []

    # --- training-side techniques -------------------------------------------
    print("\ntraining")
    base = MLP(sizes, seed=seed, dtype=np.float64)
    with EnergyTracker("baseline") as m_base:
        train(base, *tr, epochs=epochs, seed=seed)
    results.append(row("Baseline (float64, fixed epochs)", "training", base, te, n_inf, m_base,
                       trainable=base.n_trainable(), note=f"{epochs} epochs"))

    fp32 = MLP(sizes, seed=seed, dtype=np.float32)
    with EnergyTracker("fp32") as m_fp32:
        train(fp32, *tr, epochs=epochs, seed=seed)
    results.append(row("Reduced precision (float32)", "training", fp32, te, n_inf, m_fp32,
                       trainable=fp32.n_trainable(), note=f"{epochs} epochs"))

    es = MLP(sizes, seed=seed, dtype=np.float32)
    with EnergyTracker("early stopping") as m_es:
        ran = train(es, *tr, epochs=epochs, seed=seed, val=va, patience=3)
    results.append(row("float32 + early stopping", "training", es, te, n_inf, m_es,
                       trainable=es.n_trainable(), note=f"stopped after {ran} of {epochs} epochs"))
    teacher = es

    # --- deployment-side techniques -------------------------------------------
    print("\ndeployment (starting from the early-stopped float32 model)")
    ft_epochs = 3

    for sparsity in (0.8, 0.95):
        pruned = magnitude_prune(teacher, sparsity)
        with EnergyTracker() as m:
            train(pruned, *tr, epochs=ft_epochs, seed=seed, lr=5e-4)
        results.append(row(f"Unstructured pruning ({int(sparsity * 100)}%)", "deployment", pruned, te,
                           n_inf, m, size=sparse_size_bytes(pruned),
                           note=f"+{ft_epochs} fine-tune epochs; size assumes CSR storage"))

    sp = structured_prune(teacher, 0.25)
    with EnergyTracker() as m:
        train(sp, *tr, epochs=ft_epochs, seed=seed, lr=5e-4)
    results.append(row("Structured pruning (keep 25% neurons)", "deployment", sp, te, n_inf, m,
                       note=f"+{ft_epochs} fine-tune epochs; hidden {sp.sizes[1:-1]}"))

    q = QuantisedMLP(teacher, bits=8)
    q.dtype = np.float32
    results.append(row("int8 weight quantisation", "deployment", q, te, n_inf,
                       note="post-training, no retraining"))

    student_sizes = [sizes[0], 64, sizes[-1]]
    plain = MLP(student_sizes, seed=seed + 1)
    with EnergyTracker() as m:
        train(plain, *tr, epochs=epochs, seed=seed, val=va, patience=3)
    results.append(row("Small model, trained alone", "deployment", plain, te, n_inf, m,
                       trainable=plain.n_trainable(), note="2048-64-5, same data"))

    teacher_probs = softmax(teacher.forward(tr[0]), T=4.0)
    student = MLP(student_sizes, seed=seed + 1)
    with EnergyTracker() as m:
        train(student, *tr, epochs=epochs, seed=seed, val=va, patience=3, teacher_probs=teacher_probs)
    results.append(row("Knowledge distillation (student)", "deployment", student, te, n_inf, m,
                       trainable=student.n_trainable(), note="2048-64-5, T=4, alpha=0.7"))

    # --- adaptation: full fine-tune vs LoRA -------------------------------------
    print("\nadapting to a new task (different stdlib packages)")
    atr, ava, ate = data.load(data.ADAPT_PACKAGES, seed=seed)
    n_new = len(data.ADAPT_PACKAGES)

    full = copy.deepcopy(teacher)
    full.replace_head(n_new, seed=seed)
    with EnergyTracker() as m_full:
        train(full, *atr, epochs=epochs, seed=seed, val=ava, patience=3)
    results.append(row("Adaptation: full fine-tuning", "adaptation", full, ate, n_inf, m_full,
                       trainable=full.n_trainable(), note=f"{n_new} new classes"))

    lora = copy.deepcopy(teacher)
    lora.replace_head(n_new, seed=seed)
    lora.add_lora(rank=8, alpha=16, seed=seed)
    with EnergyTracker() as m_lora:
        train(lora, *atr, epochs=epochs, seed=seed, val=ava, patience=3, lr=2e-3)
    results.append(row("Adaptation: LoRA (rank 8)", "adaptation", lora, ate, n_inf, m_lora,
                       trainable=lora.n_trainable(), note="base weights frozen"))

    # --- infrastructure: carbon-aware scheduling --------------------------------
    sched = carbon.compare(m_base.kwh, hours=2, naive_start=17)

    # --- write everything out ---------------------------------------------------
    os.makedirs(OUT, exist_ok=True)
    meta = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} logical CPUs",
        "energy_method": results[0]["method"],
        "grid_intensity_g_per_kwh": DEFAULT_INTENSITY,
        "quick": args.quick,
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"meta": meta, "results": results, "carbon_scheduling": sched}, f, indent=2)
    write_markdown(results, sched, meta)
    plot(results)
    print(f"\nwrote {OUT}/results.md, results.json, tradeoff.png")


def write_markdown(results, sched, meta):
    lines = [
        "# Results",
        "",
        f"Machine: {meta['machine']} · Python {meta['python']} · NumPy {meta['numpy']} · "
        f"energy measured via **{meta['energy_method']}**",
        "",
        "| Technique | Accuracy | Train energy (J) | Inference (mJ / 1k preds) | Size (KB) | FLOPs / sample | Trainable params | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        te = "-" if r["train_energy_j"] is None else f"{r['train_energy_j']:.1f}"
        tp = "-" if r["trainable_params"] is None else f"{r['trainable_params']:,}"
        lines.append(f"| {r['technique']} | {r['accuracy']:.3f} | {te} | {r['infer_mj_per_1k']:.2f} | "
                     f"{r['size_kb']:.1f} | {r['flops_per_sample']:,} | {tp} | {r['note']} |")
    lines += [
        "",
        "## Carbon-aware scheduling",
        "",
        f"Running the baseline training job (2h window) at {sched['naive_start']:02d}:00 vs the cleanest "
        f"window ({sched['best_start']:02d}:00) on the example grid profile cuts emissions by "
        f"**{sched['saving_pct']:.0f}%** for exactly the same energy.",
        "",
    ]
    with open(os.path.join(OUT, "results.md"), "w") as f:
        f.write("\n".join(lines))


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = [r for r in results if r["stage"] != "adaptation"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([r["infer_mj_per_1k"] for r in pts], [r["accuracy"] for r in pts],
               s=60, color="#2a78d6", edgecolor="white", linewidth=2, zorder=3)
    # short labels with hand-placed offsets so the cluster near the top stays readable
    short = {
        "Baseline (float64, fixed epochs)": ("float64 baseline", (-8, 8), "right"),
        "Reduced precision (float32)": ("float32", (0, 9), "center"),
        "float32 + early stopping": ("early stopping", (-8, -12), "right"),
        "int8 weight quantisation": ("int8", (8, -12), "left"),
        "Unstructured pruning (80%)": ("80% pruned", (8, -3), "left"),
        "Unstructured pruning (95%)": ("95% pruned", (8, -3), "left"),
        "Structured pruning (keep 25% neurons)": ("structured pruning", (8, -3), "left"),
        "Small model, trained alone": ("small model", (8, -10), "left"),
        "Knowledge distillation (student)": ("distilled student", (8, 4), "left"),
    }
    for r in pts:
        text, offset, ha = short.get(r["technique"], (r["technique"], (8, -3), "left"))
        ax.annotate(text, (r["infer_mj_per_1k"], r["accuracy"]), textcoords="offset points",
                    xytext=offset, ha=ha, fontsize=9, color="#333333")
    ax.set_xscale("log")
    xs = [r["infer_mj_per_1k"] for r in pts]
    ax.set_xlim(min(xs) / 1.6, max(xs) * 1.6)
    ys = [r["accuracy"] for r in pts]
    ax.set_ylim(min(ys) - 0.005, max(ys) + 0.006)
    ax.set_xlabel("Inference energy, mJ per 1,000 predictions (log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs inference energy", loc="left")
    ax.grid(True, color="#e5e5e5", linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tradeoff.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
