# Results

Machine: Linux x86_64, 2 logical CPUs · Python 3.13.18 · NumPy 2.5.3 · energy measured via **estimate**

| Technique | Accuracy | Train energy (J) | Inference (mJ / 1k preds) | Size (KB) | FLOPs / sample | Trainable params | Notes |
|---|---|---|---|---|---|---|---|
| Baseline (float64, fixed epochs) | 0.936 | 960.8 | 1470.69 | 9232.0 | 2,361,856 | 1,181,701 | 40 epochs |
| Reduced precision (float32) | 0.936 | 477.8 | 686.54 | 4616.0 | 2,361,856 | 1,181,701 | 40 epochs |
| float32 + early stopping | 0.933 | 81.3 | 600.74 | 4616.0 | 2,361,856 | 1,181,701 | stopped after 7 of 40 epochs |
| Unstructured pruning (80%) | 0.919 | 35.4 | 676.22 | 1861.2 | 474,418 | - | +3 fine-tune epochs; size assumes CSR storage |
| Unstructured pruning (95%) | 0.880 | 39.9 | 722.72 | 478.8 | 120,526 | - | +3 fine-tune epochs; size assumes CSR storage |
| Structured pruning (keep 25% neurons) | 0.911 | 16.6 | 217.44 | 1058.0 | 541,312 | - | +3 fine-tune epochs; hidden [128, 64] |
| int8 weight quantisation | 0.933 | - | 699.94 | 1159.3 | 2,361,856 | - | post-training, no retraining |
| Small model, trained alone | 0.913 | 19.7 | 100.69 | 513.5 | 262,784 | 131,461 | 2048-64-5, same data |
| Knowledge distillation (student) | 0.916 | 34.1 | 100.98 | 513.5 | 262,784 | 131,461 | 2048-64-5, T=4, alpha=0.7 |
| Adaptation: full fine-tuning | 0.936 | 51.6 | 714.94 | 4615.0 | 2,361,344 | 1,181,444 | 4 new classes |
| Adaptation: LoRA (rank 8) | 0.941 | 19.0 | 791.39 | 4615.0 | 2,414,592 | 27,652 | base weights frozen |

## Carbon-aware scheduling

Running the baseline training job (2h window) at 17:00 vs the cleanest window (03:00) on the example grid profile cuts emissions by **52%** for exactly the same energy.
