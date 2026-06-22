# GNN-DE: a learned Differential Evolution operator (reproducibility release)

Code and the deployed checkpoint for the paper
*"Learning the Differential Evolution Operator End-to-End: A Trajectory-Aware Sparse GATv2
Encoder for Zero-Shot Population-Based Optimization at Low Dimension"* (Mathematics, MDPI).

GNN-DE replaces hand-designed Differential Evolution (DE) operators with a neural network
that learns proposal geometry and adaptive hyperparameters from scratch. A sparse GATv2
backbone over a k-nearest-neighbour graph encodes the population, an all-to-all
donor-selection head picks the donor triple (pbest, r1, r2), and a single DE head produces
M displacements per parent per generation with Beta-sampled (F, CR).

Repository: https://github.com/ARCuet0/GNN-DE

## 1. Environment
- Python 3.12, PyTorch + `torch_geometric`, `numpy`, `scipy`, `opfunu`.
- `pip install -r requirements.txt`
- A CPU is enough for evaluation. A CUDA GPU is used for training and is faster for the
  GNN-DE forward pass.
- The BBOB and protein-docking transfer experiments (`eval_metabox/`) additionally require
  MetaBox / `metaevobox`.

## 2. Deployed model
- Checkpoint: `checkpoint/published_checkpoint.pth` (~625K parameters).
- Inference recipe: `--selection random_1pp --M-var 20 --per-m-donors`. The trained surrogate
  is no better than random selection at inference, so deployment uses random selection over
  M=20 proposals (see paper Section 4.4).

## 3. Benchmark integrity
- CEC2017 uses the corrected official implementation, selected with `TERSQ_BENCHMARK=official`
  (the legacy torch suite is non-standard). All reported CEC2017 numbers use `official`.
  The official shift/rotation/shuffle data ships under `cec2017/reference/input_data/`.

## 4. Reproduce the main results
- **CEC2017 per-function evaluation (D = 10/30/50/100):**
  ```
  TERSQ_BENCHMARK=official python eval_e7d_parallel.py \
    --ckpt checkpoint/published_checkpoint.pth \
    --selection random_1pp --M-var 20 --per-m-donors --D 10
  ```
- **Zero-shot BBOB and protein-docking transfer (MetaBox roster):** the runner and scoring
  scripts are under `eval_metabox/` (`run_metabox_bbob.py`, `run_protein_raw_ckpt.py`,
  `run_protein_rlde.py`, `score_aei.py`, `score_bbob_15inst.py`, `score_bbob_wilcoxon.py`).
  These require MetaBox and regenerate the raw per-seed result files locally.

The bulky per-seed result JSONs and the paper PDF/figures are not bundled in this code
release. They can be regenerated with the scripts above (and will be archived separately;
the Zenodo DOI will be added here on publication: **TODO**).

## 5. Train from scratch
```
python train_distributed.py --operators k1 --gate-type surrogate \
  --surrogate-M 50 --surrogate-m-final 5 --per-m-donors \
  --contrafactual --contra-weight 0.02 \
  --fcr-beta-weight 0.3 --fcr-oracle-mode from_m \
  --improvement-weight 0.3 --bptt-w-init 800 --bptt-chunk 9 \
  --D 10 --N 50 --budget-mult 1000 --steps 5000
```
Multi-GPU: `torchrun --nproc_per_node=4 train_distributed.py [same flags]`.

## 6. Seeds and granularity
All paper evaluations use 51 seeds. The evaluation scripts store every per-seed final value,
function-evaluation count and full cost curve, so AEI / Wilcoxon / Friedman statistics can be
recomputed under any definition from the regenerated result files.

## 7. Notes and honest caveats (mirrored from the paper)
- Competitive in the low-dimension regime the operator targets (D <= 30). Rank degrades at
  D >= 50 (last of six at D = 100, matched).
- Ablation toolkit deltas are 3-replication, directional estimates.
- The protein-docking advantage over RLDE-AFL is systematic (28/28 complexes) but small
  (about 0.04%).
- The deployed checkpoint was trained on the legacy CEC2017 suite. All reported numbers are
  re-evaluated on the corrected official benchmark (`TERSQ_BENCHMARK=official`).

## License
MIT. See `LICENSE`.
