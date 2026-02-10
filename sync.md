# DREAM-C2L Sync Context
**Last updated:** 2026-02-10 08:36 PST

## Active Jobs
| Job ID | Name | Status | Node | Notes |
|--------|------|--------|------|-------|
| 6210056 | hyperparam | RUNNING | b11-13 | Phase 1 hyperparameter search |
| 6195866 | ablation_full | CANCELLED | b01-15 | Old ablation, killed (had bugs) |

## What's Running: Hyperparameter Search (Job 6210056)
**Script:** `scripts/run_hyperparam_search.sh`
**Purpose:** Find optimal (data_size, epochs) for std_cd before running the full ablation.

**Grid:**
- Data sizes: 50, 100, 200, 500, 1000 samples
- Epochs: 1, 3, 5, 7, 9 (trains for 9, saves checkpoints at every epoch)
- 25 total checkpoints to evaluate

**Pipeline flow:**
1. Phase 1A: Generate 100 shared benign queries for utility eval
2. Phase 1B: Generate 1000 std_cd samples, subsample to each size
3. Phase 2: Train 5 models × 9 epochs with `--save_every_epoch` callback
4. Phase 3: Evaluate safety (HarmBench RR), utility (Win%), KL at epochs 1,3,5,7,9
5. Phase 4: Collect all 25 data points into `results/hyperparam/summary.json`

**Cluster paths:**
- Data: `/project2/jessetho_1732/zizhaoh/DREAM-C2L/dataset/hyperparam/`
- Models: `/project2/jessetho_1732/zizhaoh/DREAM-C2L/models/hyperparam/`
- Results: `/project2/jessetho_1732/zizhaoh/DREAM-C2L/results/hyperparam/`
- Logs: `slurm_hyperparam_6210056.out/.err`

**GPU constraint:** `--constraint="a100|a40"` added after CUDA kernel error on older node e21-13.

## Code Changes Made This Session

### 1. `scripts/1_train.py`
- Added `--save_every_epoch` flag
- Added `EpochSaveCallback` (TrainerCallback) that saves adapter to `epoch_N/` dirs at end of each epoch
- Allows evaluating any intermediate checkpoint without retraining

### 2. `scripts/run_hyperparam_search.sh` (NEW)
- Complete SLURM pipeline for hyperparameter search
- Generates data, trains, evaluates, collects results

### 3. `paper/latex/acl_latex.tex`
- Ablation table filled with modes 1-5 results + ratio_1_1
- Added colored delta macros: `\up`, `\dn`, `\badup`, `\gooddn`
- KL arrows fixed: red ↑ for increasing KL (bad), green ↓ for decreasing (good)

## Ablation Results So Far (Old Job — for reference only)
These results had a **critical bug**: rejection sampling filtered ALL positive queries to 0 (the 1.5B model was too small to self-classify). Modes 3-5 trained with ~0 safety data.

| Mode | RR↑ | Win%↑ | KL↓ | Notes |
|------|-----|-------|-----|-------|
| Base (No Ctx) | 70.0 | 50.0 | 0.0 | Baseline |
| In-Context | 94.8 | -- | -- | Oracle upper bound |
| (1) Std. CD | 66.2 | 6.7 | 0.053 | Random queries, safety degrades |
| (2) Associative | 70.0 | 3.3 | 0.011 | Context-related queries recover safety |
| (3) Dual | 51.8 | 0.0 | 0.121 | **BUG**: 0 pos samples after rej. sampling |
| (4) Rejection | 49.8 | 10.0 | 0.105 | **BUG**: same issue |
| (5) Trigger | 51.0 | 16.7 | 0.103 | **BUG**: same issue |
| Ratio 1:1 | 50.5 | 6.7 | 0.100 | Nearly identical to (5), redundant |

**Root cause:** `Rejection sampling: Q+ 19->0, Q- 100->100` — the 1.5B model classified all its own harmful queries as benign.

## Next Steps (When Resuming)
1. **Monitor job 6210056** — check `slurm_hyperparam_6210056.out`
2. **When complete:** Download `results/hyperparam/summary.json`
3. **Plot 2 figures:**
   - Safety (RR) vs Epochs — 5 curves (one per data size)
   - Utility (Win%) vs Epochs — 5 curves (one per data size)
4. **Select optimal (N, epochs)** — best safety-utility trade-off
5. **Fix rejection sampling** — use a larger judge model (e.g., Qwen2.5-7B) or bypass self-classification
6. **Run full ablation (Phase 2)** with the selected hyperparameters
7. **Update paper** with final ablation results and figures

## Key Design Decisions
- **std_cd used for hyperparameter search** because it's the simplest mode (no dependencies)
- **All ablation modes must use the SAME data size** once optimal is found
- **EpochSaveCallback** saves full adapter copies (not just checkpoints) so each is independently loadable
- **Benign queries generated once** and shared across all evaluations for consistency

## Files to Check
```
ssh zizhaoh@discovery.usc.edu
cd /project2/jessetho_1732/zizhaoh/DREAM-C2L

# Check job status
squeue -u zizhaoh

# Check progress
tail -30 slurm_hyperparam_6210056.out
tail -10 slurm_hyperparam_6210056.err

# Check if results ready
ls results/hyperparam/summary.json
cat results/hyperparam/summary.json
```
