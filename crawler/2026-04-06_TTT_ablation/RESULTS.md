# Results: TTT_ablation

Date: 2026-04-06
Track: crawler
Branch: refined-focus

## Gate status (seed=444)

### TTT-00_ouro_ctrl (PASS)

- model_params: `26270292`
- raw_bpb: `1.2640`
- int6_sw_bpb: `1.24372982`
- step_avg_ms: `110.53`
- artifact_bytes: `15630351`

### TTT-01_ouro_ttt (FAIL, strike 1)

Observed error during warmup/train forward under compile:

- `RuntimeError: torch.compile with aot_autograd does not currently support double backward`

Resolution applied:

- disable `torch.compile` automatically when `TTT_DIM>0` in `Hyperparameters.from_env`.

### TTT-01_ouro_ttt (FAIL, strike 2)

Observed error during validation (`eval_val`):

- `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

Root cause:

- TTT performs `torch.autograd.grad(...)` in forward for its inner adaptation step.
- Validation path still ran under grad-disabled context, so inner-step gradient computation failed.

Resolution applied:

- `eval_val`: use `torch.enable_grad()` when TTT is active and compute CE via logits path.
- `eval_val_sliding`: same grad-context fix; also force `batch_seqs=1` for TTT to avoid cross-window coupling.

## Next run

Run arm 1 again from this branch after pulling latest `refined-focus`:

`SEED=444 NPROC_PER_NODE=8 bash crawler/2026-04-06_TTT_ablation/gate.sh`

(or run only the TTT arm command used by your launcher).
