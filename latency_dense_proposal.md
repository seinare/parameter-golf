# Latency Baseline Proposal

## Arterial Pure Profile Target

This is the clean latency target derived from the current arterial run, with XSA and U-Net residual slots removed:

- `NUM_LAYERS=10`
- `NUM_ARTERIES=2`
- `MODEL_DIM=768`
- `artery_dim=384`
- `NUM_HEADS=3`
- `NUM_KV_HEADS=3`
- `MLP_MULT=2`
- `MIXER_DIM=192`
- `MIXER_HEADS=3`
- `ATTN_WINDOWS=1024,4096`
- `TRAIN_SEQ_LEN=16384`
- `TRAIN_BATCH_TOKENS=524288`
- `XSA_ARTERIES=`
- `MIXER_UNET_RESIDUAL_KV=0`
- `MIXER_SLOT_ROPE=0`

Expected parameter count is approximately the same as the previous full run, because XSA, U-Net residual slots, and slot RoPE do not add trainable parameters:

- arterial params: `27,363,132`

## Same-Parameter Dense Proposal

Use a single dense stream with width adjusted to match the arterial model's parameter count:

- `NUM_LAYERS=10`
- `NUM_ARTERIES=1`
- `MODEL_DIM=576`
- `NUM_HEADS=6`
- `NUM_KV_HEADS=6`
- `MLP_MULT=2.03125`
- `ATTN_WINDOWS=4096`
- `TRAIN_SEQ_LEN=16384`
- `TRAIN_BATCH_TOKENS=524288`

With the current parameterization, `MLP_MULT=2.03125` gives an MLP hidden size of `1170`, resulting in:

- dense params: `27,362,364`
- difference vs arterial: `-768` parameters

This is the closest clean dense baseline while keeping a conventional head layout and the same long-context window as the longer arterial branch.

For latency attribution, a secondary dense run with `ATTN_WINDOWS=1024` is also useful. It isolates the overhead of the arterial loop and mixer against a short-window dense baseline.
