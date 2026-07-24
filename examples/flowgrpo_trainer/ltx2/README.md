# LTX-2.3 text-to-audio-video FlowGRPO

This recipe trains `dg845/LTX-2.3-Diffusers` with a vLLM-Omni rollout, joint
audio-video CPS transitions, and the CLAP plus ImageBind rewards.
The checkpoint advertises `_class_name: LTX2Pipeline`; the registered rollout
adapter uses vLLM-Omni's LTX-2.3-specific `LTX23Pipeline` implementation behind
that checkpoint architecture key.

Two actor engines are supported:

| Script                        | Engine | Weights   | Notes                                  |
|-------------------------------|--------|-----------|----------------------------------------|
| `run_ltx2_3_t2av_lora.sh`     | FSDP2  | LoRA      | Default; lower GPU memory.             |
| `run_ltx2_3_t2av_veomni.sh`  | VeOmni | Full      | Requires VeOmni; param/optimizer offload enabled by default. |

## Prepare data

Use `dataset/vid_prompt/train.txt` and `test.txt`:

```bash
python3 examples/flowgrpo_trainer/ltx2/prepare_data.py \
  --input_dir ./dataset/vid_prompt \
  --output_dir "$WORKSPACE/data/vid_prompt/verl_omni"
```

The default training cap is 1,024 prompts, matching the reference YAML.

## Install reward dependencies

CLAP uses the existing `transformers` and `torchaudio` dependencies. ImageBind
is optional software under the CC-BY-NC-SA 4.0 non-commercial license:

```bash
pip install 'git+https://github.com/facebookresearch/ImageBind.git'
pip install 'git+https://github.com/facebookresearch/pytorchvideo.git'
```

Review the ImageBind license before enabling this reward in your environment.

## Launch

### LoRA (FSDP2)

```bash
bash examples/flowgrpo_trainer/ltx2/run_ltx2_3_t2av_lora.sh
```

### Full-weight (VeOmni)

```bash
bash examples/flowgrpo_trainer/ltx2/run_ltx2_3_t2av_veomni.sh
```

VeOmni does not support LoRA; the VeOmni recipe trains full weights and uses a
lower learning rate (`3e-5` vs `3e-4` for LoRA). Install VeOmni first per
[`docs/start/install.md`](../../docs/start/install.md) "Optional engine backends".

Both recipes default to 8 GPUs and vLLM-Omni tensor parallel size 8. Override
`NUM_GPUS`, `ROLLOUT_TP`, `MODEL_PATH`, `DATA_DIR`, `OUTPUT_DIR`, or
`TOTAL_TRAINING_STEPS` through environment variables. Extra Hydra overrides can
be appended to the command. One verl-omni global step consumes the same 48
unique prompts and 16 responses per prompt as one reference training epoch. The
default 15 global steps and a 24-prompt PPO mini-batch therefore reproduce the
reference recipe's 15 epochs and two optimizer updates per epoch.

The reference training recipe maintains a separate EMA evaluation copy.
The current verl-omni FlowGRPO trainer evaluates and checkpoints the live policy
(LoRA adapters for the FSDP2 recipe, full weights for the VeOmni recipe), so
the EMA-only evaluation behavior is the one reference option not mapped by
these launch scripts.
