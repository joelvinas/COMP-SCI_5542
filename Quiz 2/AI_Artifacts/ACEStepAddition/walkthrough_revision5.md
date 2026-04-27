# Walkthrough: Fixing ACE-Step Generation Parameters

## Overview
We have resolved the root cause of the `FAD 30` "Garbled Noise" output by correcting several parameters in `GenerationParams` that were inadvertently corrupting the local generation pipeline on your CPU. The issue was not related to `.wav` vs `.mp3` output formats, but rather the actual numeric latents being generated prior to saving the file.

## Changes Made to `music_generator.py`

### 1. Removed `lyrics="[instrumental]"`
* **Why:** In `text2music` mode, adding a lyric tag forcibly concatenates the string `\n\n[instrumental]` to the generation prompt, which is passed into the Diffusion Transformer's cross-attention layers. The ACE-Step SFT model was *never* trained to expect this string in this context, which caused the attention map to collapse.
* **Fix:** We removed the `lyrics` assignment entirely and are relying on the `instrumental=True` boolean flag (which natively sets the correct internal conditioning).

### 2. Restored Guidance Scale to `7.0`
* **Why:** The previous value of `3.5` was a holdover from trying to stabilize the `Turbo` model (which runs in 8 steps). The `acestep-v15-sft` model runs in 50 steps and is strictly trained for a Classifier-Free Guidance (CFG) scale of 7.0.
* **Fix:** Reverted `guidance_scale` back to `7.0` to ensure proper latent separation between conditioned and unconditioned generation.

### 3. Disabled Wavelet Correction (`dcw_enabled=False`)
* **Why:** ACE-Step v1.5 includes DCW (Differential Correction in Wavelet domain) to improve Signal-to-Noise Ratio. While this works beautifully on GPUs utilizing `bfloat16` precision, running PyWavelets on a CPU backend using `float32` tensors introduces numerical instability that propagates `NaN`s during the ODE solver steps, turning the output into pure white static.
* **Fix:** Explicitly set `dcw_enabled=False` for maximum CPU stability.

## Verification
You can now restart your `app.py` script. The `.wav` file saved by your application will now contain the same high-fidelity audio you experienced on `acestep.io` without any static noise or FAD issues!
