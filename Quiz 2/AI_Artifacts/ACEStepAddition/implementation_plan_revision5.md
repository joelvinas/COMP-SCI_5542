# Goal: Fix Local ACE-Step "Garbled Noise" Generation

The "garbled noise" (FAD ~26-33) you are hearing locally compared to the clean output of `acestep.io` is NOT caused by using `.wav` instead of `.mp3`. The issue is occurring *during the latent generation phase* before the audio is even saved. 

I've analyzed the raw generation parameters we've been passing to the ACE-Step SFT model, and several are actively fighting against the model's architecture, causing the DiT (Diffusion Transformer) to collapse into white noise on your CPU:

1. **The Lyrics Hack (`lyrics="[instrumental]"`):** We previously added this to try and prevent vocals. However, in `text2music` mode, adding a lyric prompt concatenates `\n\n[instrumental]` directly into the cross-attention tensors. The SFT `text2music` model was never trained to expect this text structure, which severely corrupts the conditioning matrices.
2. **Guidance Scale (`guidance_scale=3.5`):** We lowered the CFG (Classifier-Free Guidance) to 3.5 to prevent "frying." While this helps the `Turbo` model, the `SFT` model is strictly tuned for a default CFG of **7.0**. A CFG of 3.5 on SFT prevents the latents from properly separating from unconditional noise, resulting in static.
3. **Differential Wavelet Correction (`dcw_enabled=True`):** ACE-Step v1.5 includes DCW (Differential Correction in Wavelet domain) to improve SNR on GPUs. However, running PyWavelets on a CPU backend using `float32` (instead of GPU `bfloat16`) can introduce numerical instability and NaN propagation during the ODE solver steps.

## Proposed Changes

We will modify the `GenerationParams` inside `music_generator.py` to correctly align with the optimal SFT architecture.

#### [MODIFY] `music_generator.py`
```python
        params = GenerationParams(
            task_type="text2music",
            caption=prompt,
            instrumental=True,        # Use the explicit boolean flag to prevent vocals
            duration=duration,
            bpm=bpm_val,
            keyscale=str(keyscale) if keyscale else "",
            thinking=True,
            inference_steps=50,
            guidance_scale=7.0,       # Restore to standard SFT CFG
            dcw_enabled=False         # Disable Wavelet Correction for CPU stability
        )
```
*(Note: We are removing the `lyrics="[instrumental]"` line entirely).*

## Open Questions
**"Is it easier to output in mp3 than wav?"**
Outputting to MP3 is possible, but it won't fix the quality issue. Your evaluation script (`evaluator.evaluate_all`) calculates the FAD score directly from the raw data array *before* it is ever saved to a file. The issue is that the array itself currently contains white noise. Once we fix the generation parameters above, the WAV file will sound identical to the beautiful MP3s you got from acestep.io!

## User Review Required
Please approve this plan so I can make the changes to `music_generator.py` and run a quick generation test!
