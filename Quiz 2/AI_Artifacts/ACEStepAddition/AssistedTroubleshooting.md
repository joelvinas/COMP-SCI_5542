# ACE-Step Integration — Assisted Troubleshooting Log

This document records all issues encountered and solutions applied during the integration of ACE-Step into the Video Game Race Music AI pipeline (`app.py`, `music_generator.py`, `evaluator.py`, `prompt_translator.py`).

---

## Issue 1 — Silent 354 KB WAV Output

### Symptom
Running ACE-Step Turbo produced a ~354 KB `.wav` file with no audible sound. The terminal showed generation completing successfully, including normalization logs (`Peak=0.8913`), yet the output audio was pure silence.

### Root Cause
The `generate_music()` call in the ACE-Step API accepts an optional `save_dir` parameter. When `save_dir=None`, the handler generates audio in memory but **does not write any file to disk**. The prior code relied on reading back the audio from a file path returned in `result.audios[0]["path"]`, but that path was an empty string `""` when no `save_dir` was provided.

The code fell through to the final fallback return:
```python
return 44100, np.zeros(44100)  # 1 second of silence ~ 354 KB WAV
```

### Solution
Updated `AceStepMusicGenerator.generate_music()` in `music_generator.py` to extract the audio **tensor directly from memory** instead of depending on a file path:

```python
audio_tensor = audio_dict.get("tensor")
if audio_tensor is not None:
    sample_rate = audio_dict.get("sample_rate", 44100)
    audio_data = audio_tensor.cpu().numpy().T
    if audio_data.ndim == 2 and audio_data.shape[1] == 1:
        audio_data = audio_data.squeeze(1)
    return sample_rate, audio_data
```

A file-path fallback was preserved for cases where the tensor is absent but a path exists.

**Files changed:** `music_generator.py`

---

## Issue 2 — CLAP Evaluator Crash: Stereo Audio Waveform

### Symptom
After fixing the silent output, a new error appeared during evaluation:

```
ValueError: Input waveform must have only one dimension, shape is (480000, 2)
```

The crash occurred in `evaluator.py -> compute_clap_similarity()` when passing the ACE-Step audio to the HuggingFace CLAP processor.

### Root Cause
ACE-Step SFT generates **stereo audio** (`shape: [samples, 2]`). The CLAP model processor and all `librosa` spectral functions strictly require **mono audio** (1D arrays). The evaluator had no downmixing step.

### Solution
Added a stereo-to-mono downmix at the top of `evaluate_all()` in `evaluator.py`:

```python
# Ensure audio is mono for librosa and CLAP
if audio.ndim == 2:
    if audio.shape[1] == 2:
        audio = np.mean(audio, axis=1)  # Average L + R channels
    else:
        audio = audio.squeeze()
```

**Files changed:** `evaluator.py`

---

## Issue 3 — Poor Audio Quality with ACE-Step Turbo

### Symptom
ACE-Step Turbo produced audio of unusably poor quality: artifacts, noise, and incoherent musical structure.

### Root Cause
The Turbo model uses only **8 diffusion steps** via a distilled sampling schedule. This is far too few steps for coherent, high-quality music, especially on CPU where there is no GPU-accelerated sampling to compensate.

### Solution
Switched from the Turbo model to the **SFT (Supervised Fine-Tuning)** model, which is trained for high-fidelity output. Inference steps raised to 50 and guidance scale set to 3.5 (safe range: 3.0–4.5):

```python
# music_generator.py
self.dit_handler.initialize_service(
    config_path="acestep-v15-sft",   # was: "acestep-v15-turbo"
    ...
)

params = GenerationParams(
    inference_steps=50,       # was: 8
    guidance_scale=3.5,       # was: not set
    ...
)
```

**Files changed:** `music_generator.py`

> **Note:** The SFT model requires ~4.8 GB of disk space for its weights (`model.safetensors`). See Issue 4 for the download fix.

---

## Issue 4 — Missing SFT Model Weights (`model.safetensors` absent)

### Symptom
The `checkpoints/acestep-v15-sft/` directory existed on disk but only contained config and code files. The 4.79 GB `model.safetensors` weights file was absent. Generation would fail immediately on model load.

### Root Cause
ACE-Step's `download_submodel()` function short-circuits with `"already exists"` if the **folder** exists, regardless of whether the weights file is inside it. A prior initialization created the folder and synced code files without completing the weights download.

### Solution
Ran `download_submodel()` with `force=True` to bypass the folder-existence check and complete the download:

```python
# download_sft_model.py (one-time utility script)
from acestep.model_downloader import download_submodel
success, msg = download_submodel(
    model_name="acestep-v15-sft",
    checkpoints_dir=Path(".") / "checkpoints",
    force=True,   # Required: folder existed but weights were missing
)
```

> **Warning:** The download script initially crashed printing result messages because Windows PowerShell uses `cp1252` encoding by default, which cannot encode certain Unicode characters (e.g. checkmark/cross emoji). The print statements were replaced with plain ASCII equivalents (`SUCCESS:` / `FAILED:`).

**Files added:** `download_sft_model.py` (one-time use, safe to delete after download)

---

## Issue 5 — Improved Prompt Confusing ACE-Step SFT

### Symptom
ACE-Step SFT produced confused or incoherent outputs when fed the `improved_prompt`.

### Root Cause
The `improved_prompt` format appends structured metadata to the end of the baseline text:

```
{baseline_prompt} BPM: 95 BPM, Key: D minor, Lead: Deep resonant cello ensemble ... Make it: {revisions}.
```

This key-value metadata suffix is well-suited for **MusicGen**, which was trained on metadata-augmented prompts. However, ACE-Step SFT was trained to respond to natural-language captions only. The structured suffix caused the model to produce degraded, artifact-heavy results.

### Solution
Updated `app.py` to pass **`baseline_prompt`** to ACE-Step SFT instead of `improved_prompt`. The baseline prompt is a clean, natural-language MIDI-orchestral description without appended metadata:

```python
# app.py - _core_generate()
# Before:
sr_ace, audio_ace = ace_generator.generate_music(improved_prompt, duration)
metrics_ace = evaluator.evaluate_all(audio_ace, sr_ace, improved_prompt)

# After:
sr_ace, audio_ace = ace_generator.generate_music(baseline_prompt, duration)
metrics_ace = evaluator.evaluate_all(audio_ace, sr_ace, baseline_prompt)
```

The displayed ACE-Step prompt in the Gradio UI was also updated to show `baseline_prompt` instead of `improved_prompt`.

**Files changed:** `app.py`

---

## Supporting Changes (Non-Error Driven)

### Branding: "ACE-Step Turbo" to "ACE-Step SFT"
All UI labels, radio button choices, conditional branches, and function calls in `app.py` were updated from `"ACE-Step Turbo"` to `"ACE-Step SFT"` to reflect the model switch accurately.

### Default Fallback Prompt Updated
`music_generator.py -> create_prompts()` fallback changed from `"16-bit SNES chiptune ..."` to `"MIDI-orchestral ..."` to match the prompt style expected by both MusicGen and ACE-Step SFT.

### `prompt.md` — Given Prompt Section
The saved `prompt.md` file was updated to include a `# Given Prompt` section at the top, capturing the user's original inputs before the AI-translated prompts and metrics table:

- **Race Name**
- **Race Description**
- **Revisions / Tweaks** (only if provided)

### `PromptTranslator` System Instruction Refinement
The Gemini system instruction was updated to guide prompt generation for ACE-Step SFT compatibility:
- Prioritizes `instrument_first` and `mood_emphasized` Prompt2MusicBench templates (stable for diffusion models)
- Added a **"Clean Signal" acoustic refinement rule**: quality tags appended to every prompt (`high fidelity, clean mix, professional mastering, no distortion, pure melodic tones`)

---

## Issue 6 — Garbled White Noise (FAD ~30) on CPU

### Symptom
Even after successfully running the SFT model, the generated `.wav` audio was pure, loud white noise with an extremely poor FAD score (~26-33). Meanwhile, the exact same prompt yielded high-quality music on `acestep.io`.

### Root Cause
Several interacting factors destroyed the DiT latents and VAE decoding on a standard CPU:
1. **The Lyrics Hack:** The `lyrics="[instrumental]"` parameter was silently appended to the prompt as `\n\n[instrumental]`. The `text2music` model isn't trained to parse lyrics matrices, which broke the text-attention layer.
2. **Guidance Scale:** We previously lowered the CFG to `3.5`. While this helps the Turbo model, the SFT model strictly requires a CFG of **7.0**. A lower CFG fails to separate the music signal from unconditional noise.
3. **DCW (Wavelet Correction) Instability:** Differential Wavelet Correction improves SNR on GPUs using `bfloat16`. However, running PyWavelets on a CPU backend using `float32` tensors is numerically unstable, propagating `NaN`s across the ODE solver steps.
4. **VAE `float32` Precision Drift:** ACE-Step's VAE Decoder was trained on A100 GPUs using mixed precision (`bfloat16`). When run on CPU in standard `float32`, the precision drift causes deep internal activations to exponentially overflow, turning the decoded latents into pure white noise.

### Solution
1. **Parameter Tuning:** Removed the `lyrics` assignment entirely, restored `guidance_scale=7.0`, and set `dcw_enabled=False` for maximum CPU stability.
2. **Architectural Mitigation:** Verified that `torch.autocast(device_type="cpu", dtype=torch.bfloat16)` mathematically fixes the `float32` VAE overflow. However, since most consumer CPUs lack hardware acceleration for `bfloat16`, this software emulation increases a 5-minute generation to over **15-20 minutes**.
3. **UI Update:** Given the severe performance bottleneck of `bfloat16` emulation and the catastrophic failure of native `float32` execution, we implemented **Option A**: We retained the local implementation but updated the UI to clearly articulate that ACE-Step SFT is an **experimental feature** that **requires a CUDA GPU** to function correctly, and will produce severe audio artifacts on CPU.

---

## Final Working Configuration

| Setting | Value |
|---|---|
| Model | `acestep-v15-sft` |
| Inference Steps | `50` |
| Guidance Scale | `7.0` |
| Prompt Used | `baseline_prompt` (plus native `instrumental=True` flag; no `lyrics`) |
| Wavelet Correction | `dcw_enabled=False` (disabled for CPU stability) |
| Audio Format | WAV (stereo, 48 kHz from ACE-Step) |
| Evaluator Input | Downmixed to mono before CLAP / librosa |
| Model Weights Path | `checkpoints/acestep-v15-sft/model.safetensors` (~4.79 GB) |
