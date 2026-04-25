# Integrate ACE-Step Model for a 3rd Generation Target

The user has requested that we generate a 3rd musical track using the `ACE-Step` model from HuggingFace (specifically the `ACE-Step/acestep-v15-sft` or a similar variant), alongside the existing baseline and improved `MusicGen` generations. The UI will be updated to display a 3rd column for this new track.

## Finalized Decisions
Based on user feedback, we will proceed with the following architectural decisions:

1. **Serial VRAM-Swapping Execution:** To prevent GPU Out-Of-Memory (OOM) crashes, we will run the generators purely serially. After `MusicGen` completes the Baseline and Improved generations, it will be forcefully offloaded to the CPU (`model.to("cpu")`) and the GPU cache will be cleared (`torch.cuda.empty_cache()`). Then, `ACE-Step` will be loaded onto the GPU, execute the 3rd generation, and also be offloaded. This dynamic swapping ensures the models never share VRAM simultaneously.
2. **Model Version:** We will use `ACE-Step/acestep-v15-turbo`. While structurally the same size as `sft`, the `turbo` variant uses adversarial diffusion distillation to render audio in 8 steps instead of 50. This lightning-fast generation will perfectly offset the 10-15 seconds of latency introduced by the dynamic VRAM-swapping mechanism.

## Proposed Changes

### `requirements.txt`
#### [MODIFY] `requirements.txt`
- Add `vector_quantize_pytorch` and any other custom ACE-Step dependencies.

### Application Logic (`app.py` and `music_generator.py`)
#### [MODIFY] `music_generator.py`
- We will update the `VideoGameMusicGenerator` class (or create a new `AceStepGenerator` wrapper) to load the `ACE-Step` model using `AutoModel.from_pretrained(..., trust_remote_code=True)`.
- We will add a `generate_ace_music(prompt)` function that mimics the existing `generate_music` structure but uses the ACE-Step API.

#### [MODIFY] `app.py`
- In the `gr.Blocks` layout, we will modify the row that currently displays "Baseline" and "Improved" to be a 3-column layout.
- We will add the **3rd Column:** "ACE-Step Generation", which includes its own `gr.Audio` player and download button.
- We will update the `generate_and_evaluate` function to intercept the improved prompt, pass it to the ACE-Step model, and return the 3rd audio path to the frontend.

## Verification Plan

### Automated Tests
- Run `python app.py` and ensure the server boots without CUDA Out-Of-Memory (OOM) errors.
- Trigger a generation sequence and verify that all 3 audio files (Base, Improved, ACE-Step) are successfully created and saved in the output directory.

### Manual Verification
- Ask the user to verify the Gradio UI accurately reflects a 3-column layout and the third column plays the ACE-Step generated audio.
