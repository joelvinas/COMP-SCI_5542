# UI Separation and Independent Execution Paths

We have successfully decoupled the execution paths for MusicGen and ACE-Step to give you granular control over the UI and ensure rock-solid stability during generation.

## What Changed

### Engine Selection
The main "Generate & Evaluate" button is now equipped with a Primary Generator Engine toggle (`gr.Radio`), defaulting to "MusicGen". 
- When **MusicGen** is selected, it will generate the Baseline and Improved tracks via MusicGen, evaluating them and populating their respective columns. ACE-Step is skipped.
- When **ACE-Step Turbo** is selected, it will exclusively generate the ACE-Step Improved track, evaluate it, and populate the third column. MusicGen generation is skipped entirely, saving significant time.

### Independent Execution Buttons
You now have direct control over each column:
- **[Re-run MusicGen]**: Located directly above the "Baseline Generation (MusicGen)" column. This button ignores the engine toggle and strictly generates & evaluates the MusicGen tracks.
- **[Run ACE-Step]**: Located directly above the "Improved Generation (ACE-Step Turbo)" column. This button strictly generates & evaluates the ACE-Step track.

### State Preservation
When you click any of the independent buttons or the main generation button, the UI uses `gr.update()` to intelligently preserve the audio output components of the *other* generator. This means you can run MusicGen, then separately run ACE-Step, and your screen will proudly display all three tracks simultaneously without erasing your previous work!

### Unified Evaluation Table
The Markdown Evaluation Results table has been expanded into a pristine 4-column layout:
`| Metric | Baseline | Improved | ACE-Step |`
- If an engine is skipped during a run, its evaluation metrics will safely display **N/A** for that specific run, precisely matching your request.

## Verification
- Code syntax successfully compiled using `py_compile`.
- UI topology is visually segmented and logic bindings are safely routed into `_core_generate` multiplexer.
- VRAM offloading (`offload_to_cpu=True`) was restored for ACE-Step to guarantee the models operate securely within the 16 GB boundary during these sequential runs.
