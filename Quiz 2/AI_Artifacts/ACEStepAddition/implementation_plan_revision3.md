# Goal Description
Separate the generation workflows of MusicGen and ACE-Step to avoid VRAM congestion and provide the user with more granular control over the generation process.

## Proposed Changes

### UI Additions (`app.py`)
1. **Engine Selection Toggle:** Add a `gr.Radio` or `gr.Dropdown` next to the Transcribe checkbox with options `["MusicGen", "ACE-Step Turbo"]`, defaulting to `"MusicGen"`.
2. **Independent MusicGen Button:** Add a `gr.Button("Run MusicGen")` in the column above "Baseline Generation (MusicGen)".
3. **Independent ACE-Step Button:** Add a `gr.Button("Run ACE-Step")` in the column above "Improved Generation (ACE-Step Turbo)".

### Logic Updates (`app.py`)
1. **Main `Generate & Evaluate` Button Routing:**
   - Modify `generate_and_evaluate` to accept the `engine_choice` parameter.
   - If `MusicGen` is selected: Generate Baseline (MusicGen) and Improved (MusicGen). Skip ACE-Step.
   - If `ACE-Step Turbo` is selected: Generate Improved (ACE-Step). Skip MusicGen generation for both Baseline and Improved.
2. **Independent Handlers:**
   - Create a `run_musicgen_only` function bound to the "Run MusicGen" button. It will use the current prompts and duration to generate and evaluate the MusicGen columns.
   - Create a `run_acestep_only` function bound to the "Run ACE-Step" button. It will use the current Improved prompt and duration to generate the ACE-Step column.

## Open Questions
> [!IMPORTANT]  
> 1. When the main "Generate & Evaluate" button is clicked and "ACE-Step Turbo" is selected as the engine, should we *skip* generating the Baseline track completely, or should we still generate a Baseline using MusicGen and only use ACE-Step for the Improved track?
> 2. For the independent "Run MusicGen" and "Run ACE-Step" buttons, should they automatically run the evaluation (Quality, Alignment, etc.) and update the Markdown results table, or should they strictly just generate the audio?
