# Video Game Race Music Generator AI

This plan outlines the design and development of a Python application that generates loopable, retro-style video game music (SNES/N64 era) based on a text description of a video game race's culture, strengths, and weaknesses.

## User Review Required

> [!IMPORTANT]
> Please review the chosen approach for the user interface and evaluation method. I have proposed a **Gradio** web interface because it provides excellent built-in audio players and is very standard for Hugging Face model interaction.

## Open Questions

> [!WARNING]
> 1. **User Interface:** Is a Gradio web application acceptable, or would you prefer a Streamlit app, a Jupyter Notebook, or a Command Line Interface (CLI)?
> 2. **Foundation Model:** I recommend starting with `facebook/musicgen-small` via the `transformers` library, as it is lightweight and specifically tuned for music generation. Does that work for you, or do you strongly prefer AudioLDM2?
> 3. **Evaluation Metrics:** Should the comparison of baseline vs. improved settings be an automated process (e.g., using an LLM to evaluate alignment, or DSP libraries to evaluate loopability) or a manual subjective rating interface where a user listens and scores both outputs?

## Proposed Architecture

### 1. Pretrained Foundation Model Integration
- **Model:** `facebook/musicgen-small` (or `audioldm2`).
- **Library:** `transformers` and `torch`.
- **Functionality:** Will load the model and generate a waveform based on text conditioning.

### 2. Prompt / Input Engineering Engine
- **Input:** User provides a description of the race (e.g., "An ancient race of tree-dwellers, strong in magic but physically weak, secretive and melodic").
- **Baseline Prompt:** Passes the description almost directly to the model.
- **Improved Prompt:** Uses a prompt template designed to elicit retro video game music. E.g., `"{user_description}. 16-bit SNES style video game music, chiptune, synthetic instruments, loopable, rhythmic, atmospheric, retro gaming soundtrack."`
- **Revision System:** A conversational input where the user can say "make it faster" or "add more drums", which modifies the aggregate prompt and regenerates.

### 3. Evaluation Framework
- A module to run both the Baseline and Improved configurations side-by-side.
- A rating system (1-5 scale) across the requested metrics:
  - **Musical Quality** (melodic coherence, etc.)
  - **Alignment** (does it fit the prompt?)
  - **Realism** (does it sound like a pre-2000 game?)
  - **Creativity** (divergence)
  - **Loopability** (seamlessness)
- Output the comparison results (e.g., to a CSV or within the UI).

## Proposed Changes

### `c:\Users\audranian\source\repos\COMP-SCI_5542\Quiz 2\requirements.txt`
#### [NEW] `requirements.txt`
Dependencies for the project: `torch`, `transformers`, `gradio`, `scipy`, `torchaudio`.

### `c:\Users\audranian\source\repos\COMP-SCI_5542\Quiz 2\app.py`
#### [NEW] `app.py`
The main application file containing:
- The model loading logic.
- The prompt engineering templates.
- The Gradio UI layout for generating, revising, and evaluating the music.

### `c:\Users\audranian\source\repos\COMP-SCI_5542\Quiz 2\music_generator.py`
#### [NEW] `music_generator.py`
A modular file to handle the model inference so the Gradio app stays clean. Includes functions for baseline generation, improved generation, and audio saving.

## Verification Plan

### Automated/Manual Tests
- **Setup:** Run `pip install -r requirements.txt`.
- **Execution:** Run `python app.py` and open the local web server.
- **Verification:**
  1. Input a race description.
  2. Verify that two distinct audio clips are generated (Baseline vs. Improved).
  3. Play the audio to ensure it sounds like retro video game music.
  4. Test the revision feature (e.g., "Make it more ominous") and ensure the newly generated audio reflects the change.
  5. Fill out the evaluation metrics and ensure the scores are recorded.
