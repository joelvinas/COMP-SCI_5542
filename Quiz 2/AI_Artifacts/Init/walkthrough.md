# Video Game Race Music Generator AI - Walkthrough

I have successfully designed and built the requested Python application that uses `facebook/musicgen-small` to generate 16-bit retro video game music based on user descriptions of a video game race.

## Architecture & Modules Created

### 1. Music Generation (`music_generator.py`)
This module initializes the `MusicgenForConditionalGeneration` model and the `AutoProcessor`. 
- **`create_prompts(description, revisions)`**: Implements the Prompt Engineering strategy by taking the raw user input and creating two prompts. The *Baseline* prompt simply uses the user's description. The *Improved* prompt wraps the description in a structured template tailored for retro 16-bit chiptune style video game soundtracks.
- **`generate_music(prompt, duration)`**: Processes the prompt through the model to produce waveform audio data and the sample rate.
- **`save_audio(...)`**: Utility to save the generated numpy arrays to `.wav` files.

### 2. Automated Evaluation Metrics (`evaluator.py`)
This module uses a combination of DSP techniques (`librosa`) and Foundation Models (`laion/clap-htsat-unfused`) to provide automated scoring:
*   **Musical Quality:** Measured by the variance of spectral contrast (approximating musical dynamic complexity).
*   **Alignment:** Computes the cosine similarity between the generated audio embedding and the aggregate text prompt embedding using the CLAP model.
*   **Realism:** Computes a normalized score comparing the CLAP similarity of the audio against the prompt "retro 16-bit chiptune video game music" versus "modern realistic orchestral music".
*   **Creativity:** Measured by the variance of the spectral centroid to detect musical divergence/variety.
*   **Loopability:** Calculates the normalized cross-correlation between the first 1-second segment and the final 1-second segment of the generated audio to evaluate how seamlessly it can repeat.

### 3. Application Interfaces (`app.py` & `cli.py`)
Both interfaces instantiate the models once upon startup to ensure quick consecutive generations.
- **`app.py`**: A fully functional Gradio web application that presents a text box for the race description and an optional text box for revisions. It outputs the generated audio files side-by-side alongside a markdown table showing the exact performance on the 5 metrics.
- **`cli.py`**: A Command-Line Interface allowing for scriptable usage. You can run it via:
  ```bash
  python cli.py --description "An ancient race of tree-dwellers, secretive and melodic" --revisions "add more drums" --duration 10
  ```

## Setup Instructions

To get started, make sure you have installed the required dependencies on your local machine:
```bash
pip install -r requirements.txt
```

Then you can launch the Gradio web interface:
```bash
python app.py
```
Or you can use the CLI tool as described above!
