# MIDI Transcription Engine Implementation Plan

This plan outlines the integration of a multi-instrument transcription pipeline that converts generated WAV files into MIDI stems and MusicXML sheet music.

## User Review Required

> [!IMPORTANT]
> Please review the architectural breakdown below and let me know your final decision:
> **Option A:** Downgrade to Python 3.8 and use `omnizart`.
> **Option B:** Stay on Python 3.13 and use `demucs` + `basic-pitch`. (Note: I highly recommend `demucs` over `spleeter` as it is actively maintained by Facebook, uses PyTorch, runs natively on Python 3.13, and provides superior stem separation).

### Architectural Comparison: Omnizart vs. Separation + Basic-Pitch

| Feature | `omnizart` (Downgrade Required) | `demucs` + `basic-pitch` (Python 3.13) |
| :--- | :--- | :--- |
| **Complexity (Setup)** | **High.** Requires setting up a Python 3.8 virtual environment and dealing with deprecated TensorFlow 2.x dependencies. May break compatibility with newer versions of `transformers` and `gradio`. | **Medium.** We stay on Python 3.13. Requires wiring two modern APIs together (separation first, then pitch detection on the stems). |
| **Complexity (Code)** | **Low.** Omnizart has a single API call that handles multi-instrument transcription directly to a multi-track MIDI file. | **High.** We must write custom logic to separate the WAV, run basic-pitch on each stem separately, and then programmatically merge them back into a single multi-track MIDI file using `mido` or `pretty_midi`. |
| **Compute / Time** | **Extremely Heavy.** Processing a 10s audio file can take 1-3 minutes and requires significant RAM/VRAM. | **Heavy but Faster.** Spleeter/Demucs is very fast on GPU. Basic-pitch is lightweight. Expected total time is ~15-30 seconds. |
| **Transcription Quality** | **Excellent.** Specifically trained for ensemble polyphonic transcription (differentiating piano from drums natively). | **Good.** Spleeter/Demucs expects pop/rock tracks (vocals/drums/bass/other). Chiptune synths may bleed across stems. Basic-pitch will accurately transcribe whatever is isolated, but the instrument mapping might be loose. |

**My Recommendation:** 
Given that you are building a modern AI application, downgrading to Python 3.8 to support a deprecated TensorFlow library (`omnizart`) introduces massive technical debt and risks breaking your current HuggingFace setup. We should use **Option B (`demucs` + `basic-pitch`)**.

## Proposed Architecture (Assuming Option B)

### 1. Requirements Update
Add `basic-pitch`, `demucs`, `mido` (for MIDI merging), and `music21` to `requirements.txt`.

### 2. Music Generator Expansion (`music_generator.py`)
Add a new method `generate_score(self, wav_path, output_dir, transcribe=False)`:
- If `transcribe=False`, skip and return `None`.
- Run `demucs` to split the WAV into `drums.wav`, `bass.wav`, `melody.wav`.
- Loop through each stem and run `basic_pitch.inference.predict(stem_path)`.
- Use `mido` to assign each stem to a different MIDI channel/instrument and merge them into `stems.mid`.
- Use `music21`: `score = music21.converter.parse(midi_path)` and export it: `score.write('musicxml', fp=xml_path)`.
- Return the paths to the generated MIDI and XML files.

### 3. Gradio Interface Update (`app.py`)
- Add a new `gr.Checkbox(label="Generate MIDI/XML Stems (Takes extra time)", value=False)` to the UI.
- Add new `gr.File` components to the UI under the "Improved Generation" column for downloading the `stems.mid` and `sheet_music.xml`.
- Update the `generate_and_evaluate` function to accept the boolean toggle and pass it to `generate_score`.
