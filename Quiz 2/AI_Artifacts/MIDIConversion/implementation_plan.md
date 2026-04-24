# MIDI Transcription Engine Implementation Plan

This plan outlines the integration of a multi-instrument transcription pipeline that converts generated WAV files into MIDI stems and MusicXML sheet music.

## User Review Required

> [!WARNING]
> **Library Compatibility Risk:** The `omnizart` library heavily depends on TensorFlow and is historically optimized for older Python versions (3.6 - 3.8). I noticed from your error logs that you are running **Python 3.13.5**. `omnizart` may fail to install or execute in this environment due to deep dependency conflicts.

## Open Questions

> [!IMPORTANT]
> 1. **Alternative Transcription Library:** If `omnizart` fails due to Python 3.13 compatibility, would you be open to using Spotify's `basic-pitch`? It is modern, actively maintained, and handles polyphonic audio-to-MIDI transcription extremely well, though it doesn't do multi-instrument separation quite exactly like Omnizart.
> 2. **Execution Blocking:** `omnizart` transcription is very computationally expensive and can take several minutes per WAV file. Do you want this process to run automatically every time a track is generated, or should we add a separate "Transcribe to MIDI/XML" button in the UI?
> 3. **Omnizart Checkpoints:** Omnizart requires downloading model checkpoints via `omnizart download-checkpoints` on the first run. I will add this initialization to the code, but note it will make the first run very slow.

## Proposed Architecture

### 1. Requirements Update
Add `omnizart` and `music21` to `requirements.txt`.

### 2. Music Generator Expansion (`music_generator.py`)
Add a new method `generate_score(self, wav_path, output_dir)`:
- Initialize Omnizart transcription: `from omnizart.music import app as omni_app` and run `omni_app.transcribe(wav_path, output=output_dir)`.
- Locate the resulting `.mid` file.
- Use `music21`: `score = music21.converter.parse(midi_path)` and export it: `score.write('musicxml', fp=xml_path)`.
- Return the paths to the generated MIDI and XML files.

### 3. Gradio Interface Update (`app.py`)
- Add new `gr.File` components to the UI under the "Improved Generation" column for downloading the `stems.mid` and `sheet_music.xml`.
- Update the `generate_and_evaluate` function to pass the saved `improved_audio_path` to the new `generate_score` method, and return the resulting files so the user can download them.

## Verification Plan

### Automated/Manual Tests
- Ensure `pip install -r requirements.txt` succeeds (monitoring for `omnizart` failures).
- Run `python app.py` and generate a short 5-second track.
- Verify the transcription engine engages, creates a MIDI file, and subsequently exports a MusicXML file.
- Verify the UI populates the download links correctly and the files are successfully written to the `Output/{race_name}_{idx}/` directory.
