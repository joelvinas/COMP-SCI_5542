# MIDI Transcription Engine Walkthrough

I have successfully integrated the complete **Demucs + Basic-Pitch** pipeline and updated the aesthetic targets of the application from SNES chiptune to modern **MIDI-orchestral** styling.

## Changes Made

1. **Transcription Pipeline (`midi_engine.py`)**:
   - Created the `MidiTranscriptionEngine` class.
   - Used `demucs.api` to split the generated WAV file into independent audio stems (drums, bass, vocals, other).
   - Passed each stem sequentially into Spotify's `basic_pitch` to detect multi-timbral polyphony and write raw MIDI tracks.
   - Hooked up `mido` to dynamically assign appropriate General MIDI instrument program numbers to each stem and merge them into a single `stems.mid` file.
   - Connected `music21` to seamlessly translate the merged MIDI file into `sheet_music.xml` format for professional editing in Sibelius/MuseScore/Logic Pro.

2. **Stylistic Shift**:
   - Updated `prompt_translator.py`'s Gemini Instructions to specifically act as a "Music Theory Specialist for MIDI-orchestral Video Game Music".
   - Swapped out chiptune terms (e.g., "square waves") for orchestral terms (e.g., "pizzicato strings").

3. **Gradio UI Integration (`app.py`)**:
   - Updated all text overlays to reflect the new MIDI-orchestral goal.
   - Appended a new `Generate MIDI/XML Stems (Takes extra time)` toggle to the form parameters.
   - Exposed two new download components beneath the Improved Audio player for `stems.mid` and `sheet_music.xml`.

4. **Dependencies**:
   - Added `basic-pitch`, `demucs`, `mido`, and `music21` to your `requirements.txt`.

## How to Test

Before running, you must install the hefty new dependencies:
```bash
pip install -r requirements.txt
python app.py
```

Check the new toggle box in your browser, generate a race theme, and you'll immediately get high-quality orchestral stems and sheet music!
