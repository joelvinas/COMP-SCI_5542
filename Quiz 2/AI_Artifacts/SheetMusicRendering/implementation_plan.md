# Add Verovio Sheet Music Rendering

This plan outlines the integration of the `verovio` library to automatically render the generated MusicXML into a visual sheet music format (SVG), and displaying it within the Gradio interface.

## User Review Required
> [!NOTE]
> Please review the plan below. If approved, I will install the `verovio` package and update the UI.

## Proposed Changes

### Configuration
#### [MODIFY] `requirements.txt`
- Add `verovio` to the dependencies list.

### Backend Engine
#### [MODIFY] `midi_engine.py`
- Import `verovio`.
- In `generate_score()`, immediately after `sheet_music.xml` is successfully exported by `music21`:
  - Initialize the `verovio.toolkit()`.
  - Load the `sheet_music.xml`.
  - Render it as `sheet_music.svg` into the Output directory.
- Update the return signature of `generate_score` to return `(final_midi_path, xml_path, svg_path)`.

### Frontend Application
#### [MODIFY] `app.py`
- Update the internal `transcribe_only` and `generate_and_evaluate` functions to capture the new `svg_path` return value.
- Add a new `gr.HTML` or `gr.Image` component underneath the Improved Audio player to visually display the rendered SVG sheet music, mimicking the Soundslice viewer functionality.
- Add an output specifically to download the SVG.

## Verification Plan

### Automated Tests
- Run `pip install verovio`.
- Test `verovio.toolkit()` rendering on an existing `sheet_music.xml` file.

### Manual Verification
- Ask the user to click the "Transcribe Stems" button and visually confirm that the sheet music renders correctly on the Gradio interface.
