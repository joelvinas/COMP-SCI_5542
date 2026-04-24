import os
import sys
import subprocess
import warnings
import logging
import verovio
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

# Suppress verbose basic-pitch warnings about missing backends and TFLite deprecation
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
logging.getLogger().setLevel(logging.ERROR)

# Suppress Verovio font errors and Gradio SVG watermarking warnings
verovio.enableLog(verovio.LOG_OFF)
warnings.filterwarnings("ignore", message=".*Watermarking for SVG images.*")

import mido
import music21

class MidiTranscriptionEngine:
    def __init__(self):
        print("MidiTranscriptionEngine initialized.")

    def generate_score(self, wav_path, output_dir):
        print(f"Running Demucs separation on {wav_path}...")
        
        try:
            subprocess.run([sys.executable, "run_demucs_patched.py", wav_path, "-o", output_dir], check=True)
        except Exception as e:
            print(f"Demucs separation failed: {e}")
            return None, None, None
            
        base_wav_name = os.path.splitext(os.path.basename(wav_path))[0]
        demucs_out_dir = os.path.join(output_dir, "htdemucs", base_wav_name)
        
        stem_names = ["drums", "bass", "other", "vocals"]
        stem_midi_paths = []
        
        for stem_name in stem_names:
            stem_wav_path = os.path.join(demucs_out_dir, f"{stem_name}.wav")
            if not os.path.exists(stem_wav_path):
                continue
                
            print(f"Transcribing stem: {stem_name}")
            
            # Run basic-pitch
            try:
                predict_and_save(
                    audio_path_list=[stem_wav_path],
                    output_directory=output_dir,
                    save_midi=True,
                    sonify_midi=False,
                    save_model_outputs=False,
                    save_notes=False,
                    model_or_model_path=f"{ICASSP_2022_MODEL_PATH}.onnx"
                )
            except Exception as e:
                print(f"Basic-pitch failed on {stem_name}: {e}")
            
            midi_path = os.path.join(output_dir, f"{stem_name}_basic_pitch.mid")
            
            if os.path.exists(midi_path):
                stem_midi_paths.append((stem_name, midi_path))
                
        print("Merging MIDI stems...")
        final_midi_path = os.path.join(output_dir, "stems.mid")
        if stem_midi_paths:
            self._merge_midis(stem_midi_paths, final_midi_path)
        else:
            print("No MIDI stems generated.")
            return None, None, None
        
        print("Exporting MusicXML...")
        xml_path = os.path.join(output_dir, "sheet_music.xml")
        svg_path = os.path.join(output_dir, "sheet_music.svg")
        
        try:
            score = music21.converter.parse(final_midi_path)
            
            # Explicitly set part names for the sheet music
            for i, p in enumerate(score.parts):
                if i < len(stem_midi_paths):
                    p.partName = stem_midi_paths[i][0].capitalize()
                    p.partAbbreviation = p.partName[:3]
                    
            score.write('musicxml', fp=xml_path)
            
            # Render SVG using verovio
            tk = verovio.toolkit()
            
            # Dynamically locate the verovio data directory containing the SMuFL fonts
            verovio_data_path = os.path.join(os.path.dirname(verovio.__file__), "data")
            tk.setResourcePath(verovio_data_path)
            
            tk.loadFile(xml_path)
            tk.renderToSVGFile(svg_path, 1) # Render first page
            
        except Exception as e:
            print(f"MusicXML/SVG conversion failed: {e}")
            xml_path = None
            svg_path = None
            
        # Cleanup individual stem midis
        for _, path in stem_midi_paths:
            if os.path.exists(path):
                os.remove(path)
                
        return final_midi_path, xml_path, svg_path

    def _merge_midis(self, stem_midi_paths, output_path):
        merged_midi = mido.MidiFile()
        
        instruments = {
            'drums': 0, 
            'bass': 33, 
            'other': 0, 
            'vocals': 52 
        }
        
        for idx, (stem_name, midi_path) in enumerate(stem_midi_paths):
            try:
                mid = mido.MidiFile(midi_path)
                channel = 9 if stem_name == 'drums' else idx
                if channel == 9 and stem_name != 'drums':
                    channel = 10 
                
                track = mido.MidiTrack()
                track.append(mido.MetaMessage('track_name', name=stem_name.capitalize(), time=0))
                merged_midi.tracks.append(track)
                
                if channel != 9:
                    program = instruments.get(stem_name, 0)
                    track.append(mido.Message('program_change', program=program, time=0, channel=channel))
                    
                # Basic-pitch usually outputs notes in track 0 or 1.
                target_track = mid.tracks[1] if len(mid.tracks) > 1 else mid.tracks[0]
                for msg in target_track:
                    if not msg.is_meta:
                        if hasattr(msg, 'channel'):
                            msg.channel = channel
                    track.append(msg)
            except Exception as e:
                print(f"Error merging {stem_name}: {e}")
                
        merged_midi.save(output_path)
