import os
import json
from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

class ImprovedParameters(BaseModel):
    BPM: str = Field(description="BPM of the track")
    Key: str = Field(description="Musical key of the track")
    Lead: str = Field(description="Lead instrument of the track")

class PromptTranslatorOutput(BaseModel):
    baseline_prompt: str = Field(description="The cleaned musical instruction prefixed with MIDI-orchestral.")
    improved_parameters: ImprovedParameters = Field(description="Parameters derived from the text's energy.")

class PromptTranslator:
    def __init__(self):
        # Load environment variables from a local .env file if it exists
        load_dotenv()
        
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in your environment variables or a .env file.")
            
        self.client = genai.Client(api_key=api_key)
        
        # self.system_instruction = (
        #     "Role: You are a Music Theory Specialist for MIDI-orchestral Video Game Music.\n"
        #     "Task: Translate cultural flavor text into a MusicGen Signature.\n"
        #     "Core Logic Requirements:\n"
        #     "- Zero-Hardcoding Rule: Do not use specific race names (e.g. Goblins, Ogres). Use a Semantic Mapping Pattern to extract attributes.\n"
        #     "- Physical-to-Musical Extraction:\n"
        #     "  - Map Size/Power (e.g., 'towering', 'tiny') to Frequency/BPM (e.g., 'Heavy sub-bass' vs 'High-pitched staccato').\n"
        #     "  - Map Environment (e.g., 'jungle', 'underground') to Acoustics (e.g., 'Short dry reverb' vs 'Spacious dark hall reverb').\n"
        #     "  - Map Culture (e.g., 'tribal', 'hoarding') to Instrumentation (e.g., 'Woodblocks/Percussion' vs 'Metallic/Found-object synths').\n"
        #     "- The 'Noise-Strip' Filter: Automatically remove 'clipping-prone' adjectives (e.g., physical colors, specific nouns like 'trash', and abstract actions like 'hoarding') that lack a direct musical signal.\n"
        #     "Avoidance Rules: Never include the words 'laughter', 'coins', or 'trash'. Replace them with 'Rhythmic pizzicato strings' or 'Randomized woodblock staccato'.\n"
        #     "Formatting: Prefix all outputs with 'MIDI-orchestral '."
        # )

        # self.system_instruction = (
        #     "Role: You are a Lead Audio Director specializing in 16-bit SNES and PS1-era RPG soundtracks.\n"
        #     "Task: Convert physical race lore into a 'Musical Composition Script' for MusicGen.\n\n"
        #     "CORE COMPOSITION RULES:\n"
        #     "1. PHYSICAL -> FREQUENCY: Translate size to pitch. Tiny/Shy = High-register, thin-textured synths. Large/Strong = Deep, resonant bass and sub-frequencies.\n"
        #     "2. ENVIRONMENT -> TEXTURE: Forest/Glowing = Shimmering pads, glockenspiel, and high-shelf reverb. Underground/Cave = Dark low-pass filters and long tail delays.\n"
        #     "3. CULTURE -> RHYTHM: Repairers/Builders = Staccato, syncopated rhythms (woodblocks, triangles). Warriors = Driving, low-mid percussion (taikos, heavy snares).\n"
        #     "4. NOISE-STRIP: ABSOLUTELY PURGE physical traits (hair color, skin type, specific nouns like 'mushroom-houses'). These cause digital 'static' and hallucinations in MusicGen.\n\n"
        #     "OUTPUT SCHEMA REQUIRMENT:\n"
        #     "Return only a structured musical prompt focusing on BPM, Key, Instrumentation, and Arrangement Style.\n"
        #     "Example for a mystical forest race: '145 BPM, F# Major. Shimmering glockenspiel arpeggios over ethereal woodwind pads. Fast staccato woodblock percussion. Light and magical 16-bit chiptune.'"
        # )

        self.system_instruction = (
            "Role: You are a Lead Audio Director specializing in 16-bit SNES and PS1-era RPG soundtracks.\n"
            "Task: Dynamically translate physical race lore into a 'Musical Composition Script' for MusicGen using Prompt2MusicBench templates.\n\n"
            
            "PHASE 1: MUSICAL MAPPING LOGIC:\n"
            "1. PHYSICAL -> FREQUENCY: Translate size/power to pitch. Tiny/Shy = High-register, thin-textured synths. Large/Strong = Deep, resonant bass and sub-frequencies.\n"
            "2. ENVIRONMENT -> TEXTURE: Glowing/Forest = Shimmering pads and high-shelf reverb. Underground/Cavern = Low-pass filters and long tail delays.\n"
            "3. CULTURE -> RHYTHM: Artisans/Repairers = Staccato, syncopated rhythms (woodblocks). Warriors/Raiders = Driving, low-mid percussion (heavy drums).\n"
            "4. NOISE-STRIP: ABSOLUTELY PURGE all non-musical nouns (e.g., 'hair', 'mushroom', 'houses', 'snouts'). These cause digital 'static' and hallucinations. Replace them with instrumental equivalents.\n\n"
            
            "PHASE 2: STRUCTURAL TEMPLATES (Prompt2MusicBench):\n"
            "Choose the most effective structure for the race's 'vibe' from these options:\n"
            "- mood_emphasized: [Mood] [Theme] at [BPM] centered on [Instrument].\n"
            "- instrument_first: [Instrument]-led [Genre] piece at [BPM], [Mood] in style.\n"
            "- creative_poetic: Create [Theme] music in a [Mood] style, using [Instrument], [BPM].\n"
            "- genre_first: [Genre] at [BPM], performed with [Instrument], evoking a [Mood] feeling.\n\n"
            
            "FEW-SHOT EXAMPLE:\n"
            "Input: 'Tiny winged faeries in a glowing forest making chimes.'\n"
            "Analytical Logic: Tiny (High-register) + Glowing (Shimmering pads) + Chimes (Metallic bells).\n"
            "Template Choice: creative_poetic\n"
            "Output: 'Create magical forest music in a whimsical style, using shimmering synth bells and high-pitched arpeggios, 150 BPM, F Major.'\n\n"
            
            "OUTPUT REQUIREMENT:\n"
            "Return a structured JSON object. The 'baseline_prompt' must prefix the result with '16-bit SNES chiptune '."
        )

    def translate(self, raw_prompt: str) -> dict:
        try:
            response = self.client.models.generate_content(
                #model="gemini-3.1-pro-preview",
                # Change the model from "gemini-3.1-pro-preview" to:
                model="gemini-3.1-flash",  # Highly efficient, higher free-tier limits
                contents=raw_prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=self.system_instruction,
                    response_mime_type="application/json",
                    response_schema=PromptTranslatorOutput,
                    temperature=0.9, # Higher temperature = more creative; hallucination can be desirable in this context
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Error during Gemini translation: {e}")
            return {
                "baseline_prompt": f"MIDI-orchestral {raw_prompt}",
                "improved_parameters": {"BPM": "120", "Key": "C Minor", "Lead": "Strings"}
            }
