Mission: Implement a dynamic PromptTranslator class that utilizes the Gemini 3.1 Pro API to convert abstract racial descriptions into structured musical prompts.
Core Logic Requirements:
Zero-Hardcoding Rule: Do not use specific race names (Goblins, Ogres) in the logic. Use a Semantic Mapping Pattern to extract attributes.
Physical-to-Musical Extraction:
Map Size/Power (e.g., "towering," "tiny") to Frequency/BPM (e.g., "Heavy sub-bass" vs "High-pitched staccato").
Map Environment (e.g., "jungle," "underground") to Acoustics (e.g., "Short dry reverb" vs "Spacious dark hall reverb").
Map Culture (e.g., "tribal," "hoarding") to Instrumentation (e.g., "Woodblocks/Percussion" vs "Metallic/Found-object synths").
The "Noise-Strip" Filter: Automatically remove "clipping-prone" adjectives (e.g., physical colors, specific nouns like "trash," and abstract actions like "hoarding") that lack a direct musical signal.
Output Signature: The function must return a JSON object containing:
baseline_prompt: The cleaned musical instruction.
improved_parameters: BPM, Key, and Lead instrument suggestions derived from the text's "energy".

System Instruction for the Translator:
Role: You are a Music Theory Specialist for 16-bit Video Game Music.
Task: Translate cultural flavor text into a MusicGen Signature.
Avoidance Rules: Never include the words "laughter," "coins," or "trash." Replace them with "Rhythmic metallic high-pitched square waves" or "Randomized woodblock staccato".
Formatting: Prefix all outputs with 16-bit SNES chiptune.