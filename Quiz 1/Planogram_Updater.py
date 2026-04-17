# pip install diffusers transformers accelerate opencv-python Pillow
# pip install --upgrade torch torchvision sympy
# pip install gTTS
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler
from gtts import gTTS

# Setup output
local_output_path = os.path.join('.', 'Store Output')
os.makedirs(local_output_path, exist_ok=True)
print(f"Local output path set to: {local_output_path}")

# Setup briefing audio
briefing_text = "The cold-medicine aisle is empty. I've updated the planogram to fill the top shelf with our generic store-brand labels until the shipment arrives."
tts = gTTS(briefing_text)
tts.save(os.path.join(local_output_path, "manager_briefing.mp3"))

# --- Helper Functions ---
def create_masks(image):
    """
    Refined masks to prevent merchandise from appearing on the ceiling.
    """
    width, height = image.size
    
    # Define vertical boundaries (as percentages of total height)
    ceiling_boundary = int(height * 0.18)  # Starts below the lights
    shelf_split = int(height * 0.42)      # Boundary between top shelf and others
    bottom_boundary = height              # End of image
    
    # 1. Top Shelf Mask (Only the top red section)
    top_mask = Image.new("L", (width, height), 0)
    top_draw = ImageDraw.Draw(top_mask)
    # Draw white rectangle ONLY between ceiling and first shelf split
    top_draw.rectangle([0, ceiling_boundary, width, shelf_split], fill=255)
    
    # 2. Bottom Shelves Mask (From the split to the floor)
    bottom_mask = Image.new("L", (width, height), 0)
    bottom_draw = ImageDraw.Draw(bottom_mask)
    bottom_draw.rectangle([0, shelf_split, width, bottom_boundary], fill=255)
    
    return top_mask, bottom_mask
# ---------------------

def create_shelf_masks(image):
    width, height = image.size
    masks = {}
    
    # Define vertical boundaries for 5 shelves (y_min, y_max)
    # These are calibrated to stay within the red background areas
    # shelf_bounds = {
    #     "shelf_1": (0.18, 0.31),
    #     "shelf_2": (0.33, 0.45),
    #     "shelf_3": (0.47, 0.59),
    #     "shelf_4": (0.61, 0.74),
    #     "shelf_5": (0.76, 0.92)
    # }
    # Updated vertical boundaries for the 7-shelf layout
    # (y_start, y_end) as percentages of total image height
    shelf_bounds = {
        "shelf_1": (0.16, 0.25),  # Top-most shelf (below light bar)
        "shelf_2": (0.27, 0.36),
        "shelf_3": (0.38, 0.47),
        "shelf_4": (0.49, 0.58),
        "shelf_5": (0.60, 0.69),
        "shelf_6": (0.71, 0.80),
        "shelf_7": (0.82, 0.91)   # Bottom-most shelf (above base)
    }

    for shelf_name, (y_start, y_end) in shelf_bounds.items():
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Calculate pixel coordinates
        top = int(height * y_start)
        bottom = int(height * y_end)
        
        # Draw the "active" zone for this specific shelf
        draw.rectangle([0, top, width, bottom], fill=255)
        masks[shelf_name] = mask
        
    return masks



# --- Pipeline Loading ---
def setup_pipeline():
    print("Loading Stable Diffusion Inpainting model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=dtype)
    inpaint_pipe.scheduler = UniPCMultistepScheduler.from_config(inpaint_pipe.scheduler.config)

    if device == "cuda":
        inpaint_pipe.enable_model_cpu_offload()
    else:
        inpaint_pipe = inpaint_pipe.to(device)

    return inpaint_pipe

# --- Main App ---
def main_pharmacy_scenario():
    # 1. Setup Pipeline
    inpaint_pipe = setup_pipeline()

    # 2. Load Visual Anchor
    # Standardizing to 512x512 is crucial for SD 1.5 mask alignment
    template_path = os.path.join('.', 'Template', 'shelving_template.png')
    anchor_image = Image.open(template_path).convert("RGB").resize((512, 512))
    
    # 3. Get the Dictionary of Shelf Masks
    # This uses the percentages we discussed to isolate the red zones
    masks = create_shelf_masks(anchor_image)

    # 4. Generate the "Base Layout" (Stocking all shelves)
    # We start with the empty anchor and progressively add stock shelf by shelf
    current_working_image = anchor_image
    
    shelf_order = ["shelf_1", "shelf_2", "shelf_3", "shelf_4", "shelf_5", "shelf_6", "shelf_7"]
    
    print("Populating Base Layout shelf by shelf...")
    for shelf in shelf_order:
        print(f"Stocking {shelf}...")
        prompt = ("Highly organized colorful pharmacy products, medicine boxes, "
                  "neatly stacked on a retail shelf, professional lighting, 8k.")
        
        current_working_image = inpaint_pipe(
            prompt=prompt,
            image=current_working_image,
            mask_image=masks[shelf],
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]

    current_working_image.save(os.path.join(local_output_path, "pharmacy_base_full.png"))

    # 5. Create the "Updated Planogram" (Swapping only Shelf 1)
    # We take the full base layout and overwrite JUST the top shelf with generics
    print("Updating Top Shelf (Shelf 1) with generic brand labels...")
    
    update_prompt = ("Organized white-label generic medicine bottles, simple "
                     "healthcare packaging, pharmacy store brand, photorealistic.")
    
    updated_planogram = inpaint_pipe(
        prompt=update_prompt,
        image=current_working_image, # Use the already stocked image as the base
        mask_image=masks["shelf_1"],  # Only modify the top shelf
        num_inference_steps=30,
        guidance_scale=8.0
    ).images[0]
    
    updated_planogram.save(os.path.join(local_output_path, "pharmacy_updated_planogram.png"))
    print("Process complete. Files saved to 'Store Output'.")

# def main_pharmacy_scenario():
#     # 0. Load Pipe
#     inpaint_pipe = setup_pipeline()

#     # 1. Load Visual Anchor
#     template_path = os.path.join('.', 'Template', 'shelving_template.jpg')
#     if not os.path.exists(template_path):
#         raise FileNotFoundError(f"Could not find anchor template at {template_path}")
    
#     # Open and standardize to 512x512 for SD 1.5 stability
#     anchor_image = Image.open(template_path).convert("RGB").resize((512, 512))
    
#     # 2. Create complementary masks
#     top_shelf_mask, bottom_shelves_mask = create_masks(anchor_image)

#     # 3. Intermediate Template: Modify anchor to add stock to all but ONE shelf
#     print("Generating Intermediate Template (Bottom Shelves Stocked)...")
#     intermediate_prompt = ("Fully filled bottom shelves of a pharmacy display, brightly lit retail lighting, "
#                            "highly organized colorful generic medicine boxes and standard health items, 8k, photorealistic.")
    
#     intermediate_template = inpaint_pipe(
#         prompt=intermediate_prompt,
#         image=anchor_image,
#         mask_image=bottom_shelves_mask,
#         num_inference_steps=30,
#         guidance_scale=8.0
#     ).images[0]
#     intermediate_template.save(os.path.join(local_output_path, "pharmacy_intermediate_template.png"))
#     print("Intermediate Template saved.")

#     # 4. Base Layout: Modify intermediate to add items to the ONE shelf
#     print("Generating Base Layout (All Shelves Stocked)...")
#     base_prompt = ("Fully filled top shelf of a pharmacy display, brightly lit retail lighting, "
#                    "highly organized colorful general brand medicine boxes and standard health items, 8k, photorealistic.")

#     base_image = inpaint_pipe(
#         prompt=base_prompt,
#         image=intermediate_template,
#         mask_image=top_shelf_mask,
#         num_inference_steps=30,
#         guidance_scale=8.0
#     ).images[0]
#     base_image.save(os.path.join(local_output_path, "pharmacy_base_layout.png"))
#     print("Pharmacy Base Layout saved.")

#     # 5. Updated Layout: Modify intermediate to add DIFFERENT items to the ONE shelf
#     print("Generating Updated Planogram (Top Shelf Generics)...")
#     update_prompt = ("Fully filled top shelf of a pharmacy display, brightly lit retail lighting, "
#                      "highly organized generic store-brand white-label medicine bottles, simple healthcare packaging, 8k, photorealistic.")

#     updated_image = inpaint_pipe(
#         prompt=update_prompt,
#         image=intermediate_template,
#         mask_image=top_shelf_mask,
#         num_inference_steps=30,
#         guidance_scale=8.0
#     ).images[0]
#     updated_image.save(os.path.join(local_output_path, "pharmacy_updated_planogram.png"))
#     print("Updated planogram saved to Local.")

def main():
    main_pharmacy_scenario()

if __name__ == '__main__':
    main()
