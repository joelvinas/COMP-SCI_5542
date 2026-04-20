# pip install diffusers transformers accelerate opencv-python Pillow
# pip install --upgrade torch torchvision sympy
# pip install gTTS
import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler
from gtts import gTTS

import pandas as pd
import random
import glob

# Setup output
local_output_path = os.path.join('.', 'Store Output')
os.makedirs(local_output_path, exist_ok=True)
print(f"Local output path set to: {local_output_path}")

# Setup briefing audio
briefing_text = "The cold-medicine aisle is empty. I've updated the planogram to fill the top shelf with our generic store-brand labels until the shipment arrives."
tts = gTTS(briefing_text)
tts.save(os.path.join(local_output_path, "manager_briefing.mp3"))

# --- Helper Functions ---
def create_shelf_masks(image):
    width, height = image.size
    masks = {}
    shelf_bounds = {
        "shelf_1": (0.16, 0.25),
        "shelf_2": (0.27, 0.36),
        "shelf_3": (0.38, 0.47),
        "shelf_4": (0.49, 0.58),
        "shelf_5": (0.60, 0.69),
        "shelf_6": (0.71, 0.80),
        "shelf_7": (0.82, 0.91)
    }

    for shelf_name, (y_start, y_end) in shelf_bounds.items():
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        top = int(height * y_start)
        bottom = int(height * y_end)
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

def generate_svg_layout(filename, background_path, shelf_assignments, image_size=(512, 512)):
    """
    Generates an SVG leveraging a background base and layering SKUs.
    scale_factor allows adapting visual_width to pixels, e.g., 1 inch = 6 pixels.
    """
    svg_path = os.path.join(local_output_path, filename)
    width, height = image_size
    
    # Needs to be a relative path from the SVG for browsers, but for standalone we use file paths or relative to output
    # Since background is in the same folder, just use the filename
    bg_rel_path = os.path.basename(background_path)

    svg_lines = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg_lines.append(f'  <image href="{bg_rel_path}" x="0" y="0" width="{width}" height="{height}" />')
    
    for shelf, skus in shelf_assignments.items():
        for sku in skus:
            # sku contains: sku_id, x_pos, y_pos, block_width, block_height
            sku_href = f"../sku_assets/{sku['sku_id']}.png"
            svg_lines.append(f'  <image href="{sku_href}" x="{sku["x_pos"]}" y="{sku["y_pos"]}" width="{sku["block_width"]}" height="{sku["block_height"]}" />')
            
    svg_lines.append("</svg>")
    
    with open(svg_path, 'w') as f:
        f.write("\n".join(svg_lines))
    print(f"Generated {svg_path}")

def generate_animated_svg(filename, background_path, intermediate_assignments, base_shelf_3, update_shelf_3, image_size=(512, 512)):
    svg_path = os.path.join(local_output_path, filename)
    width, height = image_size
    bg_rel_path = os.path.basename(background_path)

    svg_lines = []
    svg_lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg_lines.append(f'  <style>')
    svg_lines.append(f'    .base-items {{ animation: fadeOutBase 8s infinite; }}')
    svg_lines.append(f'    .update-items {{ opacity: 0; animation: fadeInUpdate 8s infinite; }}')
    svg_lines.append(f'    @keyframes fadeOutBase {{')
    svg_lines.append(f'      0%, 40% {{ opacity: 1; }}')
    svg_lines.append(f'      50%, 90% {{ opacity: 0; }}')
    svg_lines.append(f'      100% {{ opacity: 1; }}')
    svg_lines.append(f'    }}')
    svg_lines.append(f'    @keyframes fadeInUpdate {{')
    svg_lines.append(f'      0%, 40% {{ opacity: 0; }}')
    svg_lines.append(f'      50%, 90% {{ opacity: 1; }}')
    svg_lines.append(f'      100% {{ opacity: 0; }}')
    svg_lines.append(f'    }}')
    svg_lines.append(f'  </style>')
    
    # Layer Background
    svg_lines.append(f'  <image href="{bg_rel_path}" x="0" y="0" width="{width}" height="{height}" />')
    
    # Layer fixed static shelves (everything from intermediate)
    for shelf, skus in intermediate_assignments.items():
        if shelf == "shelf_3": continue
        for sku in skus:
            sku_href = f"../sku_assets/{sku['sku_id']}.png"
            svg_lines.append(f'  <image href="{sku_href}" x="{sku["x_pos"]}" y="{sku["y_pos"]}" width="{sku["block_width"]}" height="{sku["block_height"]}" />')
            
    # Base Layout Shelf 3 (Fading Out)
    svg_lines.append(f'  <g class="base-items">')
    for sku in base_shelf_3:
        sku_href = f"../sku_assets/{sku['sku_id']}.png"
        svg_lines.append(f'    <image href="{sku_href}" x="{sku["x_pos"]}" y="{sku["y_pos"]}" width="{sku["block_width"]}" height="{sku["block_height"]}" />')
    svg_lines.append(f'  </g>')

    # Updated Layout Shelf 3 (Fading In)
    svg_lines.append(f'  <g class="update-items">')
    for sku in update_shelf_3:
        sku_href = f"../sku_assets/{sku['sku_id']}.png"
        svg_lines.append(f'    <image href="{sku_href}" x="{sku["x_pos"]}" y="{sku["y_pos"]}" width="{sku["block_width"]}" height="{sku["block_height"]}" />')
    svg_lines.append(f'  </g>')

    svg_lines.append("</svg>")
    
    with open(svg_path, 'w') as f:
        f.write("\n".join(svg_lines))
    print(f"Generated {svg_path}")

def allocate_skus_to_shelves(shelf_bounds, df, skip_shelves=None):
    """
    Creates a dict of items placed on each shelf.
    Prioritizes shelf_3, shelf_4, shelf_5 for high velocity/margin.
    """
    if skip_shelves is None: skip_shelves = []
    
    center_shelves = ["shelf_3", "shelf_4", "shelf_5"]
    
    # Sort for center: combine velocity & margin
    df_center = df.sort_values(by=['Velocity', 'Margin'], ascending=[False, False])
    # The rest are lower velocity/margin
    df_rest = df.sort_values(by=['Velocity', 'Margin'], ascending=[True, True])
    
    # Convert visually inches to pixels assuming 8 inch max height
    IMG_WIDTH = 512
    IMG_HEIGHT = 512
    SCALE_PX_PER_INCH = 6 # e.g. 50 pixels tall = 8 inch height
    
    shelf_assignments = {}
    for shelf_name, (y_start, y_end) in shelf_bounds.items():
        if shelf_name in skip_shelves:
            continue
            
        shelf_assignments[shelf_name] = []
        target_fill_width = IMG_WIDTH * 0.8 # 80% full
        current_x = 0
        
        # Select pool
        pool = df_center if shelf_name in center_shelves else df_rest
        
        # Shelf pixel boundaries
        shelf_px_top = int(IMG_HEIGHT * y_start)
        shelf_px_bottom = int(IMG_HEIGHT * y_end)
        shelf_px_height = shelf_px_bottom - shelf_px_top
        
        # We place randomly from the pool until width is around 80% 
        # but randomly skip spaces to fill 80% out of 100% physically.
        # i.e., total actual item width should be 0.8 * IMG_WIDTH.
        items_placed_width = 0
        current_x = 0
        
        while current_x < IMG_WIDTH:
            # If we need to leave a gap (20% chance to leave a gap of random size)
            if random.random() < 0.2 and current_x > 0:
                current_x += random.randint(5, 15) # empty space
            
            # Select random item from top 10-15
            row = pool.head(20).sample(1).iloc[0]
            sku_width_px = row['Visual_Width'] * SCALE_PX_PER_INCH
            sku_height_px = row['Visual_Height'] * SCALE_PX_PER_INCH
            
            # If item width brings us way past IMG_WIDTH, break
            if current_x + sku_width_px > IMG_WIDTH:
                break
                
            # Items sit on the bottom of the shelf
            y_pos = shelf_px_bottom - sku_height_px
            
            shelf_assignments[shelf_name].append({
                "sku_id": row['SKU'],
                "x_pos": round(current_x, 2),
                "y_pos": round(y_pos, 2),
                "block_width": round(sku_width_px, 2),
                "block_height": round(sku_height_px, 2)
            })
            
            current_x += sku_width_px
            items_placed_width += sku_width_px

    return shelf_assignments

# --- Main App ---
def main_pharmacy_scenario():
    # 0. Load SKU Data
    csv_path = os.path.join('data', 'pharmacy_stock_items.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing stock data at {csv_path}")
    df = pd.read_csv(csv_path)

    # 1. Setup Pipeline for background enhancement
    inpaint_pipe = setup_pipeline()

    # 2. Load Visual Anchor
    template_path = os.path.join('.', 'Template', 'shelving_template.png')
    anchor_image = Image.open(template_path).convert("RGB").resize((512, 512))
    masks = create_shelf_masks(anchor_image)
    
    # We create a generic mask covering all shelves to translate it into an "Intermediate Background"
    width, height = anchor_image.size
    full_mask = Image.new("L", (width, height), 0)
    for mask in masks.values():
        # Composite masks to create one big mask
        full_mask = Image.composite(Image.new("L", (width, height), 255), full_mask, mask)

    print("Translating template into realistic shelf background...")
    prompt = "Clean empty pharmacy shelves, realistic lighting, highly detailed retail texture, brightly lit."
    intermediate_bg = inpaint_pipe(
        prompt=prompt,
        image=anchor_image,
        mask_image=full_mask,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    
    bg_out_path = os.path.join(local_output_path, "pharmacy_intermediate_background.png")
    intermediate_bg.save(bg_out_path)
    
    # 6-shelf layout
    shelf_bounds = {
        "shelf_1": (0.16, 0.30),
        "shelf_2": (0.32, 0.38), 
        "shelf_3": (0.40, 0.47),
        "shelf_4": (0.49, 0.58), 
        "shelf_5": (0.60, 0.69), 
        "shelf_6": (0.71, 0.80)
    }

    # Step 1: Intermediate Template (All shelves EXCEPT shelf 3)
    print("Generating Intermediate Template SVG (Skipping Shelf 3)...")
    intermediate_assignments = allocate_skus_to_shelves(shelf_bounds, df, skip_shelves=["shelf_3"])
    generate_svg_layout("pharmacy_intermediate_template.svg", bg_out_path, intermediate_assignments)
    
    # Step 2: Base Layout (Intermediate Template + Shelf 3 added)
    print("Generating Base Layout SVG (Shelf 3 Stocked)...")
    base_assignments = intermediate_assignments.copy()
    base_shelf_3 = allocate_skus_to_shelves(
        { "shelf_3": shelf_bounds["shelf_3"] }, df
    )
    base_assignments["shelf_3"] = base_shelf_3["shelf_3"]
    generate_svg_layout("pharmacy_base_layout.svg", bg_out_path, base_assignments)
    
    # Step 3: Updated Layout (Intermediate Template + Different SKUs on Shelf 3)
    print("Generating Updated Planogram SVG (Different items on Shelf 3)...")
    update_assignments = intermediate_assignments.copy()
    # Different items by shuffling the DF slightly
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    update_shelf_3 = allocate_skus_to_shelves(
        { "shelf_3": shelf_bounds["shelf_3"] }, df_shuffled
    )
    update_assignments["shelf_3"] = update_shelf_3["shelf_3"]
    generate_svg_layout("pharmacy_updated_planogram.svg", bg_out_path, update_assignments)
    
    # Step 4: Animated SVG Video
    print("Generating Animated SVG Transition (Base -> Update)...")
    generate_animated_svg("pharmacy_transition_video.svg", bg_out_path, intermediate_assignments, base_shelf_3["shelf_3"], update_shelf_3["shelf_3"])

    print("Process complete. Files saved to 'Store Output'.")

def main():
    main_pharmacy_scenario()

if __name__ == '__main__':
    main()

