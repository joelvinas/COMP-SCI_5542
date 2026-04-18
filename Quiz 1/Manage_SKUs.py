import os
import pandas as pd
import random
import torch
from diffusers import StableDiffusionPipeline
from rembg import remove # Required: pip install rembg
from PIL import Image

CSV_FILE_PATH = os.path.join('data', 'pharmacy_stock_items.csv')

def generate_stock_csv(num_items=50):
    products = [
        "Ibuprofen", "Acetaminophen", "Cough Syrup", "Vitamin C", "Multivitamin",
        "Band-Aids", "Hand Sanitizer", "Thermometer", "Antacid", "Eye Drops",
        "Allergy Relief", "Sleep Aid", "Probiotic", "Lip Balm", "Sunscreen"
    ]
    
    data = []
    for i in range(num_items):
        sku = str(random.randint(100000, 999999))
        name = f"{random.choice(products)} {random.choice(['Extra Strength', 'Maximum', 'Gentle', 'Daily'])}"
        
        # Financials
        cost = round(random.uniform(1.0, 10.0), 2)
        # 120% to 300% margin against cost (multiplier is 2.2x to 4.0x)
        price = round(min(cost * random.uniform(2.2, 4.0), 20.0), 2)
        margin = round((price - cost) / price, 2)
        
        # Visual Dimensions (Constraint: Must fit in 8" shelf height)
        visual_height = round(random.uniform(2.0, 7.5), 2)
        visual_width = round(random.uniform(1.5, 5.0), 2)
        
        # Velocity Ranking (1-10)
        # We assign higher velocity to items intended for eye-level (Shelves 4-5)
        velocity = random.randint(1, 10)
        
        data.append([sku, name, cost, price, margin, visual_height, visual_width, velocity])

    df = pd.DataFrame(data, columns=[
        'SKU', 'Name', 'Cost', 'Price', 'Margin', 'Visual_Height', 'Visual_Width', 'Velocity'
    ])
    os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
    df.to_csv(CSV_FILE_PATH, index=False)
    print(f"CSV generated: {CSV_FILE_PATH}")

def generate_sku_assets(csv_path=CSV_FILE_PATH):
    df = pd.read_csv(csv_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype)
    pipe.to(device)

    os.makedirs('sku_assets', exist_ok=True)

    # Define the fixed negative prompt to enforce 2D flatness
    neg_prompt = ("3D, isometric, perspective, side-view, angle, depth, shadow, "
                  "floor, table, ground, human, hands, multiple objects, "
                  "blurry, watermark, distorted, top-down view")

    for index, row in df.iterrows():
        # Technical staging prompt

        # 1. Calculate target dimensions (Standard SD 1.5 works best near 512px)
        # We use a base scale to keep the aspect ratio from the CSV
        base_dim = 512
        ratio = row['Visual_Width'] / row['Visual_Height']
        
        if ratio > 1:
            width, height = base_dim, int(base_dim / ratio)
        else:
            width, height = int(base_dim * ratio), base_dim
            
        # Ensure dimensions are multiples of 8 for SD compatibility
        width = (width // 8) * 8
        height = (height // 8) * 8

        # 2. Refined "Flat-Only" Prompting
        # Using 'orthographic' and 'texture' helps avoid the 3D 'box' trap
        pos_prompt = (f"A flat 2D orthographic texture of a {row['Name']} medicine label, "
                      "centered, straight, white background, flat design, no depth, "
                      "high-quality pharmaceutical packaging graphic.")
        
        neg_prompt = ("3D, perspective, side view, angle, depth, shadow, box shape, "
                      "product shot, isometric, photorealistic depth, person, hands.")

        # 3. Generate with custom height/width
        image = pipe(
            prompt=pos_prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=30,
            guidance_scale=9.5,
            height=height, # Explicitly setting canvas height
            width=width    # Explicitly setting canvas width
        ).images[0]

        # 4. Final Cleanup
        # This converts the 'grey' or 'floor' area into true transparency (Alpha Channel)
        clean_asset = remove(image)
        clean_asset.save(f"sku_assets/{row['SKU']}.png")
        print(f"Generated flat asset for SKU {row['SKU']} at {width}x{height}")

#generate_stock_csv()    
generate_sku_assets()
