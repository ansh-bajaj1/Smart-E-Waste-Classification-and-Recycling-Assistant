"""
generate_dataset.py
-------------------
ReBoot AI: Smart E-Waste Classification and Recycling Assistant
Generates a reproducible synthetic dataset of common electronic waste items
using Pillow (PIL) at 96x96 resolution.

Classes:
1. mobile_phone
2. charger_cable
3. earphones
4. circuit_board

Dataset splits:
- 70% Train (~112 images per class)
- 15% Validation (~24 images per class)
- 15% Test (~24 images per class)

NOTE:
These images are synthetically generated for educational laboratory demonstration.
They are not real photographs and have domain limitations when applied to real-world waste.
"""

import os
import random
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# Set fixed seeds for reproducible generation
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Configuration
IMAGE_SIZE = (96, 96)
CLASSES = ["mobile_phone", "charger_cable", "earphones", "circuit_board"]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Image counts per class
COUNTS = {
    "train": 112,
    "val": 24,
    "test": 24
}

def random_neutral_bg(size):
    """Generate a light or dark realistic backdrop with subtle noise."""
    base_col = random.choice([
        (235, 238, 240),
        (245, 245, 247),
        (220, 224, 230),
        (210, 215, 220),
        (35, 38, 42),
        (48, 52, 58)
    ])
    img = Image.new("RGB", size, base_col)
    draw = ImageDraw.Draw(img)
    # Subtle background texture
    for _ in range(15):
        x = random.randint(0, size[0])
        y = random.randint(0, size[1])
        r = random.randint(5, 25)
        var = random.randint(-8, 8)
        col = tuple(max(0, min(255, c + var)) for c in base_col)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=col)
    return img

def draw_mobile_phone(size):
    """Draw a synthetic mobile phone with screen, bezel, speaker, and camera."""
    img = random_neutral_bg(size)
    draw = ImageDraw.Draw(img)

    w, h = size
    # Phone dimensions
    pw = random.randint(38, 46)
    ph = random.randint(64, 76)
    px = (w - pw) // 2 + random.randint(-4, 4)
    py = (h - ph) // 2 + random.randint(-4, 4)

    # Body color (black, midnight blue, rose gold, silver, emerald)
    body_col = random.choice([
        (25, 27, 30),
        (15, 25, 45),
        (180, 150, 140),
        (195, 200, 205),
        (20, 45, 35)
    ])
    # Body rounded rectangle
    draw.rounded_rectangle([px, py, px + pw, py + ph], radius=7, fill=body_col, outline=(15, 15, 15), width=1)

    # Decide if front display or rear backplate is facing camera
    is_front = random.random() > 0.35

    if is_front:
        # Screen bezel
        margin = random.randint(3, 5)
        sx0, sy0 = px + margin, py + margin + 3
        sx1, sy1 = px + pw - margin, py + ph - margin - 3

        # Screen color (lit screen or dark locked screen)
        if random.random() > 0.4:
            screen_col = random.choice([
                (20, 50, 80),
                (30, 70, 60),
                (50, 30, 70),
                (10, 15, 25)
            ])
            draw.rectangle([sx0, sy0, sx1, sy1], fill=screen_col)
            # Draw simple app icon squares or wallpaper line
            for ix in range(sx0 + 4, sx1 - 4, 8):
                for iy in range(sy0 + 6, sy1 - 8, 10):
                    if random.random() > 0.4:
                        icon_col = (random.randint(100, 240), random.randint(100, 240), random.randint(100, 240))
                        draw.rounded_rectangle([ix, iy, ix + 5, iy + 5], radius=1, fill=icon_col)
        else:
            # Glossy black locked screen with reflection
            draw.rectangle([sx0, sy0, sx1, sy1], fill=(12, 14, 18))
            draw.line([sx0 + 2, sy0 + 2, sx1 - 8, sy1 - 10], fill=(45, 50, 60), width=1)

        # Top speaker slit & selfie camera notch
        draw.line([px + pw // 2 - 4, py + 2, px + pw // 2 + 4, py + 2], fill=(60, 60, 60), width=1)
        draw.ellipse([px + pw // 2 - 1, py + margin + 1, px + pw // 2 + 1, py + margin + 3], fill=(5, 5, 5))
    else:
        # Back camera module
        cam_box_w, cam_box_h = 14, 18
        cx0, cy0 = px + 4, py + 5
        draw.rounded_rectangle([cx0, cy0, cx0 + cam_box_w, cy0 + cam_box_h], radius=3, fill=(10, 10, 12))
        # Camera lenses
        draw.ellipse([cx0 + 2, cy0 + 2, cx0 + 6, cy0 + 6], fill=(50, 70, 90), outline=(180, 180, 180))
        draw.ellipse([cx0 + 2, cy0 + 9, cx0 + 6, cy0 + 13], fill=(50, 70, 90), outline=(180, 180, 180))
        draw.ellipse([cx0 + 8, cy0 + 5, cx0 + 12, cy0 + 9], fill=(220, 220, 160)) # flash

    # Apply random subtle rotation
    angle = random.uniform(-14, 14)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(230, 230, 230))
    return img

def draw_charger_cable(size):
    """Draw a wall adapter charging brick with curved cable and plug connector."""
    img = random_neutral_bg(size)
    draw = ImageDraw.Draw(img)

    w, h = size
    # Brick color (white or black)
    is_white = random.random() > 0.4
    brick_col = (240, 240, 245) if is_white else (30, 30, 35)
    outline_col = (180, 180, 190) if is_white else (10, 10, 10)
    cable_col = (235, 235, 240) if is_white else (35, 35, 40)

    # Charger brick position
    bx0 = random.randint(10, 25)
    by0 = random.randint(12, 28)
    bw = random.randint(22, 30)
    bh = random.randint(24, 32)

    # Wall plug prongs (metallic silver)
    prong_col = (190, 195, 205)
    draw.rectangle([bx0 - 7, by0 + 5, bx0, by0 + 9], fill=prong_col, outline=(120, 120, 130))
    draw.rectangle([bx0 - 7, by0 + bh - 11, bx0, by0 + bh - 7], fill=prong_col, outline=(120, 120, 130))

    # Main adapter body
    draw.rounded_rectangle([bx0, by0, bx0 + bw, by0 + bh], radius=4, fill=brick_col, outline=outline_col, width=2)
    # USB port or seam on brick
    draw.rectangle([bx0 + bw - 5, by0 + bh // 2 - 3, bx0 + bw, by0 + bh // 2 + 3], fill=(60, 60, 65))

    # Wavy/curved cable coming out of brick
    start_pt = (bx0 + bw, by0 + bh // 2)
    pts = [start_pt]
    curr_x, curr_y = start_pt
    
    # 3-4 control nodes for winding cable
    node1 = (curr_x + random.randint(10, 25), curr_y + random.randint(15, 30))
    node2 = (node1[0] - random.randint(10, 25), node1[1] + random.randint(15, 25))
    node3 = (w - random.randint(15, 30), h - random.randint(15, 30))

    # Interpolate curve
    curve_points = []
    for t in np.linspace(0, 1, 25):
        # Cubic bezier approx
        x = (1-t)**3 * start_pt[0] + 3*(1-t)**2 * t * node1[0] + 3*(1-t)*t**2 * node2[0] + t**3 * node3[0]
        y = (1-t)**3 * start_pt[1] + 3*(1-t)**2 * t * node1[1] + 3*(1-t)*t**2 * node2[1] + t**3 * node3[1]
        curve_points.append((x, y))

    for i in range(len(curve_points) - 1):
        draw.line([curve_points[i], curve_points[i+1]], fill=cable_col, width=3)

    # Connector tip at end of cable
    end_x, end_y = curve_points[-1]
    tip_col = (195, 200, 205)
    draw.rectangle([end_x - 4, end_y - 3, end_x + 4, end_y + 3], fill=outline_col)
    draw.rectangle([end_x + 4, end_y - 2, end_x + 10, end_y + 2], fill=tip_col, outline=(140, 140, 150))

    # Slight angle rotation
    angle = random.uniform(-10, 10)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(230, 230, 230))
    return img

def draw_earphones(size):
    """Draw synthetic wired earphones with two earbuds, stems, and Y-split cable."""
    img = random_neutral_bg(size)
    draw = ImageDraw.Draw(img)

    w, h = size
    is_white = random.random() > 0.5
    bud_col = (245, 245, 250) if is_white else (30, 30, 35)
    cable_col = (230, 230, 235) if is_white else (45, 45, 50)
    tip_col = (160, 165, 175) if is_white else (15, 15, 18)

    # Left earbud
    lx = random.randint(18, 30)
    ly = random.randint(14, 25)
    draw.ellipse([lx - 6, ly - 6, lx + 6, ly + 6], fill=bud_col, outline=tip_col, width=1)
    draw.ellipse([lx - 3, ly - 3, lx + 3, ly + 3], fill=tip_col)
    draw.rectangle([lx - 2, ly + 4, lx + 2, ly + 14], fill=bud_col)

    # Right earbud
    rx = random.randint(62, 76)
    ry = random.randint(14, 25)
    draw.ellipse([rx - 6, ry - 6, rx + 6, ry + 6], fill=bud_col, outline=tip_col, width=1)
    draw.ellipse([rx - 3, ry - 3, rx + 3, ry + 3], fill=tip_col)
    draw.rectangle([rx - 2, ry + 4, rx + 2, ry + 14], fill=bud_col)

    # Y-joint node
    yx = (lx + rx) // 2 + random.randint(-4, 4)
    yy = random.randint(46, 56)

    # Cable from left bud to Y-joint
    draw.line([(lx, ly + 14), (lx + 2, ly + 26), (yx, yy)], fill=cable_col, width=2)
    # Cable from right bud to Y-joint
    draw.line([(rx, ry + 14), (rx - 2, ry + 26), (yx, yy)], fill=cable_col, width=2)

    # Y-joint capsule
    draw.rounded_rectangle([yx - 2, yy - 3, yx + 2, yy + 4], radius=2, fill=(80, 80, 85))

    # Main cable going down to jack
    jack_x = random.randint(25, 65)
    jack_y = random.randint(80, 90)
    draw.line([(yx, yy + 4), (yx + random.randint(-8, 8), (yy + jack_y) // 2), (jack_x, jack_y - 6)], fill=cable_col, width=2)

    # 3.5mm audio jack plug
    draw.rectangle([jack_x - 2, jack_y - 6, jack_x + 2, jack_y], fill=(60, 65, 70))
    # Gold/silver pin
    draw.rectangle([jack_x - 1, jack_y, jack_x + 1, jack_y + 5], fill=(215, 180, 50))

    angle = random.uniform(-12, 12)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(230, 230, 230))
    return img

def draw_circuit_board(size):
    """Draw a green PCB circuit board with IC chip, metallic pins, solder traces, and capacitors."""
    img = Image.new("RGB", size, (225, 228, 230))
    draw = ImageDraw.Draw(img)

    w, h = size
    # PCB board rectangle
    margin = random.randint(5, 10)
    board_x0, board_y0 = margin, margin
    board_x1, board_y1 = w - margin, h - margin

    # Green or dark teal PCB substrate
    pcb_color = random.choice([
        (12, 92, 54),
        (18, 110, 68),
        (8, 75, 42),
        (10, 65, 75)
    ])
    draw.rectangle([board_x0, board_y0, board_x1, board_y1], fill=pcb_color, outline=(5, 40, 25), width=2)

    # Gold/copper solder traces (circuit lines)
    trace_col = random.choice([(190, 160, 60), (140, 180, 120), (210, 175, 80)])
    for _ in range(12):
        tx0 = random.randint(board_x0 + 4, board_x1 - 4)
        ty0 = random.randint(board_y0 + 4, board_y1 - 4)
        # Manhattan wiring (orthogonal lines)
        mid_x = tx0 + random.choice([-15, -8, 8, 15])
        mid_y = ty0 + random.choice([-15, -8, 8, 15])
        draw.line([(tx0, ty0), (mid_x, ty0), (mid_x, mid_y)], fill=trace_col, width=1)
        # Solder pad at end
        draw.ellipse([mid_x - 1, mid_y - 1, mid_x + 1, mid_y + 1], fill=(210, 215, 220))

    # Main Integrated Circuit (IC Chip) in center
    chip_w = random.randint(22, 32)
    chip_h = random.randint(20, 28)
    cx0 = (w - chip_w) // 2 + random.randint(-4, 4)
    cy0 = (h - chip_h) // 2 + random.randint(-4, 4)

    # IC chip body (matte black)
    draw.rectangle([cx0, cy0, cx0 + chip_w, cy0 + chip_h], fill=(20, 20, 22), outline=(10, 10, 10))
    # Pin 1 indicator dot
    draw.ellipse([cx0 + 2, cy0 + 2, cx0 + 4, cy0 + 4], fill=(120, 120, 120))

    # Metallic IC pins on left and right sides
    pin_col = (200, 205, 215)
    for py in range(cy0 + 3, cy0 + chip_h - 2, 4):
        # Left pins
        draw.line([(cx0 - 4, py), (cx0, py)], fill=pin_col, width=1)
        # Right pins
        draw.line([(cx0 + chip_w, py), (cx0 + chip_w + 4, py)], fill=pin_col, width=1)

    # Surface mount capacitors / resistors (tiny yellow or gray blocks)
    for _ in range(6):
        cap_x = random.randint(board_x0 + 4, board_x1 - 10)
        cap_y = random.randint(board_y0 + 4, board_y1 - 8)
        if not (cx0 - 6 < cap_x < cx0 + chip_w + 6 and cy0 - 6 < cap_y < cy0 + chip_h + 6):
            cap_col = random.choice([(185, 140, 70), (120, 125, 130), (50, 50, 55)])
            draw.rectangle([cap_x, cap_y, cap_x + 5, cap_y + 3], fill=cap_col)

    angle = random.uniform(-8, 8)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(225, 228, 230))
    return img

GENERATORS = {
    "mobile_phone": draw_mobile_phone,
    "charger_cable": draw_charger_cable,
    "earphones": draw_earphones,
    "circuit_board": draw_circuit_board
}

def generate_all():
    """Generates the full dataset into train, val, and test splits."""
    print("=" * 60)
    print("ReBoot AI - Synthetic E-Waste Dataset Generator")
    print("=" * 60)
    print(f"Target image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels")
    print(f"Classes: {', '.join(CLASSES)}")
    print(f"Splits: Train={COUNTS['train']}, Val={COUNTS['val']}, Test={COUNTS['test']} per class")
    print("Notice: Generating synthetic images for educational demonstration.")
    print("-" * 60)

    total_created = 0

    for split, count in COUNTS.items():
        for category in CLASSES:
            cat_dir = os.path.join(DATA_DIR, split, category)
            os.makedirs(cat_dir, exist_ok=True)

            gen_func = GENERATORS[category]
            for i in range(1, count + 1):
                img = gen_func(IMAGE_SIZE)

                # Minor post-processing (slight contrast/brightness variation)
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(random.uniform(0.92, 1.08))

                filename = f"syn_{category}_{split}_{i:03d}.png"
                filepath = os.path.join(cat_dir, filename)
                img.save(filepath, "PNG")
                total_created += 1

            print(f"  [{split.upper():5s}] Created {count:3d} images for '{category}' -> {cat_dir}")

    print("-" * 60)
    print(f"Dataset generation complete! Total images created: {total_created}")
    print(f"Data root: {DATA_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    generate_all()
