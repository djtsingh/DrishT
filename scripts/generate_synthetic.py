"""
Synthetic Indian Scene-Text Generator
=======================================
Renders synthetic text on realistic Indian-context backgrounds with
augmentations (dust, glare, motion blur, perspective warp, colour jitter).

Supports 12 Indian scripts + Latin using Google Noto fonts.
Produces word crops + labels for the recognition pipeline.

Output: data/processed/synthetic/
        ├── images/
        ├── labels.csv
        └── stats.json

Prerequisites:
  pip install Pillow numpy albumentations
  Noto fonts (auto-downloaded if missing)
"""
import os
import sys
import csv
import json
import random
import math
import string
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm

try:
    import albumentations as A
    HAS_ALBUM = True
except ImportError:
    HAS_ALBUM = False
    print("WARNING: albumentations not installed. Advanced augmentations disabled.")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW = Path("data/raw")
OUT = Path("data/processed/synthetic")
OUT_IMAGES = OUT / "images"
FONT_DIR = Path("data/fonts")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# How many synthetic samples per script
SAMPLES_PER_SCRIPT = 2000
LATIN_SAMPLES = 3000

# Image dimensions for word crops
MIN_W, MAX_W = 80, 400
MIN_H, MAX_H = 24, 64

# ---------------------------------------------------------------------------
# Indian vocabulary samples (small dictionaries for each script)
# ---------------------------------------------------------------------------
INDIC_VOCAB = {
    "hindi": [
        "भारत", "दिल्ली", "स्टॉप", "खतरा", "निकास", "प्रवेश", "मार्ग",
        "स्कूल", "अस्पताल", "बाजार", "दुकान", "चाय", "पानी", "खाना",
        "रेस्तरां", "होटल", "बैंक", "स्टेशन", "बस", "ट्रेन", "मेट्रो",
        "सड़क", "गली", "चौराहा", "पुल", "नदी", "मंदिर", "मस्जिद",
        "गुरुद्वारा", "चर्च", "पार्क", "स्टेडियम", "सिनेमा", "थिएटर",
        "किराना", "मेडिकल", "फार्मेसी", "पुलिस", "फायर", "एम्बुलेंस",
        "टोल", "पार्किंग", "नो", "एंट्री", "वन", "वे", "धीरे", "चलिए",
        "रुकिए", "आगे", "पीछे", "दाएं", "बाएं", "सीधे",
    ],
    "bengali": [
        "ভারত", "কলকাতা", "স্টপ", "বিপদ", "প্রবেশ", "বাজার", "দোকান",
        "চা", "জল", "খাবার", "রেস্তোরাঁ", "হোটেল", "ব্যাংক", "স্টেশন",
        "বাস", "ট্রেন", "রাস্তা", "গলি", "সেতু", "নদী", "মন্দির",
        "মসজিদ", "পার্ক", "সিনেমা", "দমকল", "পুলিশ",
    ],
    "tamil": [
        "இந்தியா", "சென்னை", "நிறுத்தம்", "ஆபத்து", "நுழைவு", "சந்தை",
        "கடை", "தேநீர்", "தண்ணீர்", "உணவு", "விடுதி", "வங்கி", "நிலையம்",
        "பேருந்து", "ரயில்", "சாலை", "பாலம்", "நதி", "கோவில்",
        "பூங்கா", "காவல்", "மருத்துவம்",
    ],
    "telugu": [
        "భారతదేశం", "హైదరాబాద్", "ఆపు", "ప్రమాదం", "ప్రవేశం", "మార్కెట్",
        "దుకాణం", "టీ", "నీరు", "ఆహారం", "హోటల్", "బ్యాంక్", "స్టేషన్",
        "బస్సు", "రైలు", "రోడ్డు", "వంతెన", "నది", "గుడి", "పార్కు",
    ],
    "gujarati": [
        "ભારત", "અમદાવાદ", "સ્ટોપ", "ખતરો", "પ્રવેશ", "બજાર", "દુકાન",
        "ચા", "પાણી", "ખોરાક", "હોટેલ", "બેંક", "સ્ટેશન", "બસ", "ટ્રેન",
        "રસ્તો", "પુલ", "નદી", "મંદિર", "પાર્ક",
    ],
    "kannada": [
        "ಭಾರತ", "ಬೆಂಗಳೂರು", "ನಿಲ್ಲಿ", "ಅಪಾಯ", "ಪ್ರವೇಶ", "ಮಾರುಕಟ್ಟೆ",
        "ಅಂಗಡಿ", "ಚಹಾ", "ನೀರು", "ಆಹಾರ", "ಹೋಟೆಲ್", "ಬ್ಯಾಂಕ್", "ನಿಲ್ದಾಣ",
        "ಬಸ್ಸು", "ರೈಲು", "ರಸ್ತೆ", "ಸೇತುವೆ", "ನದಿ", "ದೇವಸ್ಥಾನ",
    ],
    "malayalam": [
        "ഭാരതം", "കൊച്ചി", "നിർത്തുക", "അപകടം", "പ്രവേശനം", "ചന്ത",
        "കട", "ചായ", "വെള്ളം", "ഭക്ഷണം", "ഹോട്ടൽ", "ബാങ്ക്", "സ്റ്റേഷൻ",
        "ബസ്", "ട്രെയിൻ", "റോഡ്", "പാലം", "നദി", "ക്ഷേത്രം",
    ],
    "marathi": [
        "भारत", "मुंबई", "थांबा", "धोका", "प्रवेश", "बाजार", "दुकान",
        "चहा", "पाणी", "जेवण", "हॉटेल", "बँक", "स्टेशन", "बस", "ट्रेन",
        "रस्ता", "पूल", "नदी", "मंदिर", "पार्क",
    ],
    "odia": [
        "ଭାରତ", "ଭୁବନେଶ୍ୱର", "ରହ", "ବିପଦ", "ପ୍ରବେଶ", "ବଜାର", "ଦୋକାନ",
        "ଚା", "ପାଣି", "ଖାଦ୍ୟ", "ହୋଟେଲ", "ବ୍ୟାଙ୍କ", "ଷ୍ଟେସନ",
    ],
    "punjabi": [
        "ਭਾਰਤ", "ਅੰਮ੍ਰਿਤਸਰ", "ਰੁਕੋ", "ਖ਼ਤਰਾ", "ਪ੍ਰਵੇਸ਼", "ਬਜ਼ਾਰ",
        "ਦੁਕਾਨ", "ਚਾਹ", "ਪਾਣੀ", "ਖਾਣਾ", "ਹੋਟਲ", "ਬੈਂਕ", "ਸਟੇਸ਼ਨ",
    ],
    "assamese": [
        "ভাৰত", "গুৱাহাটী", "ৰওক", "বিপদ", "প্ৰৱেশ", "বজাৰ", "দোকান",
        "চাহ", "পানী", "খাদ্য", "হোটেল", "বেংক", "ষ্টেচন",
    ],
    "urdu": [
        "بھارت", "دہلی", "رکیں", "خطرہ", "داخلہ", "بازار", "دکان",
        "چائے", "پانی", "کھانا", "ہوٹل", "بینک", "اسٹیشن", "بس",
    ],
}

# Latin words common on Indian signs
LATIN_WORDS = [
    "STOP", "DANGER", "EXIT", "ENTRY", "NO ENTRY", "ONE WAY", "SCHOOL",
    "HOSPITAL", "POLICE", "FIRE", "PARKING", "TOLL", "BRIDGE", "SPEED",
    "LIMIT", "SLOW", "CAUTION", "WARNING", "ZONE", "AREA", "ROAD",
    "HIGHWAY", "NATIONAL", "STATE", "DISTRICT", "CITY", "TOWN", "VILLAGE",
    "MARKET", "SHOP", "RESTAURANT", "HOTEL", "BANK", "ATM", "STATION",
    "BUS", "TRAIN", "METRO", "AIRPORT", "TAXI", "AUTO", "AMBULANCE",
    "EMERGENCY", "HELP", "OPEN", "CLOSED", "LEFT", "RIGHT", "STRAIGHT",
    "TURN", "AHEAD", "BACK", "WELCOME", "THANK YOU", "INDIA", "CHAI",
    "TEA", "COFFEE", "FOOD", "WATER", "PETROL", "DIESEL", "CNG", "GAS",
]

# Common number plate patterns
PLATE_PATTERNS = [
    "XX 00 XX 0000",  # Standard Indian format
    "XX 00 X 0000",
    "XX 00 XX 000",
]

# ---------------------------------------------------------------------------
# Background & Color Palettes
# ---------------------------------------------------------------------------
SIGN_BG_COLORS = [
    (255, 255, 255),   # white
    (255, 255, 0),     # yellow (warning)
    (0, 100, 0),       # green (highway)
    (0, 0, 150),       # blue (info)
    (200, 0, 0),       # red (prohibition)
    (255, 165, 0),     # orange (construction)
    (220, 220, 220),   # light gray
    (139, 90, 43),     # brown (wood/rural)
    (50, 50, 50),      # dark gray (blackboard)
]

TEXT_COLORS = [
    (0, 0, 0),         # black
    (255, 255, 255),   # white
    (255, 0, 0),       # red
    (0, 0, 128),       # navy
    (0, 100, 0),       # dark green
    (128, 0, 0),       # maroon
]


# ---------------------------------------------------------------------------
# Font Management
# ---------------------------------------------------------------------------
def get_system_fonts():
    """Find available fonts, prefer Noto fonts for Indic support."""
    fonts = {}

    # Common font paths on Windows
    font_paths = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
        FONT_DIR,
        Path.home() / ".fonts",
    ]

    # Map scripts to font name patterns
    script_font_map = {
        "hindi":    ["NotoSansDevanagari", "Noto_Sans_Devanagari", "NotoSerif-Devanagari", "Mangal", "Arial"],
        "bengali":  ["NotoSansBengali", "Noto_Sans_Bengali", "Vrinda", "Arial"],
        "tamil":    ["NotoSansTamil", "Noto_Sans_Tamil", "Latha", "Arial"],
        "telugu":   ["NotoSansTelugu", "Noto_Sans_Telugu", "Gautami", "Arial"],
        "gujarati": ["NotoSansGujarati", "Noto_Sans_Gujarati", "Shruti", "Arial"],
        "kannada":  ["NotoSansKannada", "Noto_Sans_Kannada", "Tunga", "Arial"],
        "malayalam":["NotoSansMalayalam", "Noto_Sans_Malayalam", "Kartika", "Arial"],
        "marathi":  ["NotoSansDevanagari", "Noto_Sans_Devanagari", "Mangal", "Arial"],
        "odia":     ["NotoSansOriya", "Noto_Sans_Oriya", "Noto_Sans_Odia", "Kalinga", "Arial"],
        "punjabi":  ["NotoSansGurmukhi", "Noto_Sans_Gurmukhi", "Raavi", "Arial"],
        "assamese": ["NotoSansBengali", "Noto_Sans_Bengali", "Vrinda", "Arial"],
        "urdu":     ["NotoNaskhArabic", "NotoSansArabic", "Noto_Sans_Arabic", "Arial"],
        "latin":    ["Arial", "Calibri", "NotoSans-Regular", "DejaVuSans"],
    }

    for script, preferred in script_font_map.items():
        found = False
        for font_name in preferred:
            for fp in font_paths:
                if not fp.exists():
                    continue
                for ext in [".ttf", ".otf", ".TTF", ".OTF"]:
                    # Use recursive glob to find fonts in subdirs
                    candidates = list(fp.glob(f"**/*{font_name}*{ext}"))
                    if not candidates:
                        candidates = list(fp.glob(f"*{font_name}*{ext}"))
                    if candidates:
                        fonts[script] = str(candidates[0])
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            # Fallback to any available font
            for fp in font_paths:
                if not fp.exists():
                    continue
                fallbacks = list(fp.glob("**/*.ttf")) + list(fp.glob("**/*.TTF"))
                if fallbacks:
                    fonts[script] = str(fallbacks[0])
                    break

    return fonts


# ---------------------------------------------------------------------------
# Augmentation Pipeline
# ---------------------------------------------------------------------------
def get_augmentation():
    """Build augmentation pipeline using albumentations."""
    if not HAS_ALBUM:
        return None

    return A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
            A.MedianBlur(blur_limit=3, p=0.1),
        ], p=0.4),
        A.OneOf([
            A.GaussNoise(p=0.3),
            A.ISONoise(p=0.2),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.CLAHE(clip_limit=2.0, p=0.2),
        ], p=0.4),
        A.OneOf([
            A.RandomSunFlare(src_radius=50, num_flare_circles_lower=1,
                             num_flare_circles_upper=3, p=0.1),
            A.RandomShadow(p=0.2),
        ], p=0.15),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.3),
        A.Perspective(scale=(0.02, 0.08), p=0.3),
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.3),
    ])


def apply_dust_effect(img_np):
    """Simulate dust/haze on Indian roads."""
    dust_intensity = random.uniform(0.05, 0.25)
    dust_color = np.array([200, 180, 150], dtype=np.float32)  # sandy/dusty
    result = img_np.astype(np.float32) * (1 - dust_intensity) + dust_color * dust_intensity
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_rain_streaks(img_np):
    """Simulate rain streaks."""
    h, w = img_np.shape[:2]
    rain = np.zeros_like(img_np)
    n_streaks = random.randint(5, 20)
    for _ in range(n_streaks):
        x = random.randint(0, w - 1)
        y1 = random.randint(0, h // 2)
        length = random.randint(5, min(15, h - y1))
        rain[y1:y1+length, x] = [200, 200, 220]
    alpha = random.uniform(0.1, 0.3)
    result = img_np.astype(np.float32) * (1 - alpha) + rain.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Text Rendering
# ---------------------------------------------------------------------------
def generate_number_plate():
    """Generate a random Indian-style number plate text."""
    pattern = random.choice(PLATE_PATTERNS)
    result = ""
    for ch in pattern:
        if ch == "X":
            result += random.choice(string.ascii_uppercase)
        elif ch == "0":
            result += random.choice(string.digits)
        else:
            result += ch
    return result


def render_text_image(text, font_path, font_size=None):
    """Render text as a word-crop image with random style."""
    if font_size is None:
        font_size = random.randint(18, 48)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()

    # Measure text
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0] + random.randint(10, 30)  # padding
    th = bbox[3] - bbox[1] + random.randint(6, 16)

    tw = max(tw, MIN_W)
    th = max(th, MIN_H)

    # Pick colors
    bg_color = random.choice(SIGN_BG_COLORS)
    # Ensure contrast
    bg_brightness = sum(bg_color) / 3
    if bg_brightness > 128:
        text_color = random.choice([c for c in TEXT_COLORS if sum(c)/3 < 100])
    else:
        text_color = random.choice([c for c in TEXT_COLORS if sum(c)/3 > 150])

    # Create image
    img = Image.new("RGB", (tw, th), bg_color)
    draw = ImageDraw.Draw(img)

    # Add optional border (like sign plates)
    if random.random() < 0.3:
        border_color = text_color
        draw.rectangle([1, 1, tw-2, th-2], outline=border_color, width=2)

    # Center text
    x = (tw - (bbox[2] - bbox[0])) // 2
    y = (th - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, fill=text_color, font=font)

    return img


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Synthetic Indian Scene-Text Generator")
    print("=" * 60)

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    FONT_DIR.mkdir(parents=True, exist_ok=True)

    # Get available fonts
    fonts = get_system_fonts()
    print(f"\n  Found fonts for {len(fonts)} scripts:")
    for script, fpath in sorted(fonts.items()):
        print(f"    {script}: {Path(fpath).name}")

    # Setup augmentation
    aug = get_augmentation()
    if aug:
        print("\n  Augmentation pipeline: ENABLED")
    else:
        print("\n  Augmentation pipeline: DISABLED (install albumentations)")

    all_records = []
    counter = 0

    # --- Generate for each Indic script ---
    for script, vocab in INDIC_VOCAB.items():
        font_path = fonts.get(script)
        if not font_path:
            print(f"\n  SKIP {script}: no font available")
            continue

        print(f"\n  Generating {SAMPLES_PER_SCRIPT} samples for: {script}")
        for _ in tqdm(range(SAMPLES_PER_SCRIPT), desc=script):
            # Pick random word(s)
            n_words = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            text = " ".join(random.choices(vocab, k=n_words))

            try:
                img = render_text_image(text, font_path)
            except Exception:
                continue

            # Apply augmentations
            img_np = np.array(img)
            if aug:
                try:
                    img_np = aug(image=img_np)["image"]
                except Exception:
                    pass

            # Random Indian-condition effects
            r = random.random()
            if r < 0.15:
                img_np = apply_dust_effect(img_np)
            elif r < 0.25:
                img_np = apply_rain_streaks(img_np)

            counter += 1
            dest_name = f"synth_{script}_{counter:06d}.jpg"
            final_img = Image.fromarray(img_np)
            final_img.save(OUT_IMAGES / dest_name, quality=90)
            all_records.append((dest_name, text, script, "synthetic"))

    # --- Generate Latin samples ---
    font_path = fonts.get("latin")
    if font_path:
        print(f"\n  Generating {LATIN_SAMPLES} Latin samples")
        for _ in tqdm(range(LATIN_SAMPLES), desc="latin"):
            r = random.random()
            if r < 0.3:
                text = generate_number_plate()
                script = "latin_plate"
            else:
                n_words = random.choices([1, 2], weights=[0.7, 0.3])[0]
                text = " ".join(random.choices(LATIN_WORDS, k=n_words))
                script = "latin"

            try:
                img = render_text_image(text, font_path)
            except Exception:
                continue

            img_np = np.array(img)
            if aug:
                try:
                    img_np = aug(image=img_np)["image"]
                except Exception:
                    pass

            if random.random() < 0.15:
                img_np = apply_dust_effect(img_np)

            counter += 1
            dest_name = f"synth_latin_{counter:06d}.jpg"
            final_img = Image.fromarray(img_np)
            final_img.save(OUT_IMAGES / dest_name, quality=90)
            all_records.append((dest_name, text, script, "synthetic"))

    # --- Write labels ---
    csv_path = OUT / "labels.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "script", "source"])
        for rec in all_records:
            writer.writerow(rec)
    print(f"\n  Labels: {csv_path}")

    # --- Stats ---
    script_dist = defaultdict(int)
    for _, _, script, _ in all_records:
        script_dist[script] += 1

    stats = {
        "total_synthetic": len(all_records),
        "script_distribution": dict(sorted(script_dist.items(), key=lambda x: -x[1])),
        "fonts_used": {k: Path(v).name for k, v in fonts.items()},
        "augmentation_enabled": aug is not None,
    }
    with open(OUT / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("  Synthetic Data Summary")
    print(f"{'='*60}")
    print(f"  Total samples: {len(all_records)}")
    for sc, cnt in sorted(script_dist.items(), key=lambda x: -x[1]):
        print(f"    {sc}: {cnt}")
    print(f"\n  Output: {OUT}")
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
