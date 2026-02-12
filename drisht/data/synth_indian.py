"""
Indian Text Image Synthesizer
=============================

Generate photorealistic synthetic text images for training.
Supports all major Indian scripts with proper rendering.

Features:
- 10 Indian scripts + Latin
- Random fonts, colors, effects
- Perspective/affine transforms
- Realistic degradations (blur, noise, JPEG artifacts)
- Background textures
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import cv2


@dataclass
class TextSample:
    """Generated text sample."""
    image: np.ndarray
    text: str
    script: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    polygon: List[Tuple[int, int]]  # For curved/transformed text


class FontManager:
    """Manage fonts for different scripts."""
    
    SCRIPT_FONTS = {
        'devanagari': [
            'Noto Sans Devanagari',
            'Mangal',
            'Aparajita',
            'Kokila',
            'Utsaah',
        ],
        'bengali': [
            'Noto Sans Bengali',
            'Vrinda',
            'Shonar Bangla',
            'Akaash',
        ],
        'tamil': [
            'Noto Sans Tamil',
            'Latha',
            'Vijaya',
        ],
        'telugu': [
            'Noto Sans Telugu',
            'Gautami',
            'Vani',
        ],
        'kannada': [
            'Noto Sans Kannada',
            'Tunga',
        ],
        'malayalam': [
            'Noto Sans Malayalam',
            'Kartika',
            'Rachana',
        ],
        'gujarati': [
            'Noto Sans Gujarati',
            'Shruti',
        ],
        'gurmukhi': [
            'Noto Sans Gurmukhi',
            'Raavi',
        ],
        'odia': [
            'Noto Sans Oriya',
            'Kalinga',
        ],
        'latin': [
            'Arial',
            'Times New Roman',
            'Verdana',
            'Georgia',
            'Tahoma',
            'Impact',
            'Comic Sans MS',
        ],
    }
    
    def __init__(self, font_dirs: Optional[List[str]] = None):
        self.font_dirs = font_dirs or [
            'C:/Windows/Fonts',
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            str(Path.home() / '.fonts'),
        ]
        
        self.fonts = {}
        self._scan_fonts()
    
    def _scan_fonts(self):
        """Scan directories for available fonts."""
        for script, font_names in self.SCRIPT_FONTS.items():
            self.fonts[script] = []
            
            for font_name in font_names:
                font_path = self._find_font(font_name)
                if font_path:
                    self.fonts[script].append(font_path)
            
            # Fallback to any available TTF
            if not self.fonts[script]:
                for font_dir in self.font_dirs:
                    if os.path.exists(font_dir):
                        for f in os.listdir(font_dir):
                            if f.lower().endswith('.ttf'):
                                self.fonts[script].append(os.path.join(font_dir, f))
                                break
    
    def _find_font(self, font_name: str) -> Optional[str]:
        """Find font file by name."""
        search_names = [
            font_name + '.ttf',
            font_name + '.TTF',
            font_name.replace(' ', '') + '.ttf',
            font_name.replace(' ', '-') + '.ttf',
            font_name.lower().replace(' ', '') + '.ttf',
        ]
        
        for font_dir in self.font_dirs:
            if not os.path.exists(font_dir):
                continue
            
            for search_name in search_names:
                font_path = os.path.join(font_dir, search_name)
                if os.path.exists(font_path):
                    return font_path
        
        return None
    
    def get_font(self, script: str, size: int) -> ImageFont.FreeTypeFont:
        """Get random font for script."""
        if script not in self.fonts or not self.fonts[script]:
            # Fallback to default
            return ImageFont.load_default()
        
        font_path = random.choice(self.fonts[script])
        try:
            return ImageFont.truetype(font_path, size)
        except:
            return ImageFont.load_default()


class TextCorpus:
    """Manage text corpora for different scripts."""
    
    # Sample words for each script (expand with real corpora)
    SAMPLE_WORDS = {
        'devanagari': [
            'नमस्ते', 'भारत', 'दिल्ली', 'मुंबई', 'स्वागत', 'धन्यवाद',
            'खुला', 'बंद', 'दुकान', 'होटल', 'रेस्टोरेंट', 'पानी',
            'चाय', 'कॉफी', 'खाना', 'दवाई', 'डॉक्टर', 'अस्पताल',
        ],
        'bengali': [
            'নমস্কার', 'বাংলা', 'কলকাতা', 'ধন্যবাদ', 'স্বাগতম',
            'দোকান', 'হোটেল', 'জল', 'চা', 'খাবার',
        ],
        'tamil': [
            'வணக்கம்', 'தமிழ்', 'சென்னை', 'நன்றி', 'வரவேற்பு',
            'கடை', 'உணவகம்', 'தண்ணீர்', 'டீ', 'காபி',
        ],
        'telugu': [
            'నమస్కారం', 'తెలుగు', 'హైదరాబాద్', 'ధన్యవాదాలు',
            'స్వాగతం', 'దుకాణం', 'హోటల్', 'నీరు', 'టీ',
        ],
        'kannada': [
            'ನಮಸ್ಕಾರ', 'ಕನ್ನಡ', 'ಬೆಂಗಳೂರು', 'ಧನ್ಯವಾದ',
            'ಸ್ವಾಗತ', 'ಅಂಗಡಿ', 'ಹೋಟೆಲ್', 'ನೀರು',
        ],
        'malayalam': [
            'നമസ്കാരം', 'മലയാളം', 'കൊച്ചി', 'നന്ദി',
            'സ്വാഗതം', 'കട', 'ഹോട്ടൽ', 'വെള്ളം',
        ],
        'gujarati': [
            'નમસ્તે', 'ગુજરાતી', 'અમદાવાદ', 'આભાર',
            'સ્વાગત', 'દુકાન', 'હોટેલ', 'પાણી',
        ],
        'gurmukhi': [
            'ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'ਪੰਜਾਬੀ', 'ਚੰਡੀਗੜ੍ਹ', 'ਧੰਨਵਾਦ',
            'ਸਵਾਗਤ', 'ਦੁਕਾਨ', 'ਹੋਟਲ', 'ਪਾਣੀ',
        ],
        'odia': [
            'ନମସ୍କାର', 'ଓଡ଼ିଆ', 'ଭୁବନେଶ୍ୱର', 'ଧନ୍ୟବାଦ',
            'ସ୍ୱାଗତ', 'ଦୋକାନ', 'ହୋଟେଲ', 'ପାଣି',
        ],
        'latin': [
            'SHOP', 'HOTEL', 'RESTAURANT', 'OPEN', 'CLOSED',
            'WELCOME', 'EXIT', 'ENTRY', 'PARKING', 'TAXI',
            'HOSPITAL', 'PHARMACY', 'BANK', 'ATM', 'POLICE',
        ],
    }
    
    def __init__(self, corpus_dirs: Optional[Dict[str, str]] = None):
        self.words = dict(self.SAMPLE_WORDS)
        
        # Load additional corpora if provided
        if corpus_dirs:
            for script, path in corpus_dirs.items():
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        words = [line.strip() for line in f if line.strip()]
                        self.words[script] = words
    
    def sample_text(self, script: str, min_words: int = 1, max_words: int = 4) -> str:
        """Sample random text for script."""
        if script not in self.words:
            script = 'latin'
        
        num_words = random.randint(min_words, max_words)
        words = random.choices(self.words[script], k=num_words)
        return ' '.join(words)


class BackgroundGenerator:
    """Generate realistic backgrounds."""
    
    def __init__(self, texture_dir: Optional[str] = None):
        self.textures = []
        
        if texture_dir and os.path.exists(texture_dir):
            for f in os.listdir(texture_dir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.textures.append(os.path.join(texture_dir, f))
    
    def generate(self, width: int, height: int) -> Image.Image:
        """Generate background image."""
        
        if self.textures and random.random() > 0.3:
            # Use texture
            texture_path = random.choice(self.textures)
            bg = Image.open(texture_path).convert('RGB')
            bg = bg.resize((width, height), Image.BILINEAR)
        else:
            # Generate solid/gradient
            if random.random() > 0.5:
                # Solid color
                color = self._random_color(bright=random.random() > 0.5)
                bg = Image.new('RGB', (width, height), color)
            else:
                # Gradient
                bg = self._create_gradient(width, height)
        
        # Add noise
        if random.random() > 0.5:
            bg = self._add_noise(bg)
        
        return bg
    
    def _random_color(self, bright: bool = True) -> Tuple[int, int, int]:
        """Generate random color."""
        if bright:
            return (
                random.randint(180, 255),
                random.randint(180, 255),
                random.randint(180, 255),
            )
        else:
            return (
                random.randint(0, 100),
                random.randint(0, 100),
                random.randint(0, 100),
            )
    
    def _create_gradient(self, width: int, height: int) -> Image.Image:
        """Create gradient background."""
        c1 = self._random_color(bright=True)
        c2 = self._random_color(bright=True)
        
        img = Image.new('RGB', (width, height))
        pixels = img.load()
        
        for y in range(height):
            ratio = y / height
            r = int(c1[0] * (1 - ratio) + c2[0] * ratio)
            g = int(c1[1] * (1 - ratio) + c2[1] * ratio)
            b = int(c1[2] * (1 - ratio) + c2[2] * ratio)
            for x in range(width):
                pixels[x, y] = (r, g, b)
        
        return img
    
    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Add subtle noise."""
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, 5, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


class TextRenderer:
    """Render text with effects."""
    
    def __init__(self, font_manager: FontManager):
        self.font_manager = font_manager
    
    def render(
        self,
        text: str,
        script: str,
        font_size: int = 40,
        color: Optional[Tuple[int, int, int]] = None,
        effects: Optional[Dict] = None,
    ) -> Tuple[Image.Image, List[Tuple[int, int]]]:
        """
        Render text to image with effects.
        
        Returns:
            (image, polygon) where polygon is corners of text region
        """
        effects = effects or {}
        
        # Get font
        font = self.font_manager.get_font(script, font_size)
        
        # Calculate text size
        dummy_img = Image.new('RGBA', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        bbox = dummy_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Add padding
        padding = int(font_size * 0.3)
        width = text_width + padding * 2
        height = text_height + padding * 2
        
        # Create image
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Choose color
        if color is None:
            color = self._random_text_color()
        
        # Draw shadow if enabled
        if effects.get('shadow', False):
            shadow_offset = random.randint(2, 5)
            shadow_color = (0, 0, 0, 128)
            draw.text(
                (padding + shadow_offset, padding + shadow_offset),
                text,
                font=font,
                fill=shadow_color,
            )
        
        # Draw outline if enabled
        if effects.get('outline', False):
            outline_width = random.randint(1, 3)
            outline_color = self._contrasting_color(color)
            for dx in range(-outline_width, outline_width + 1):
                for dy in range(-outline_width, outline_width + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text(
                        (padding + dx, padding + dy),
                        text,
                        font=font,
                        fill=outline_color,
                    )
        
        # Draw main text
        draw.text((padding, padding), text, font=font, fill=color)
        
        # Get polygon (corners)
        polygon = [
            (0, 0),
            (width, 0),
            (width, height),
            (0, height),
        ]
        
        return img, polygon
    
    def _random_text_color(self) -> Tuple[int, int, int]:
        """Generate random text color (usually dark)."""
        if random.random() > 0.3:
            # Dark text
            v = random.randint(0, 80)
            return (v, v, v)
        else:
            # Colored text
            return (
                random.randint(0, 200),
                random.randint(0, 200),
                random.randint(0, 200),
            )
    
    def _contrasting_color(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Get contrasting color."""
        brightness = (color[0] + color[1] + color[2]) / 3
        if brightness > 128:
            return (0, 0, 0)
        else:
            return (255, 255, 255)


class ImageTransformer:
    """Apply geometric transforms to images."""
    
    def perspective_transform(
        self,
        img: Image.Image,
        polygon: List[Tuple[int, int]],
        max_angle: float = 20,
    ) -> Tuple[Image.Image, List[Tuple[int, int]]]:
        """Apply random perspective transform."""
        
        width, height = img.size
        
        # Random perspective distortion
        def random_offset():
            return random.uniform(-max_angle, max_angle)
        
        # Source points (corners)
        src = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ])
        
        # Destination points (with random offsets)
        margin = min(width, height) * 0.1
        dst = np.float32([
            [random_offset() * margin / max_angle, random_offset() * margin / max_angle],
            [width - random_offset() * margin / max_angle, random_offset() * margin / max_angle],
            [width - random_offset() * margin / max_angle, height - random_offset() * margin / max_angle],
            [random_offset() * margin / max_angle, height - random_offset() * margin / max_angle],
        ])
        
        # Compute transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        
        # Apply transform
        img_np = np.array(img)
        transformed = cv2.warpPerspective(img_np, M, (width, height))
        
        # Transform polygon
        new_polygon = []
        for px, py in polygon:
            pt = np.array([[[px, py]]], dtype=np.float32)
            new_pt = cv2.perspectiveTransform(pt, M)
            new_polygon.append((int(new_pt[0, 0, 0]), int(new_pt[0, 0, 1])))
        
        return Image.fromarray(transformed), new_polygon
    
    def rotate(
        self,
        img: Image.Image,
        polygon: List[Tuple[int, int]],
        max_angle: float = 15,
    ) -> Tuple[Image.Image, List[Tuple[int, int]]]:
        """Apply random rotation."""
        
        angle = random.uniform(-max_angle, max_angle)
        
        # Rotate image
        rotated = img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0))
        
        # Rotate polygon
        width, height = img.size
        cx, cy = width / 2, height / 2
        
        rad = np.radians(-angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        new_polygon = []
        for px, py in polygon:
            # Translate to origin
            px -= cx
            py -= cy
            
            # Rotate
            new_x = px * cos_a - py * sin_a
            new_y = px * sin_a + py * cos_a
            
            # Translate back (to new center)
            new_w, new_h = rotated.size
            new_x += new_w / 2
            new_y += new_h / 2
            
            new_polygon.append((int(new_x), int(new_y)))
        
        return rotated, new_polygon


class ImageDegrader:
    """Apply realistic degradations."""
    
    def degrade(
        self,
        img: Image.Image,
        blur: float = 0.0,
        noise: float = 0.0,
        jpeg_quality: int = 100,
        brightness: float = 1.0,
        contrast: float = 1.0,
    ) -> Image.Image:
        """Apply degradations."""
        
        # Brightness/contrast
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)
        
        # Blur
        if blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=blur))
        
        # Noise
        if noise > 0:
            arr = np.array(img, dtype=np.float32)
            gaussian = np.random.normal(0, noise * 255, arr.shape)
            arr = np.clip(arr + gaussian, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
        
        # JPEG compression
        if jpeg_quality < 100:
            from io import BytesIO
            buffer = BytesIO()
            img.convert('RGB').save(buffer, 'JPEG', quality=jpeg_quality)
            buffer.seek(0)
            img = Image.open(buffer)
        
        return img


class IndianTextSynthesizer:
    """
    Main synthesizer for generating Indian text images.
    """
    
    SCRIPTS = [
        'devanagari', 'bengali', 'tamil', 'telugu', 'kannada',
        'malayalam', 'gujarati', 'gurmukhi', 'odia', 'latin'
    ]
    
    def __init__(
        self,
        texture_dir: Optional[str] = None,
        corpus_dirs: Optional[Dict[str, str]] = None,
    ):
        self.font_manager = FontManager()
        self.corpus = TextCorpus(corpus_dirs)
        self.background_gen = BackgroundGenerator(texture_dir)
        self.renderer = TextRenderer(self.font_manager)
        self.transformer = ImageTransformer()
        self.degrader = ImageDegrader()
    
    def generate(
        self,
        script: Optional[str] = None,
        difficulty: str = 'medium',
        width: int = 640,
        height: int = 480,
    ) -> TextSample:
        """
        Generate a synthetic text image.
        
        Args:
            script: Script to use (random if None)
            difficulty: 'easy', 'medium', or 'hard'
            width, height: Output dimensions
            
        Returns:
            TextSample with image, text, and annotations
        """
        # Choose script
        if script is None:
            script = random.choice(self.SCRIPTS)
        
        # Sample text
        text = self.corpus.sample_text(script)
        
        # Difficulty settings
        settings = self._get_difficulty_settings(difficulty)
        
        # Render text
        font_size = random.randint(settings['font_min'], settings['font_max'])
        effects = {
            'shadow': random.random() > 0.5,
            'outline': random.random() > 0.7,
        }
        text_img, polygon = self.renderer.render(text, script, font_size, effects=effects)
        
        # Apply transforms
        if random.random() < settings['perspective_prob']:
            text_img, polygon = self.transformer.perspective_transform(
                text_img, polygon, max_angle=settings['perspective_max']
            )
        
        if random.random() < settings['rotate_prob']:
            text_img, polygon = self.transformer.rotate(
                text_img, polygon, max_angle=settings['rotate_max']
            )
        
        # Generate background
        bg = self.background_gen.generate(width, height)
        
        # Place text on background
        text_w, text_h = text_img.size
        max_x = max(0, width - text_w)
        max_y = max(0, height - text_h)
        
        x = random.randint(0, max_x) if max_x > 0 else 0
        y = random.randint(0, max_y) if max_y > 0 else 0
        
        # Composite
        if text_img.mode == 'RGBA':
            bg.paste(text_img, (x, y), text_img)
        else:
            bg.paste(text_img, (x, y))
        
        # Update polygon with position
        polygon = [(px + x, py + y) for px, py in polygon]
        
        # Apply degradations
        bg = self.degrader.degrade(
            bg,
            blur=random.uniform(0, settings['blur_max']),
            noise=random.uniform(0, settings['noise_max']),
            jpeg_quality=random.randint(settings['jpeg_min'], 100),
            brightness=random.uniform(0.8, 1.2),
            contrast=random.uniform(0.8, 1.2),
        )
        
        # Get bounding box
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        bbox = (min(xs), min(ys), max(xs), max(ys))
        
        return TextSample(
            image=np.array(bg),
            text=text,
            script=script,
            bbox=bbox,
            polygon=polygon,
        )
    
    def _get_difficulty_settings(self, difficulty: str) -> Dict:
        """Get settings for difficulty level."""
        
        if difficulty == 'easy':
            return {
                'font_min': 30,
                'font_max': 60,
                'perspective_prob': 0.1,
                'perspective_max': 5,
                'rotate_prob': 0.1,
                'rotate_max': 5,
                'blur_max': 0.5,
                'noise_max': 0.02,
                'jpeg_min': 80,
            }
        elif difficulty == 'medium':
            return {
                'font_min': 20,
                'font_max': 50,
                'perspective_prob': 0.3,
                'perspective_max': 15,
                'rotate_prob': 0.3,
                'rotate_max': 10,
                'blur_max': 1.5,
                'noise_max': 0.05,
                'jpeg_min': 60,
            }
        else:  # hard
            return {
                'font_min': 12,
                'font_max': 40,
                'perspective_prob': 0.5,
                'perspective_max': 25,
                'rotate_prob': 0.5,
                'rotate_max': 20,
                'blur_max': 2.5,
                'noise_max': 0.1,
                'jpeg_min': 30,
            }
    
    def generate_batch(
        self,
        n: int,
        **kwargs
    ) -> List[TextSample]:
        """Generate batch of samples."""
        return [self.generate(**kwargs) for _ in range(n)]
    
    def generate_dataset(
        self,
        output_dir: str,
        n_samples: int = 10000,
        difficulty_mix: Dict[str, float] = None,
    ):
        """
        Generate full dataset and save to disk.
        
        Args:
            output_dir: Directory to save images and annotations
            n_samples: Number of samples to generate
            difficulty_mix: Dict mapping difficulty to proportion
        """
        import json
        from tqdm import tqdm
        
        output_dir = Path(output_dir)
        (output_dir / 'images').mkdir(parents=True, exist_ok=True)
        
        difficulty_mix = difficulty_mix or {'easy': 0.3, 'medium': 0.5, 'hard': 0.2}
        
        annotations = []
        
        for i in tqdm(range(n_samples), desc='Generating'):
            # Choose difficulty
            r = random.random()
            cumsum = 0
            for diff, prob in difficulty_mix.items():
                cumsum += prob
                if r < cumsum:
                    difficulty = diff
                    break
            
            # Generate sample
            sample = self.generate(difficulty=difficulty)
            
            # Save image
            img_path = output_dir / 'images' / f'{i:06d}.jpg'
            Image.fromarray(sample.image).save(img_path, quality=95)
            
            # Add annotation
            annotations.append({
                'image': f'images/{i:06d}.jpg',
                'text': sample.text,
                'script': sample.script,
                'bbox': sample.bbox,
                'polygon': sample.polygon,
            })
        
        # Save annotations
        with open(output_dir / 'annotations.json', 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        print(f'Generated {n_samples} samples in {output_dir}')


if __name__ == '__main__':
    # Test synthesizer
    synth = IndianTextSynthesizer()
    
    # Generate sample for each script
    for script in synth.SCRIPTS:
        sample = synth.generate(script=script, difficulty='medium')
        print(f'{script}: "{sample.text}" -> {sample.bbox}')
        
        # Save sample
        Image.fromarray(sample.image).save(f'sample_{script}.jpg')
    
    print('Samples saved!')
