"""
Image processing utilities for BRINC app.
"""

import base64
import io
import numpy as np
from PIL import Image


def get_base64_of_bin_file(bin_file):
    """Encode a binary file to base64 string."""
    try:
        with open(bin_file, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def get_themed_logo_base64(logo_file="logo.png", theme="dark"):
    """Return a recolored transparent PNG logo as base64.

    theme='dark'  -> white logo on transparent background
    theme='light' -> black logo on transparent background
    """
    try:
        target_rgb = (255, 255, 255) if str(theme).lower() == 'dark' else (0, 0, 0)
        with Image.open(logo_file).convert('RGBA') as img:
            alpha = img.getchannel('A')
            recolored = Image.new('RGBA', img.size, target_rgb + (0,))
            recolored.putalpha(alpha)
            buf = io.BytesIO()
            recolored.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return get_base64_of_bin_file(logo_file)


def get_transparent_product_base64(image_file="gigs.png", threshold=32):
    """Return product image as transparent PNG by removing near-black background."""
    try:
        with Image.open(image_file).convert('RGBA') as img:
            arr = np.array(img)
            r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
            mask = (r <= threshold) & (g <= threshold) & (b <= threshold)
            arr[:, :, 3] = np.where(mask, 0, a)
            result = Image.fromarray(arr, 'RGBA')
            buf = io.BytesIO()
            result.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return get_base64_of_bin_file(image_file)
