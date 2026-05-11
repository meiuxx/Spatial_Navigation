import easyocr
import numpy as np
import cv2

_reader = None

def init_ocr(languages=['en'], gpu=True):
    """Call this once at startup (e.g., in globals.py)."""
    global _reader
    if _reader is None:
        print(f"Loading EasyOCR with languages {languages} (GPU={gpu})...")
        _reader = easyocr.Reader(languages, gpu=gpu)
    return _reader

def run_ocr(image_np, bbox=None, confidence_threshold=0.5):

    global _reader
    if _reader is None:
        init_ocr()

    # Crop if bbox given
    if bbox is not None:
        x, y, w, h = bbox
        # Ensure coordinates are within image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, image_np.shape[1] - x)
        h = min(h, image_np.shape[0] - y)
        if w > 0 and h > 0:
            crop = image_np[y:y+h, x:x+w]
        else:
            crop = image_np
    else:
        crop = image_np

    # EasyOCR expects BGR? Actually it works with RGB as well, but we convert to RGB just in case.
    # crop is already RGB (from PIL), so fine.
    try:
        results = _reader.readtext(crop, paragraph=False)
        texts = [text for (_, text, conf) in results if conf > confidence_threshold]
        # Join all detected text blocks (space separated)
        return " ".join(texts).strip()
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return ""