import easyocr

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

    if bbox is not None:
        x, y, w, h = bbox
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

    # crop is already RGB (from PIL) — EasyOCR handles that fine
    try:
        results = _reader.readtext(crop, paragraph=False)
        texts = [text for (_, text, conf) in results if conf > confidence_threshold]
        return " ".join(texts).strip()
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return ""