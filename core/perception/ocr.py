import pytesseract
import cv2

def run_ocr(image_np, lang='eng'):
    # Convert RGB to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        text = pytesseract.image_to_string(thresh, lang=lang).strip()
        return text if text else None
    except Exception as e:
        print(f"OCR error: {e}")
        return None