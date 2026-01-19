import cv2
import pytesseract
import numpy as np
import platform

# Only set tesseract path on Windows
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# On Linux (Render), tesseract is in PATH - no need to set

def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """Extract text from image bytes using OCR"""
    try:
        # Convert bytes to NumPy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return "Error: Could not decode image"

        # Preprocessing
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        custom_config = r"--oem 3 --psm 6"
        text = pytesseract.image_to_string(
            thresh,
            lang="eng",
            config=custom_config
        )

        return text.strip()
    except Exception as e:
        return f"Error during OCR: {str(e)}"
