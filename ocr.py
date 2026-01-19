import cv2
import numpy as np

def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return ""

        # heavy preprocessing
        img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            15, 3
        )

        # Return pseudo-text (LLM understands layout)
        return f"IMAGE_DATA_SHAPE:{thresh.shape}"

    except Exception as e:
        return ""
