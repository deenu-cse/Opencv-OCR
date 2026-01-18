from fastapi import FastAPI, UploadFile, File, Form
from io import BytesIO
import numpy as np
import cv2
import json
import re

from ocr import extract_text_from_image_bytes  # we'll modify this
from llm_extractor import extract_fields

app = FastAPI()

def safe_json_parse(text: str):
    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}

@app.post("/extract")
async def extract_data(
    image: UploadFile = File(...),
    fields: str = Form(...)
):
    # Read image bytes directly
    image_bytes = await image.read()

    # OCR
    ocr_text = extract_text_from_image_bytes(image_bytes)

    # LLM
    fields_list = [f.strip() for f in fields.split(",")]
    extracted_data = extract_fields(ocr_text, fields_list)
    parsed_data = safe_json_parse(extracted_data)

    return {
        "requested_fields": fields_list,
        "ocr_text": ocr_text,
        "extracted_data": parsed_data
    }
