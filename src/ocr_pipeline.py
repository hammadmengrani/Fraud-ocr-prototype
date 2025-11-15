import requests
import os
import re
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OCR_API_KEY")

def ocr_space_file(img_path):
    with open(img_path, "rb") as f:
        r = requests.post(
            "https://api.ocr.space/parse/image",
            files={img_path: f},
            data={"apikey": API_KEY, "language": "eng"}
        )
    return r.json()

def extract_merchant(text_lines):
    for line in text_lines[:10]:
        line = line.strip()
        if not line or len(line) < 3:
            continue
        if sum(c.isdigit() for c in line)/len(line) > 0.5:
            continue
        if sum(c.isalpha() for c in line)/len(line) > 0.5:
            return line
    return "Unknown"

def extract_total(text_lines):
    full_text = " ".join(text_lines)
    match = re.search(r'Total[:\s]*\$?\s*([\d,]+\.\d{2})', full_text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(',', ''))
    amounts = re.findall(r'[\d,]+\.\d{2}', full_text)
    if amounts:
        amounts = [float(a.replace(',', '')) for a in amounts]
        return max(amounts)
    return None

def get_receipt_data(img_path):
    result_json = ocr_space_file(img_path)
    parsed_text = result_json["ParsedResults"][0]["ParsedText"].splitlines()
    merchant = extract_merchant(parsed_text)
    total = extract_total(parsed_text)
    return merchant, total
