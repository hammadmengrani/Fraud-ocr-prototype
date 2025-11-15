# import os
# import re
# import cv2
# import easyocr

# IMAGES_DIR = "../images"

# def preprocess_image(img_path):
#     """Simple preprocessing"""
#     img = cv2.imread(img_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Just basic thresholding
#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
#     return thresh

# def extract_merchant(text_lines):
#     """Pick first meaningful line"""
#     for line in text_lines[:10]:
#         line = line.strip()
        
#         # Skip empty, too short, or mostly numbers
#         if not line or len(line) < 3:
#             continue
#         if sum(c.isdigit() for c in line) / len(line) > 0.5:
#             continue
        
#         # If mostly letters, return it
#         if sum(c.isalpha() for c in line) / len(line) > 0.5:
#             return line
    
#     return "Unknown"

# def extract_total(text_lines):
#     """Find amount near 'Total' keyword"""
#     full_text = " ".join(text_lines)
    
#     # Look for "Total: $XX.XX" pattern
#     match = re.search(r'Total[:\s]*\$?\s*([\d,]+\.\d{2})', full_text, re.IGNORECASE)
#     if match:
#         return float(match.group(1).replace(',', ''))
    
#     # Fallback: find all amounts and return largest
#     amounts = re.findall(r'[\d,]+\.\d{2}', full_text)
#     if amounts:
#         amounts = [float(a.replace(',', '')) for a in amounts]
#         return max(amounts)
    
#     return None

# def process_receipt(img_path):
#     """Main function"""
#     reader = easyocr.Reader(['en'], gpu=False)
    
#     # Preprocess
#     preprocessed = preprocess_image(img_path)
    
#     # OCR
#     ocr_results = reader.readtext(preprocessed, detail=0)
    
#     # Extract
#     merchant = extract_merchant(ocr_results)
#     total = extract_total(ocr_results)
    
#     return {
#         "merchant_name": merchant,
#         "total": total
#     }

# def main():
#     """Process all images"""
#     results = []
    
#     for file in os.listdir(IMAGES_DIR):
#         if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
#             continue
        
#         img_path = os.path.join(IMAGES_DIR, file)
        
#         print(f"\n{'='*50}")
#         print(f"Processing: {file}")
        
#         result = process_receipt(img_path)
        
#         print(f"Merchant: {result['merchant_name']}")
#         print(f"Total: ${result['total']}")
        
#         results.append({
#             "image": file,
#             "merchant_name": result['merchant_name'],
#             "total_amount": result['total']
#         })
    
#     # Save results
#     import json
#     with open("ocr_results.json", "w") as f:
#         json.dump(results, f, indent=2)
    
#     print(f"\n✓ Saved results to ocr_results.json")
#     return results

# if __name__ == "__main__":
#     main()