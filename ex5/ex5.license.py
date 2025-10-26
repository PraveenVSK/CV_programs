# https://github.com/UB-Mannheim/tesseract/wiki
# pip install opencv-python-headless
# pip install numpy
# pip install pytesseract





# License Plate Detection using Contours + Tesseract
import cv2
import numpy as np
import pytesseract

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\CV Programs\Tesseract-OCR\tesseract.exe'

# State dictionary
states = {
 "AN": "Andaman and Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
 "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh", "DN": "Dadra and Nagar Haveli",
 "DD": "Daman and Diu", "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat", "HR": "Haryana",
 "HP": "Himachal Pradesh", "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala",
 "LD": "Lakshadweep", "MP": "Madhya Pradesh", "MH": "Maharashtra", "MN": "Manipur",
 "ML": "Meghalaya", "MZ": "Mizoram", "NL": "Nagaland", "OD": "Odisha",
 "PY": "Puducherry", "PN": "Punjab", "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu",
 "TR": "Tripura", "UP": "Uttar Pradesh", "WB": "West Bengal", "CG": "Chhattisgarh",
 "TS": "Telangana", "JH": "Jharkhand", "UK": "Uttarakhand"
}

def extract_plate(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Reduce noise

    # Edge detection
    edged = cv2.Canny(gray, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]  # Top 30 largest

    plate_img = None
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:  # Plate is likely rectangular
            x, y, w, h = cv2.boundingRect(approx)
            plate_img = img[y:y+h, x:x+w]

            # Draw rectangle on original image
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            break

    if plate_img is None:
        print("No license plate detected")
        return

    # Preprocess plate for OCR
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, plate_bin = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR
    text = pytesseract.image_to_string(
        plate_bin,
        config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )
    text = ''.join(e for e in text if e.isalnum()).upper()
    state_code = text[:2] if len(text) >= 2 else "NA"
    state_name = states.get(state_code, "Unknown")

    print(f"Detected Plate: {text}")
    print(f"State: {state_name}")

    # Annotate image
    cv2.putText(img, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(img, state_name, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show results
    cv2.imshow("Plate", plate_img)
    cv2.imshow("Result", img)
    cv2.imwrite("Detected_Plate.png", plate_img)
    cv2.imwrite("Result_Image.png", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run detection
extract_plate(r"C:\CV Programs\ex5\cars.jpeg")  # Update image path

