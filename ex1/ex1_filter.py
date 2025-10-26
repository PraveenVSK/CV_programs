import cv2
import numpy as np
import os

# --- Step 1: Load the image safely ---
img_path = r"C:\CV Programs\dog.jpg"   # change only if your file name is different

# Try reading the image
img = cv2.imread(img_path)

# Check if image loaded correctly
if img is None:
    print("Image not found or invalid format.")
    exit()

# Resize image for display
img = cv2.resize(img, (400, 400))

# --- Step 2: Add Gaussian Noise ---
gaussian_noise = np.random.normal(0, 20, img.shape).astype(np.uint8)
gaussian_noisy = cv2.add(img, gaussian_noise)

# --- Step 3: Add Salt & Pepper Noise ---
sp_noisy = img.copy()
prob = 0.02
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        rand = np.random.rand()
        if rand < prob / 2:
            sp_noisy[i, j] = 0
        elif rand < prob:
            sp_noisy[i, j] = 255

# --- Step 4: Apply Filters ---
avg_filter = cv2.blur(gaussian_noisy, (5, 5))
gauss_filter = cv2.GaussianBlur(gaussian_noisy, (5, 5), 0)
median_filter = cv2.medianBlur(sp_noisy, 5)

# --- Step 5: Display Results ---
cv2.imshow("Original Image", img)
cv2.imshow("Gaussian Noise", gaussian_noisy)
cv2.imshow("Averaging Filter", avg_filter)
cv2.imshow("Gaussian Filter", gauss_filter)
cv2.imshow("Salt & Pepper Noise", sp_noisy)
cv2.imshow("Median Filter", median_filter)

print("Press any key on an image window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
