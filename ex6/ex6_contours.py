import cv2
import matplotlib.pyplot as plt

# 1. Read image in grayscale
img_path = r"C:\CV Programs\ex3\bone.jpg"  # Replace with your MRI/CT slice
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Check image path!")
    exit()

# 2. Apply Gaussian Blur to reduce noise
blur = cv2.GaussianBlur(img, (5, 5), 0)

# 3. Otsu’s Thresholding
_, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4. Find contours
contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Draw contours on a copy of original image
img_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color drawing
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

# 5. Display results
titles = ['Original', 'Gaussian Blur', "Otsu's Threshold", 'Contours']
images = [img, blur, otsu, img_contours]

for i in range(4):
    plt.subplot(1, 4, i+1)
    if i == 3:  # Show color image for contours
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()
