import cv2
import matplotlib.pyplot as plt

# 1. Read image in grayscale
img_path = r"C:\CV Programs\ex3\bone.jpg"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: Check image path!")
    exit()

# 2. Simple Thresholding
_, simple = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# 3. Adaptive Thresholding
adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

# 4. Otsu’s Thresholding
_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 5. Display results
titles = ['Original', 'Simple', 'Adaptive', 'Otsu']
images = [img, simple, adaptive, otsu]

for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.show()
