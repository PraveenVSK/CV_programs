# Ex.No.2 Edge, Corner and Line Detection
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = r"C:\CV Programs\ex11\road.jpg"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge Detection
sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
sobel = cv2.convertScaleAbs(sobel)   # Convert to 8-bit for display
canny = cv2.Canny(gray, 100, 200)

# Corner Detection
harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
img_harris = img.copy()
img_harris[harris > 0.01 * harris.max()] = [0, 0, 255]

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
img_shi = img.copy()
for c in corners.astype(int):
    x, y = c.ravel()
    cv2.circle(img_shi, (x, y), 3, (0, 255, 0), -1)

# Line Detection
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 100, 10)
img_hough = img.copy()
for l in lines:
    x1, y1, x2, y2 = l[0]
    cv2.line(img_hough, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Display all
titles = ['Original', 'Sobel', 'Canny', 'Harris', 'Shi-Tomasi', 'Hough Lines']
images = [img, sobel, canny, img_harris, img_shi, img_hough]

for i in range(6):
    plt.subplot(2, 3, i+1)
    if len(images[i].shape) == 2:
        plt.imshow(images[i], cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
