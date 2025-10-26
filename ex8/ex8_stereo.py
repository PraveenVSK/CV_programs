import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load images
left = cv2.imread(r"C:\CV Programs\ex8\left.jpeg", cv2.IMREAD_GRAYSCALE)
right = cv2.imread(r"C:\CV Programs\ex8\right.jpg", cv2.IMREAD_GRAYSCALE)

# Check images
if left is None or right is None:
    raise FileNotFoundError("Check image paths!")

# Resize right image to match left
right = cv2.resize(right, (left.shape[1], left.shape[0]))

# StereoBM matcher
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(left, right)

# Normalize disparity for display
disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_norm = np.uint8(disp_norm)

# Depth map (example values)
f = 800  # focal length in pixels
B = 0.1  # baseline in meters
depth = f * B / (disparity + 1e-6)

# Display all images
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.title("Left Image")
plt.imshow(left, cmap='gray')
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Right Image")
plt.imshow(right, cmap='gray')
plt.axis('off')

plt.subplot(2,2,3)
plt.title("Disparity Map")
plt.imshow(disp_norm, cmap='gray')
plt.axis('off')

plt.subplot(2,2,4)
plt.title("Depth Map")
plt.imshow(depth, cmap='plasma')
plt.axis('off')

plt.tight_layout()
plt.show()
