import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt

# Load images
images = [cv2.imread(f) for f in glob.glob(r"C:\CV Programs\ex2\dataset\*.jpg")]

# Function to compute histograms
def compute_histograms(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    gray_hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    b_hist = cv2.calcHist([img],[0],None,[256],[0,256])
    g_hist = cv2.calcHist([img],[1],None,[256],[0,256])
    r_hist = cv2.calcHist([img],[2],None,[256],[0,256])
    hsv_hist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
    
    return gray_hist, (b_hist,g_hist,r_hist), hsv_hist

# Compute histograms for all images
dataset_hists = [compute_histograms(img) for img in images]

# Query image
query = cv2.imread(r"c:\CV Programs\ex2\query.jpg")
if query is None:
    print("query.jpg not found! Check the file path.")
    exit()

q_gray, q_rgb, q_hsv = compute_histograms(query)

# Plot histograms for the query image
plt.figure(figsize=(18,5))

# Grayscale
plt.subplot(1,3,1)
plt.plot(q_gray, color='black')
plt.title("Grayscale Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

# RGB
plt.subplot(1,3,2)
colors = ('b','g','r')
for hist,color in zip(q_rgb, colors):
    plt.plot(hist, color=color)
plt.title("RGB Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

# HSV (Hue vs Saturation)
plt.subplot(1,3,3)
plt.imshow(q_hsv, interpolation='nearest')
plt.title("HSV Histogram (Hue vs Saturation)")
plt.xlabel("Saturation")
plt.ylabel("Hue")

plt.tight_layout()
plt.show()

# Compare histograms (example: correlation)
def compare_hist(query_hist, dataset_hists):
    scores = []
    for hists in dataset_hists:
        score = cv2.compareHist(query_hist, hists[0], cv2.HISTCMP_CORREL) # Grayscale example
        scores.append(score)
    return np.argsort(scores)[::-1]  # Descending order

best_matches = compare_hist(q_gray, dataset_hists)

# Display top 3 matches
for idx in best_matches[:3]:
    img_rgb = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Match {idx+1}")
    plt.axis('off')
    plt.show()
