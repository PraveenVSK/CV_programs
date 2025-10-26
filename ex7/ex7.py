import cv2
import os

# Paths (use raw strings or escape backslashes)
auth_path = r"C:\CV Programs\ex7\authorized.jpg"
live_path = r"C:\CV Programs\ex7\live_face.jpg"

# Quick existence check
if not os.path.exists(auth_path) or not os.path.exists(live_path):
    raise SystemExit("Error: check file paths!")

auth_img = cv2.imread(auth_path, cv2.IMREAD_GRAYSCALE)
live_img = cv2.imread(live_path, cv2.IMREAD_GRAYSCALE)

if auth_img is None or live_img is None:
    raise SystemExit("Error: failed to read one of the images.")

# Optional: resize to speed up and normalize scale (keeps aspect)
def resize_max(img, max_dim=600):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, (int(w*scale), int(h*scale)))
    return img

auth_img = resize_max(auth_img)
live_img = resize_max(live_img)

orb = cv2.ORB_create(500)  # limit keypoints for speed
kp1, des1 = orb.detectAndCompute(auth_img, None)
kp2, des2 = orb.detectAndCompute(live_img, None)

if des1 is None or des2 is None:
    raise SystemExit("Error: no descriptors found (face not clear).")

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Use ratio of good matches to keypoints rather than raw count
good_matches = [m for m in matches if m.distance < 60]  # distance threshold tweakable
score = len(good_matches) / max(1, min(len(kp1), len(kp2)))  # normalized score

print(f"Good matches: {len(good_matches)}, Score: {score:.2f}")

# Decision threshold (tune between 0.1 and 0.4 depending on dataset)
if score > 0.18:
    print("Authorized User [OK]")
else:
    print("Unauthorized User [X]")


# Visualize top matches
vis = cv2.drawMatches(auth_img, kp1, live_img, kp2, matches[:20], None, flags=2)
cv2.imshow("Matches", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
