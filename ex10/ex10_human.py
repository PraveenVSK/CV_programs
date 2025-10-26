# ex10_human.py
# pip install mediapipe opencv-python numpy

import cv2, mediapipe as mp, numpy as np, os
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Set your full folder paths
IMG_DIR = r"C:\CV Programs\ex10\images"
OUT_DIR = r"C:\CV Programs\ex10\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def occlude(img):
    h,w = img.shape[:2]
    img2 = img.copy()
    cv2.rectangle(img2, (w//4, h//3), (3*w//4, 2*h//3), (0,0,0), -1)
    return img2

def rotate(img, angle=30):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REPLICATE)

def darken(img, factor=0.4):
    return (img * factor).astype(np.uint8)

def eval_pose(img, pose):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    if not res.pose_landmarks:
        return 0.0, 0.0, None
    lm = res.pose_landmarks.landmark
    vis = [l.visibility for l in lm]
    detected = sum(v > 0.3 for v in vis)
    return detected/len(lm), np.mean(vis), res.pose_landmarks

conds = [("original", lambda x: x), ("occluded", occlude), ("rotated", rotate), ("dark", darken)]
summary = []

with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    for fname in os.listdir(IMG_DIR):
        if not fname.lower().endswith((".jpg",".png",".jpeg")): continue
        img = cv2.imread(os.path.join(IMG_DIR, fname))
        if img is None: continue
        row = {"image": fname}
        for cname,fn in conds:
            img_t = fn(img)
            rate, score, lm = eval_pose(img_t, pose)
            row[f"{cname}_rate"] = round(rate,3)
            row[f"{cname}_score"] = round(score,3)
            vis = img_t.copy()
            if lm:
                mp_drawing.draw_landmarks(vis, lm, mp_pose.POSE_CONNECTIONS)
            cv2.imwrite(os.path.join(OUT_DIR, f"{os.path.splitext(fname)[0]}_{cname}.jpg"), vis)
        summary.append(row)

print(f"{'Image':25s} | Orig_rate Orig_score | Occ_rate Occ_score | Rot_rate Rot_score | Dark_rate Dark_score")
print("-"*110)
for r in summary:
    print(f"{r['image'][:25]:25s} | "
          f"{r['original_rate']:9.3f} {r['original_score']:10.3f} | "
          f"{r['occluded_rate']:8.3f} {r['occluded_score']:9.3f} | "
          f"{r['rotated_rate']:8.3f} {r['rotated_score']:9.3f} | "
          f"{r['dark_rate']:8.3f} {r['dark_score']:9.3f}")

print(f"\nResults saved to: {OUT_DIR}")

