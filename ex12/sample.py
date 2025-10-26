# https://www.kaggle.com/models/spsayakpaul/deeplabv3-xception65?utm_source=chatgpt.com

import cv2
import numpy as np
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=r"C:\CV Programs\ex12\vsk.tflite")
interpreter.allocate_tensors()

# Load image
img_path = r"C:\CV Programs\ex12\road.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preprocess: resize and normalize
img_resized = cv2.resize(img_rgb, (513, 513))
input_data = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

# Set input tensor
input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output tensor
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

# Post-process
segmentation_mask = np.argmax(output_data, axis=-1)
segmentation_mask_colored = cv2.applyColorMap(segmentation_mask.astype(np.uint8), cv2.COLORMAP_JET)

# Resize mask to original image size
segmentation_mask_colored = cv2.resize(segmentation_mask_colored, (img.shape[1], img.shape[0]))

# Overlay
overlay = cv2.addWeighted(img, 0.7, segmentation_mask_colored, 0.3, 0)

# Display
cv2.imshow("Segmentation Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
