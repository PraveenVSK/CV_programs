# Ex.No.5 Implementation of Face Recognition System
import cv2
import time
from mtcnn import MTCNN  # TensorFlow-based detector

def caffe_detection(image_path):
    modelFile = r"C:\CV Programs\ex4\res10_300x300_ssd_iter_140000.caffemodel"
    configFile = r"C:\CV Programs\ex4\deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return 0, 0
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)

    start = time.time()
    detections = net.forward()
    end = time.time()

    count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.3:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            count += 1

    cv2.imshow("Caffe Detection", image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    return count, end - start


def tensorflow_detection(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found!")
        return 0, 0

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = MTCNN()

    start = time.time()
    results = detector.detect_faces(rgb_image)
    end = time.time()

    for res in results:
        x, y, w, h = res['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("TensorFlow (MTCNN) Detection", image)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    return len(results), end - start


# --- Run both detections ---
img_path = r"C:\CV Programs\ex4\faces.jpg"

caffe_faces, caffe_time = caffe_detection(img_path)
tf_faces, tf_time = tensorflow_detection(img_path)

# --- Results ---
print("\n=== Comparison Report ===")
print(f"Caffe Model - Faces Detected: {caffe_faces}, Time: {caffe_time:.3f} sec")
print(f"TensorFlow (MTCNN) - Faces Detected: {tf_faces}, Time: {tf_time:.3f} sec")

# Accuracy and speed summary
if tf_faces > caffe_faces:
    print("TensorFlow detected more faces (higher accuracy).")
else:
    print("Caffe performed faster and suitable for real-time detection.")
