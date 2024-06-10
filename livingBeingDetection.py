import warnings
import numpy as np
import cv2

from ultralytics import YOLO
warnings.filterwarnings('ignore')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
              'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
              'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
              'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
              'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']

model = YOLO("weights/yolov8-med-25.9params.pt","v8")

def tensor_to_cv2(tensor_image):
    numpy_image = tensor_image.permute(1, 2, 0).numpy()
    numpy_image = (numpy_image * 255).astype(np.uint8)
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return cv2_image


def livingDetection(img):
    frame = tensor_to_cv2(img)
    if frame is None:
        print("Error: Could not open or read the image.")
        exit()

    detect_params = model.predict(source=[frame], conf=0.3, save=False)
    DP = detect_params[0].cpu().numpy()

    if len(DP) != 0:
        # Iterate over detected objects
        for i in range(len(detect_params[0])):
            boxes = detect_params[0].boxes
            box = boxes[i]
            clsID = box.cls.cpu().numpy()[0]
            conf = box.conf.cpu().numpy()[0]
            bb = box.xyxy.cpu().numpy()[0]

            # Check if the detected object belongs to living beings
            class_name = class_list[int(clsID)]
            living_being_label = classify_living_being(class_name)

            if living_being_label == 'person':
                # Draw bounding box for persons
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    (255, 0, 0),  # Color for persons
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    "Person",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 0, 0),
                    2,
                )

            elif living_being_label == 'animal':
                # Draw bounding box for animals
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    (0, 0, 255),  # Color for animals
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    "Animal",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (0, 0, 255),
                    2,
                )

            elif living_being_label == 'bird':

                # Draw bounding box for birds
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    (1, 50, 32),  # Color for birds
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    "Bird",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (1, 50, 32),
                    2,
                )
    return frame

def classify_living_being(class_name):
    if class_name == 'person':
        return 'person'
    elif class_name in ['cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']:
        return 'animal'
    elif class_name == 'bird':
        return 'bird'
    else:
        return None    