import cv2 as cv
import os
import glob
import numpy as np


class YuNet:
    def __init__(self, model_path, input_size=[320, 320], conf_threshold=0.6, nms_threshold=0.3, top_k=5000,
                 backend_id=0, target_id=0):
        self._model_path = model_path
        self._input_size = tuple(input_size)
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._backend_id = backend_id
        self._target_id = target_id
        self._model = self._create_model()

    def _create_model(self):
        return cv.FaceDetectorYN.create(
            model=self._model_path,
            config="",
            input_size=self._input_size,
            score_threshold=self._conf_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id)

    def infer(self, image):
        # Resize the image
        image_resized = cv.resize(image, self._input_size)
        return self._model.detect(image_resized)[1]


# YuNet model path
model_path = 'face_detection_yunet_2022mar.onnx'  # replace with your model path
model = YuNet(model_path=model_path)

# Directories
input_dir = "/home/firoz/Desktop/face_dataset"
output_dir = "/home/firoz/Desktop/output_faces"

# Create output directory if doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all images in the directory and its subdirectories
for folder_path in glob.glob(os.path.join(input_dir, '*')):
    folder_name = os.path.basename(folder_path)
    output_folder = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    for file_path in glob.glob(os.path.join(folder_path, '*')):
        img = cv.imread(file_path)
        if img is None:
            continue

        # Detect faces
        detections = model.infer(img)

        for i, detection in enumerate(detections):
            bbox = detection[0:4].astype(np.int32)
            # Adjust bounding box for original image size
            h, w = img.shape[:2]
            x = bbox[0] * w // model._input_size[0]
            y = bbox[1] * h // model._input_size[1]
            w = bbox[2] * w // model._input_size[0]
            h = bbox[3] * h // model._input_size[1]
            face = img[y:y + h, x:x + w]

            # Ensure face is not empty
            if face.size == 0:
                print(f"No face detected in {file_path} or bounding box coordinates are incorrect.")
                continue

            # Save the cropped face
            face_file_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_face{i}.jpg")
            cv.imwrite(face_file_path, face)


