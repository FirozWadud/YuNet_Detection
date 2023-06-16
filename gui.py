import cv2
from tkinter import *
from PIL import Image, ImageTk
import os

# Implement the YuNet class
class YuNet:
    def __init__(self, model_path, input_size=[320, 320], conf_threshold=0.6, nms_threshold=0.3, top_k=5000, backend_id=0, target_id=0):
        self._model_path = model_path
        self._input_size = tuple(input_size)
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._backend_id = backend_id
        self._target_id = target_id

        self._model = self._create_model()

    def _create_model(self):
        return cv2.FaceDetectorYN.create(
            model=self._model_path,
            config="",
            input_size=self._input_size,
            score_threshold=self._conf_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id)

    def set_backend_and_target(self, backend_id, target_id):
        self._backend_id = backend_id
        self._target_id = target_id
        self._model = self._create_model()

    def set_input_size(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        return self._model.detect(image)[1]

# Create a YuNet instance
model_path = '../FaceDetection2/face_detection_yunet_2022mar.onnx'  # replace with your model path
model = YuNet(model_path=model_path)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Global variable to hold the latest frame
frame = None

def video_stream():
    global frame
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Set the input size for the model
    h, w, _ = frame.shape
    model.set_input_size([w, h])

    # Detect faces
    results = model.infer(frame)

    # Draw bounding boxes on the frame
    for det in (results if results is not None else []):
        bbox = det[0:4].astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lbl.imgtk = imgtk
    lbl.configure(image=imgtk)
    window.after(1, video_stream)

def capture_face():
    global frame

    # Set the input size for the model
    h, w, _ = frame.shape
    model.set_input_size([w, h])

    # Detect faces
    results = model.infer(frame)

    # Crop and save faces
    for i, det in enumerate(results if results is not None else []):
        bbox = det[0:4].astype(int)
        cropped = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        cv2.imwrite(os.path.join('images', f'face_{i}.jpg'), cropped)

window = Tk()
window.title("Webcam Live Video Feed with Face Detection")
window.geometry("800x600")
window.resizable(False, False)  # Prevent resizing the window

lbl = Label(window)
lbl.pack()

btn = Button(window, text="Capture", command=capture_face)
btn.pack()

# Close window and stop video capturing
def close_window():
    cap.release()
    window.destroy()

window.protocol("WM_DELETE_WINDOW",  close_window)  # exit cleanup

video_stream()
window.mainloop()
