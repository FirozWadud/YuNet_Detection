
# Face Detection with YuNet

This code implements YuNet, a Convolutional Neural Network (CNN)-based face detection model. It provides functionality to detect faces in images or from video captured by a camera. The code supports command-line arguments for specifying input, model path, backend and target IDs, confidence threshold, NMS threshold, top K detections, and visualization options. 

The code assumes the availability of the YuNet model file ('face_detection_yunet_2022mar.onnx') and the corresponding model parameters for initialization.


## Run 

Clone the project

```bash
  git clone https://github.com/FirozWadud/YuNet_Detection.git
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install -r requirements.txt
```

run main.py

```bash
  python main.py
```


## Credits

[opencv_zoo](https://github.com/opencv/opencv_zoo)

