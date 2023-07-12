import argparse
import numpy as np
import cv2 as cv

# Class for the YuNet face recognition model
class YuNet:
    def __init__(self, model_path, input_size=[320, 320], conf_threshold=0.6, nms_threshold=0.3, top_k=5000, backend_id=0, target_id=0):
        # Initialization function with model parameters
        self._model_path = model_path
        self._input_size = tuple(input_size)
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._backend_id = backend_id
        self._target_id = target_id

        self._model = self._create_model()

    def _create_model(self):
        # Create the YuNet face detection model
        return cv.FaceDetectorYN.create(
            model=self._model_path,
            config="",
            input_size=self._input_size,
            score_threshold=self._conf_threshold,
            nms_threshold=self._nms_threshold,
            top_k=self._top_k,
            backend_id=self._backend_id,
            target_id=self._target_id)

    def set_backend_and_target(self, backend_id, target_id):
        # Update the backend and target IDs for the model
        self._backend_id = backend_id
        self._target_id = target_id
        self._model = self._create_model()

    def set_input_size(self, input_size):
        # Set the input size for the model
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        # Perform face detection inference on the given image
        return self._model.detect(image)[1]

# Function to parse command-line arguments
def parse_arguments():
    backend_target_pairs = [
        [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
        [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
        [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
        [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
        [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
    ]

    parser = argparse.ArgumentParser(description='YuNet: A Fast and Accurate CNN-based Face Detector.')
    parser.add_argument('--input', '-i', type=str)
    parser.add_argument('--model', '-m', type=str, default='face_detection_yunet_2022mar.onnx')
    parser.add_argument('--backend_target', '-bt', type=int, default=0)
    parser.add_argument('--conf_threshold', type=float, default=0.9)
    parser.add_argument('--nms_threshold', type=float, default=0.3)
    parser.add_argument('--top_k', type=int, default=5000)
    parser.add_argument('--save', '-s', action='store_true')
    parser.add_argument('--vis', '-v', action='store_true')
    args = parser.parse_args()

    return args, backend_target_pairs

# Function to visualize the results on the image
def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in (results if results is not None else []):
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)
    return output

# Main function
def main():
    args, backend_target_pairs = parse_arguments()

    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # Create an instance of the YuNet face recognition model
    model = YuNet(model_path=args.model,
                  input_size=[320, 320],
                  conf_threshold=args.conf_threshold,
                  nms_threshold=args.nms_threshold,
                  top_k=args.top_k,
                  backend_id=backend_id,
                  target_id=target_id)

    if args.input is not None:
        # Process a single image
        process_image(args, model)
    else:
        # Process video from camera
        process_camera(args, model)

# Function to process a single image
def process_image(args, model):
    image = cv.imread(args.input)
    h, w, _ = image.shape

    model.set_input_size([w, h])
    results = model.infer(image)

    print('{} faces detected.'.format(results.shape[0]))
    for idx, det in enumerate(results):
        print('{}: {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}'.format(
            idx, *det[:-1])
        )

    image = visualize(image, results)

    if args.save:
        print('Results saved to result.jpg\n')
        cv.imwrite('result.jpg', image)

    if args.vis:
        cv.namedWindow(args.input, cv.WINDOW_AUTOSIZE)
        cv.imshow(args.input, image)
        cv.waitKey(0)

# Function to process video from camera
def process_camera(args, model):
    cap = cv.VideoCapture(0)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    model.set_input_size([w, h])

    frame_counter = 0
    drop_rate = 3

    while cv.waitKey(1) < 0:
        has_frame, frame = cap.read()
        if not has_frame:
            print('No frames grabbed!')
            break

        frame_counter += 1

        if frame_counter % drop_rate != 0:
            continue

        results = model.infer(frame)
        frame_with_results = visualize(frame, results)

        cv.imshow('YuNet Demo', frame_with_results)

if __name__ == '__main__':
    main()
