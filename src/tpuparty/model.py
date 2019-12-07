import importlib.util
from pathlib import Path
from typing import Dict, List

import cv2 as cv
import numpy as np
from loguru import logger

# If tensorflow is not installed, import interpreter from tflite_runtime, else import from regular tensorflow
pkg = importlib.util.find_spec('tensorflow')
if pkg is None:
    from tflite_runtime.interpreter import Interpreter
    from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    from tensorflow.lite.python.interpreter import load_delegate


class Model:
    def __init__(
        self,
        model_folder,
        graph_file='graph.tflite',
        label_file='labels.txt',
    ):

        model_path = Path(model_folder).expanduser() / Path(graph_file)
        label_file = Path(model_folder).expanduser() / Path(label_file)

        # Load the label map
        with open(label_file, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        logger.info('loading tensorflow model')
        self.interpreter = Interpreter(
            model_path=str(model_path),
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')],
        )
        self.interpreter.allocate_tensors()
        logger.info('done loading tensorflow model')

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # the model's expected shape
        self.height = self.input_details[0]['shape'][1]
        self.width = self.input_details[0]['shape'][2]
        logger.info(f'model shape: ({self.height}x{self.width})')

        self.is_quantized = self.input_details[0]['dtype'] == np.float32
        self.input_mean = 127.5
        self.input_std = 127.5

    def fix_roi(self, size, roi):
        h, w = size
        # interpreter can return coordinates that are outside of image dimensions, truncate them to
        # be within image shape
        ymin = int(max(1, (roi[0] * h)))
        xmin = int(max(1, (roi[1] * w)))
        ymax = int(min(h, (roi[2] * h)))
        xmax = int(min(w, (roi[3] * w)))
        return (
            (xmin, ymin),
            (xmax, ymax),
        )

    def infer(self, frame: np.array) -> List[Dict]:
        """
        Infer given frame

        - Arguments
          - frame: np.array, the frame to infer on

        - Returns
          - list of detections
        """
        # resize input frame to expected model size
        frame_resized = cv.resize(frame, (self.width, self.height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # normalize pixel values if floating model, model is not quantized
        if self.is_quantized:
           input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # do the inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # bounding box coordinates of detected objects
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        # class index of detected objects
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        # confidence of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        data = zip(boxes, classes, scores)

        detections = []
        for roi, klass, score in data:
            name = self.labels[int(klass)]
            detections.append(dict(
                score=score,
                name=name,
                roi=self.fix_roi(size=frame.shape[:2], roi=roi),
            ))
        return detections
