import importlib.util
import os
import time
from pathlib import Path
from typing import Dict, List

import click
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

COLORS = {
    'BHP': (0, 84, 230),
    'BLUE': (255, 0, 0),
    'BROWN': (33, 67, 101),
    'GREEN': (0, 255, 0),
    'GREY': (188, 188, 188),
    'ORANGE': (0, 128, 255),
    'PINK': (255, 153, 255),
    'RED': (0, 0, 255),
    'YELLOW': (51, 255, 255),
    'WHITE': (0, 0, 0),
    'OLIVE': (69, 146, 148),
    'CYAN': (255, 255, 0),
}


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

    def infer(self, frame:np.array) -> List[Dict]:
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



        detections = []
        for i, score in enumerate(scores):
            try:
                name = self.labels[int(classes[i])]
            except IndexError as e:
                logger.error(e)
                name = f'index {i}'
            detections.append(dict(
                score=score,
                name=name,
                roi=boxes[i],
            ))
        return detections


@click.command(context_settings=dict(max_content_width=120))
@click.option(
    '--modeldir',
    help='Directory containing the model weight and label files',
    default='~/models/coco/',
    show_default=True,
)
@click.option(
    '-c',
    '--confidence',
    help='Confidence threshold for object inference',
    default=0.1,
    show_default=True,
)
@click.argument('source')
@click.version_option()
def cli(modeldir, source, confidence):
    """
    Runs inference over source

    Example: tpuparty "http://10.0.0.185/axis-cgi/mjpg/video.cgi?&camera=2"
    """
    model = Model(model_folder=modeldir)
    video = cv.VideoCapture(source)
    w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))


    writer = cv.VideoWriter(
        filename=f'out-{os.path.basename(source)}',
        fourcc=cv.VideoWriter_fourcc(*'MJPG'),
        fps=5.0,
        frameSize=(w, h),
        isColor=True,
    )

    while video.isOpened():
        ret, frame = video.read()
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        detections = model.infer(frame_rgb)

        for detection in detections:
            name = detection.get('name', '')
            score = detection.get('score', 0)
            roi = detection.get('roi')
            if score >= confidence:
                # get bounding box coordinates and draw box interpreter can return coordinates that
                # are outside of image dimensions, truncate them to be within image shape
                ymin = int(max(1, (roi[0] * h)))
                xmin = int(max(1, (roi[1] * w)))
                ymax = int(min(h, (roi[2] * h)))
                xmax = int(min(w, (roi[3] * w)))

                cv.rectangle(
                    frame,
                    pt1=(xmin, ymin),
                    pt2=(xmax, ymax),
                    color=COLORS.get('BHP'),
                    thickness=4,
                )

                label = f'{name}, {int(score * 100)}%'
                label_size, base_line = cv.getTextSize(
                    text=label,
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    thickness=2,
                )
                label_ymin = max(ymin, label_size[1] + 10)
                cv.putText(
                    frame,
                    text=label,
                    org=(xmin, label_ymin - 7),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.7,
                    color=COLORS['WHITE'],
                    thickness=2,
                )

        cv.imshow('TPUParty', frame)
        writer.write(frame)

        if cv.waitKey(1) == ord('q'):
            break
        time.sleep(0.1)

    video.release()
    writer.release()
    cv.destroyAllWindows()
