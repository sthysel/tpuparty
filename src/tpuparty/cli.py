import os
import time
from pathlib import Path
from typing import Dict, List

import click
import cv2 as cv
import numpy as np
from loguru import logger

from . import colors
from .model import Model
from .video import ReaderWorker


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
@click.option(
    '--fps',
    help='FPS playback for recordings',
    default=None,
    show_default=True,
)
@click.argument('source')
@click.version_option()
def cli(modeldir, source, confidence, fps):
    """
    Runs inference over source

    \b
    Examples:
    $ tpuparty "http://10.0.0.185/axis-cgi/mjpg/video.cgi?&camera=2"
    $ tpuparty 0
    """

    model = Model(model_folder=modeldir)

    with ReaderWorker(source=source, fps=fps) as reader:
        for count, timestamp, frame in reader.read():
            logger.debug(f'{count}, {timestamp}')

            # some web cams start producing empty frames
            try:
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            except cv.error as e:
                logger.error(e)
                continue

            detections = model.infer(frame_rgb)
            paint_detections(frame, detections, confidence)
            paint_metadata(frame, count, timestamp)

            cv.imshow('TPUParty', frame)

            if cv.waitKey(1) == ord('q'):
                break

    cv.destroyAllWindows()

def paint_metadata(frame, count, timestamp):
    h, w = frame.shape[:2]
    label = f'{count}, {timestamp}'
    (label_width, label_height), base_line = cv.getTextSize(
        text=label,
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        thickness=2,
    )
    label_ymin = max(h - 10, label_width + 10)
    cv.putText(
        frame,
        text=label,
        org=(w - label_width, label_ymin),
        fontFace=cv.FONT_HERSHEY_SIMPLEX,
        fontScale=0.7,
        color=colors.YELLOW,
        thickness=2,
    )


def paint_detections(frame, detections, confidence):
    for detection in detections:
        name = detection.get('name', '')
        score = detection.get('score', 0)
        pmin, pmax = detection.get('roi')
        if score >= confidence:
            cv.rectangle(
                frame,
                pt1=pmin,
                pt2=pmax,
                color=colors.BHP,
                thickness=4,
            )

            label = f'{name}, {int(score * 100)}%'
            label_size, base_line = cv.getTextSize(
                text=label,
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                thickness=2,
            )

            xmin, ymin = pmin
            label_ymin = max(ymin, label_size[1] + 10)
            cv.putText(
                frame,
                text=label,
                org=(xmin, label_ymin - 7),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7,
                color=colors.CYAN,
                thickness=2,
            )
