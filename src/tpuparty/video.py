import time
from threading import Event, Thread

import cv2 as cv
from loguru import logger


class ReaderWorker(Thread):
    """ Reads frames from source """
    def __init__(
        self,
        source=0,
        name='video_reader',
        fps=None,
    ):
        """ Image stream worker """
        super().__init__()
        self.name = name
        if fps is not None:
            self.fps = float(fps)
        else:
            self.fps = fps

        self.new_frame_ready = Event()
        self.consumed_count = 0

        self.reader = VideoStreamReader(
            source=source,
            name=name,
        )

        self.stopped = False

        # properties
        self.frame_count = 0
        self.frame_timestamp = time.time()
        self.frame = None

    def dropped(self):
        """
        Returns number of frames that was not read

        - Returns
            - count: int
        """
        return self.frame_count - self.consumed_count

    def run(self):
        self.reader.open()
        while not self.stopped:
            self.frame_count, self.frame_timestamp, self.frame = self.reader.next()
            self.new_frame_ready.set()
            if self.fps:
                time.sleep(1 / self.fps)

        self.reader.close()

    @logger.catch
    def read(self):
        """
        Return the latest frame from the reader worker

        - Returns
            - frame_count, int: number of this frame
            - timestamp, float: timestamp of frame
            - frame, np.array: image frame
        """

        while not self.stopped:
            if self.new_frame_ready.wait(0.5):
                self.new_frame_ready.clear()
                self.consumed_count += 1
                logger.debug(f'yielding frame {self.frame_count}')
                yield self.frame_count, self.frame_timestamp, self.frame
            else:
                yield self.frame_count, self.frame_timestamp, self.frame

    def stop(self):
        self.stopped = True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stopped = True


class VideoStreamReader:
    """
    Reader of video streams
    - Arguments:
        - source: (int or str) The url, filesystem path or id of the  video stream.
    """
    def __init__(
        self,
        source=0,
        name='reader',
    ):
        self.source = self._set_source_type(source)
        self._video = None
        self._frame_count = 0
        self.name = name

    def _set_source_type(self, source):
        logger.info(f'Video source: {source}')
        try:
            source = int(source)
            logger.info(f'Using local camera #{source}')
        except ValueError:
            pass
        return source

    def open(self):
        if self._video is None:
            self._video = cv.VideoCapture(self.source)

    def close(self):
        if self._video and self._video.isOpened():
            self._video.release()

    @logger.catch
    def next(self):
        """
        - Returns:
            - frame no / index  : integer value of the frame read
            - timestamp: integer value of the time the frame was read
            - frame: np.array of shape (h, w, 3)

        - Raises:
            - StopIteration: after it finishes reading the video  file
        """
        self.open()
        while True:
            if self._video.isOpened():
                success, frame = self._video.read()
                if not success:
                    if self._video.isOpened():
                        self._video.release()
                    self._video = cv.VideoCapture(self.source)
                    logger.warning(f'{self.name} video reader fail')
                else:
                    self._frame_count += 1
                    return self._frame_count, time.time(), frame.copy()
            else:
                time.sleep(1)
                self._video = cv.VideoCapture(self.source)

        raise StopIteration()
