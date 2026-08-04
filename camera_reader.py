import threading
import time
import cv2


class CameraReader:

    def __init__(self, source):

        self.cap = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None
        self.eof = False

        self.running = True

        self.lock = threading.Lock()

        self.thread = threading.Thread(
            target=self._reader,
            daemon=True,
        )
        self.thread.start()

        print("Camera backend:", self.cap.getBackendName())

    def _reader(self):

        while self.running:

            ret, frame = self.cap.read()

            if not ret:
                print("### END OF STREAM ###")
                with self.lock:
                    self.eof = True
                break

            with self.lock:
                self.frame = frame

    def read(self):

        while self.running:

            with self.lock:

                if self.eof:
                    return False, None

                if self.frame is not None:
                    return True, self.frame.copy()

            time.sleep(0.001)

        return False, None

    def isOpened(self):

        return self.cap.isOpened()

    def release(self):

        self.running = False

        self.thread.join(timeout=1)

        self.cap.release()