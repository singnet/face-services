import services.grpc.face_detect_pb2_grpc
from services.grpc.face_common_pb2 import BoundingBox
import time
import concurrent.futures as futures
import grpc
from skimage import io as ioimg
import io

import cv2
import dlib
import logging
import argparse

log = logging.getLogger(__package__ + "." + __name__)


class FaceDetectServicer(services.grpc.face_detect_pb2_grpc.FaceDetectServicer):

    cnn_face_detector_path = "models/mmod_human_face_detector.dat"
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    def __init__(self, detection_algorithm='haar_cascade'):
        self.algorithm = detection_algorithm
        if self.algorithm not in ['haar_cascade', 'dlib_hog', 'dlib_cnn']:
            raise Exception("Unknown face detection algorithm used to initialise service")
        log.debug("FaceDetectServicer created with algorithm %s" % (self.algorithm,))

        self.face_detector = None

    def FindFace(self, request_iterator, context):
        # Would be faster to do this on initialisation, but unsure about grpc worker threads and thread-safety of
        # dlib and opencv.
        if self.algorithm == 'haar_cascade':
            self.face_detector = cv2.CascadeClassifier(self.cascade_path)
        elif self.algorithm == 'dlib_hog':
            self.face_detector = dlib.get_frontal_face_detector()
        elif self.algorithm == 'dlib_cnn':
            self.face_detector = dlib.cnn_face_detection_model_v1(self.cnn_face_detector_path)

        image_data = bytearray()

        for data in request_iterator:
            image_data.extend(bytes(data.content))

        img_bytes = io.BytesIO(image_data)
        img = ioimg.imread(img_bytes)
        log.debug("Received image with shape %s" % str(img.shape))

        # Drop alpha channel if it exists
        if img.shape[-1] == 4:
            img = img[:,:,:3]
            log.debug("Dropping alpha channel from image")

        faces = []
        if self.algorithm == 'haar_cascade':
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces_det = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(100, 100),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            dets = []
            # Convert to array of dlib rectangles
            for (x, y, w, h) in faces_det:
                dets.append(dlib.rectangle(x, y, x + w, y + h))

        elif self.algorithm == 'dlib_hog':
            dets = self.face_detector(img, 1)

        elif self.algorithm == 'dlib_cnn':
            cnn_dets = self.face_detector(img, 1)
            dets = []
            for cnn_d in cnn_dets:
                # different return type because it includes confidence, get the rect
                d = cnn_d.rect
                h = d.top() - d.bottom()
                # cnn max margin detector seems to cut off the chin and this confuses landmark predictor,
                # expand height by 10%
                dets.append(dlib.rectangle(d.left(), d.top(), d.right(), d.bottom() - int(h / 10.0)))
        else:
            return services.grpc.face_common_pb2.FaceDetections(face_bbox=faces)

        for d in dets:
            faces.append(BoundingBox(x=d.left(), y=d.top(), w=d.right() - d.left(), h=d.bottom() - d.top()))
        return services.grpc.face_common_pb2.FaceDetections(face_bbox=faces)


def serve(algorithm='haar_cascade', max_workers=10, blocking=True, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    services.grpc.face_detect_pb2_grpc.add_FaceDetectServicer_to_server(
        FaceDetectServicer(algorithm), server)
    server.add_insecure_port('[::]:%d' % port)
    server.start()
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    if not blocking:
        return server
    else:
        try:
            while True:
                time.sleep(_ONE_DAY_IN_SECONDS)
        except KeyboardInterrupt:
            server.stop(0)


if __name__ == '__main__':
    import sys

    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument("--port", help="port to bind", default=50051, type=int, required=False)
    args = parser.parse_args(sys.argv[1:])
    serve(port=args.port)
