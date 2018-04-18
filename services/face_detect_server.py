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

cnn_face_detector_path = "models/mmod_human_face_detector.dat"
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# face detection/localization
face_detector_dlib_hog = dlib.get_frontal_face_detector()
face_detector_dlib_cnn = dlib.cnn_face_detection_model_v1(cnn_face_detector_path)
face_detector_opencv_haarcascade = cv2.CascadeClassifier(cascade_path)

num_face_detectors = 3
face_idx = 2

log = logging.getLogger(__package__ + "." + __name__)

class FaceDetectServicer(services.grpc.face_detect_pb2_grpc.FaceDetectServicer):
    def FindFace(self, request_iterator, context):
        start_time = time.time()

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


        # face detection
        # TODO: use intermediate representation that includes confidence?
        if face_idx == 0:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = face_detector_opencv_haarcascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(100, 100),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            dets = []
            # Convert to array of dlib rectangles
            for (x, y, w, h) in faces:
                dets.append(dlib.rectangle(x, y, x + w, y + h))

        elif face_idx == 1:
            dets = face_detector_dlib_hog(img, 1)

        else:
            cnn_dets = face_detector_dlib_cnn(img, 1)
            dets = []
            for cnn_d in cnn_dets:
                # different return type because it includes confidence, get the rect
                d = cnn_d.rect
                h = d.top() - d.bottom()
                # cnn max margin detector seems to cut off the chin and this confuses landmark predictor,
                # expand height by 10%
                dets.append(dlib.rectangle(d.left(), d.top(), d.right(), d.bottom() - int(h / 10.0)))

        faces = []
        for d in dets:
            faces.append(BoundingBox(x=d.left(), y=d.top(), w=d.right() - d.left(), h=d.bottom() - d.top()))
        return services.grpc.face_common_pb2.FaceDetections(face_bbox=faces)

def serve(max_workers=10, blocking=True, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    services.grpc.face_detect_pb2_grpc.add_FaceDetectServicer_to_server(
        FaceDetectServicer(), server)
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
    serve()
