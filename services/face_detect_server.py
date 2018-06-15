import io
import sys
import logging
import base64

import grpc
import concurrent.futures as futures

from aiohttp import web
from jsonrpcserver.aio import methods
from jsonrpcserver.exceptions import InvalidParams

import cv2
import dlib
from skimage import io as ioimg

import services.common
import services.grpc.face_detect_pb2_grpc
from services.grpc.face_common_pb2 import BoundingBox


log = logging.getLogger(__package__ + "." + __name__)


def get_detector(algorithm):
    cnn_face_detector_path = "models/mmod_human_face_detector.dat"
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    face_detector = None
    if algorithm == 'haar_cascade':
        face_detector = cv2.CascadeClassifier(cascade_path)
    elif algorithm == 'dlib_hog':
        face_detector = dlib.get_frontal_face_detector()
    elif algorithm == 'dlib_cnn':
        face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detector_path)
    return face_detector


def face_detect(img, detector, algorithm):
    dets = []
    if algorithm == 'haar_cascade':
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces_det = detector.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(100, 100),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Convert to array of dlib rectangles
        for (x, y, w, h) in faces_det:
            dets.append(dlib.rectangle(x, y, x + w, y + h))

    elif algorithm == 'dlib_hog':
        dets = detector(img, 1)

    elif algorithm == 'dlib_cnn':
        cnn_dets = detector(img, 0)
        for cnn_d in cnn_dets:
            # different return type because it includes confidence, get the rect
            d = cnn_d.rect
            h = d.top() - d.bottom()
            # cnn max margin detector seems to cut off the chin and this confuses landmark predictor,
            # expand height by 10%
            dets.append(dlib.rectangle(d.left(), d.top(), d.right(), d.bottom() - int(h / 10.0)))
    return dets


class FaceDetectServicer(services.grpc.face_detect_pb2_grpc.FaceDetectServicer):

    def __init__(self, detection_algorithm='haar_cascade'):
        self.algorithm = detection_algorithm
        if self.algorithm not in ['haar_cascade', 'dlib_hog', 'dlib_cnn']:
            raise Exception("Unknown face detection algorithm used to initialise service")
        log.debug("FaceDetectServicer created with algorithm %s" % (self.algorithm,))

        self.face_detector = None

    def FindFace(self, request_iterator, context):
        # Would be faster to do this on initialisation, but unsure about grpc worker threads and thread-safety of
        # dlib and opencv.
        self.face_detector = get_detector(self.algorithm)
        if self.face_detector is None:
            raise Exception("Unknown algorithm")

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

        dets = face_detect(img, self.face_detector, self.algorithm)

        self.face_detector = None
        faces = []
        for d in dets:
            faces.append(BoundingBox(x=d.left(), y=d.top(), w=d.right() - d.left(), h=d.bottom() - d.top()))
        return services.grpc.face_common_pb2.FaceDetections(face_bbox=faces)


def serve(algorithm='haar_cascade', max_workers=10, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    services.grpc.face_detect_pb2_grpc.add_FaceDetectServicer_to_server(
        FaceDetectServicer(algorithm), server)
    server.add_insecure_port('[::]:%d' % port)
    return server


@methods.add
async def ping():
    return 'pong'


@methods.add
async def find_face(**kwargs):
    image = kwargs.get("image", None)
    algorithm = kwargs.get("algorithm", "dlib_cnn")

    if image is None:
        raise InvalidParams("image is required")

    face_detector = get_detector(algorithm)
    if face_detector is None:
        raise InvalidParams("unknown algorithm")

    binary_image = base64.b64decode(image)
    img_data = io.BytesIO(binary_image)
    img = ioimg.imread(img_data)

    # Drop alpha channel if it exists
    if img.shape[-1] == 4:
        img = img[:, :, :3]
        log.debug("Dropping alpha channel from image")

    dets = face_detect(img, face_detector, algorithm)

    faces = []
    for d in dets:
        faces.append(dict(x=d.left(), y=d.top(), w=d.right() - d.left(), h=d.bottom() - d.top()))
    return {'faces': faces}


async def handle(request):
    request = await request.text()
    response = await methods.dispatch(request)
    if response.is_notification:
        return web.Response()
    else:
        return web.json_response(response, status=response.http_status)


if __name__ == '__main__':
    parser = services.common.common_parser(__file__)
    args = parser.parse_args(sys.argv[1:])
    serve_args = {'algorithm': 'dlib_cnn'}
    services.common.main_loop(serve, serve_args, handle, args)