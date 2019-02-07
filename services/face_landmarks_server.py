import io
import time
import sys
import base64
import logging

import grpc
import concurrent.futures as futures

from skimage import io as ioimg
import dlib

import services.common
import services.grpc.face_landmarks_pb2_grpc
from services.grpc.face_common_pb2 import BoundingBox, Point2D, FaceLandmarks, FaceLandmarkDescriptions
from services.grpc.face_landmarks_pb2 import FaceLandmarkResponse


landmark68_predictor_path = "models/shape_predictor_68_face_landmarks.dat"
landmark5_predictor_path = "models/shape_predictor_5_face_landmarks.dat"

# landmark prediction
landmark_predictors = {
    "68": dlib.shape_predictor(landmark68_predictor_path),
    "5": dlib.shape_predictor(landmark5_predictor_path),
}

log = logging.getLogger(__package__ + "." + __name__)

landmark68_descriptions = (
            (["left of face"] * 7) +
            (["chin"] * 3) +
            (["right of face"] * 7) +
            (["left eye brow"] * 5) +
            (["right eye brow"] * 5) +
            (["nose bridge"] * 4) +
            (["nostrils"] * 5) +
            (["left eye"] * 6) +
            (["right eye"] * 6) +
            (["outer lips"] * 12) +
            (["inner lips"] * 8)
)
landmark5_descriptions = [
    "outer left eye",
    "inner left eye",
    "end of nose",
    "inner right eye",
    "outer right eye"
]


def landmark_finder(img, bbox, model):
    points = []
    dlib_bbox = dlib.rectangle(bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h)

    detection_object = landmark_predictors[model](img, dlib_bbox)
    detected_landmarks = detection_object.parts()
    for p in detected_landmarks:
        points.append(Point2D(x=p.x, y=p.y))

    return points


class FaceLandmarkServicer(services.grpc.face_landmarks_pb2_grpc.FaceLandmarkServicer):


    def GetLandmarkModels(self, request, context):
        models = [
            FaceLandmarkDescriptions(landmark_model="68", landmark_description=landmark68_descriptions),
            FaceLandmarkDescriptions(landmark_model="5", landmark_description=landmark5_descriptions)
        ]
        return services.grpc.face_common_pb2.FaceLandmarkModels(model=models)

    def GetLandmarks(self, request, context):
        start_time = time.time()
        image_data = bytearray()

        header = request.header
        image_data.extend(bytes(request.image_chunk.content))

        img_bytes = io.BytesIO(image_data)
        img = ioimg.imread(img_bytes)

        # Drop alpha channel if it exists
        if img.shape[-1] == 4:
            img = img[:, :, :3]
            log.debug("Dropping alpha channel from image")

        if len(header.faces.face_bbox) == 0:
            # TODO make upstream call to face detect service.
            pass

        face_landmarks = []
        for bbox in header.faces.face_bbox:
            points = landmark_finder(img, bbox, header.landmark_model)
            face_landmarks.append(FaceLandmarks(landmark_model=header.landmark_model, point=points))

        elapsed_time = time.time() - start_time
        log.debug("Completed face landmark detection request in %.3fs" % elapsed_time)
        return FaceLandmarkResponse(landmarked_faces=face_landmarks)


def serve(max_workers=10, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    services.grpc.face_landmarks_pb2_grpc.add_FaceLandmarkServicer_to_server(
        FaceLandmarkServicer(), server)
    server.add_insecure_port('[::]:%d' % port)
    return server


if __name__ == '__main__':
    parser = services.common.common_parser(__file__)
    args = parser.parse_args(sys.argv[1:])
    serve_args = {}
    services.common.main_loop(serve, serve_args, args)
