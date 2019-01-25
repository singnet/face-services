import sys
import io
import logging
import base64
import time

import concurrent.futures as futures
import grpc

import cv2
import dlib
from skimage import io as ioimg

import services.common
import services.grpc.face_recognition_pb2_grpc
from services.grpc.face_common_pb2 import BoundingBox, Point2D
from services.grpc.face_recognition_pb2 import FaceRecognitionHeader, FaceIdentity, FaceRecognitionResponse, FaceRecognitionRequest


landmark5_predictor_path = "models/shape_predictor_5_face_landmarks.dat"
landmark_predictor = dlib.shape_predictor(landmark5_predictor_path)

recognition_model_path = "models/dlib_face_recognition_resnet_model_v1.dat"
facerec = dlib.face_recognition_model_v1(recognition_model_path)

log = logging.getLogger(__package__ + "." + __name__)


class FaceRecognitionServicer(services.grpc.face_recognition_pb2_grpc.FaceRecognitionServicer):

    # TODO: support precalculated landmarks and prealigned face images, needs dlib API changes
    # landmark prediction
    #def RecogniseFacePreAligned(self, request_iterator, context):

    def RecogniseFace(self, request_iterator, context):
        start_time = time.time()
        image_data = bytearray()
        header = None

        for i, data in enumerate(request_iterator):
            if i == 0:
                if data.HasField("header"):
                    header = data.header
                    continue
                else:
                    raise Exception("No header provided!")
            else:
                image_data.extend(bytes(data.image_chunk.content))

        img_bytes = io.BytesIO(image_data)
        img = ioimg.imread(img_bytes)

        # Drop alpha channel if it exists
        if img.shape[-1] == 4:
            img = img[:, :, :3]
            log.debug("Dropping alpha channel from image")

        if len(header.faces) == 0:
            # TODO make upstream call to face detect service.
            pass

        face_identities = []
        for bbox in header.faces:
            dlib_bbox = dlib.rectangle(bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h)

            detection_object = landmark_predictor(img, dlib_bbox)

            face_descriptor = facerec.compute_face_descriptor(img, detection_object, 10)

            face_identities.append(FaceIdentity(identity=face_descriptor))

        elapsed_time = time.time() - start_time
        log.debug("Completed face recognition request in %.3fs" % elapsed_time)
        return FaceRecognitionResponse(identities=face_identities)


def serve(max_workers=10, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    services.grpc.face_recognition_pb2_grpc.add_FaceRecognitionServicer_to_server(
        FaceRecognitionServicer(), server)
    server.add_insecure_port('[::]:%d' % port)
    return server


if __name__ == '__main__':
    parser = services.common.common_parser(__file__)
    args = parser.parse_args(sys.argv[1:])
    serve_args = {}
    services.common.main_loop(serve, serve_args, args)
