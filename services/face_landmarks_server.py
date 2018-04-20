import services.grpc.face_landmarks_pb2_grpc
from services.grpc.face_common_pb2 import BoundingBox, Point2D, FaceLandmarks, FaceLandmarkDescriptions
import time
import concurrent.futures as futures
import grpc
from skimage import io as ioimg
import io
import logging

import cv2
import dlib

landmark68_predictor_path = "models/shape_predictor_68_face_landmarks.dat"
landmark5_predictor_path = "models/shape_predictor_5_face_landmarks.dat"

# landmark prediction
landmark_predictors = {
    "68": dlib.shape_predictor(landmark68_predictor_path),
    "5": dlib.shape_predictor(landmark5_predictor_path),
}

log = logging.getLogger(__package__ + "." + __name__)

class FaceLandmarkServicer(services.grpc.face_landmarks_pb2_grpc.FaceLandmarkServicer):
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

    def GetLandmarkModels(self, request, context):
        models = [
            FaceLandmarkDescriptions(landmark_model="68", landmark_description=self.landmark68_descriptions),
            FaceLandmarkDescriptions(landmark_model="5", landmark_description=self.landmark5_descriptions)
        ]
        return services.grpc.face_common_pb2.FaceLandmarkModels(model=models)

    def GetLandmarks(self, request_iterator, context):
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

        if len(header.faces.face_bbox) == 0:
            # TODO make upstream call to face detect service.
            pass

        points = []
        for bbox in header.faces.face_bbox:
            dlib_bbox = dlib.rectangle(bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h)

            detection_object = landmark_predictors[header.landmark_model](img, dlib_bbox)
            detected_landmarks = detection_object.parts()

            for p in detected_landmarks:
                points.append(Point2D(x=p.x, y=p.y))

        elapsed_time = time.time() - start_time
        print(elapsed_time)
        return FaceLandmarks(landmark_model=header.landmark_model, point=points)

def serve(max_workers=10, blocking=True, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    services.grpc.face_landmarks_pb2_grpc.add_FaceLandmarkServicer_to_server(
        FaceLandmarkServicer(), server)
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
