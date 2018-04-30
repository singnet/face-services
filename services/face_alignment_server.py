import sys
import io
import os
import base64
import logging
import tempfile

import concurrent.futures as futures
import grpc

from aiohttp import web
from jsonrpcserver.aio import methods
from jsonrpcserver.exceptions import InvalidParams

import cv2
import dlib
from skimage import io as ioimg

import services.grpc.face_alignment_pb2_grpc
from services.grpc.face_alignment_pb2 import FaceAlignmentResponse, FaceAlignmentResponseHeader
from services.grpc.face_common_pb2 import BoundingBox, ImageRGB


log = logging.getLogger(__package__ + "." + __name__)

# TODO: due to limitations in how dlib exposes it's API, we can't arbitarily provide landmarks.
# The detection object is immutable and can only be obtained by landmark prediction.
# Joel will submit a patch to dlib, or reverse engineer the alignment algorithm.
# In the interim, this just forces a new landmark prediction based on the provided bounding box.
# It should be *exactly* the same regardless, since the landmark prediction service uses dlib,
# but isn't a long-term solution.
landmark68_predictor_path = "models/shape_predictor_68_face_landmarks.dat"
landmark5_predictor_path = "models/shape_predictor_5_face_landmarks.dat"

# landmark prediction
landmark_predictors = {
    "68": dlib.shape_predictor(landmark68_predictor_path),
    "5": dlib.shape_predictor(landmark5_predictor_path),
}

class FaceAlignmentServicer(services.grpc.face_alignment_pb2_grpc.FaceAlignmentServicer):

    def AlignFace(self, request_iterator, context):
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
        log.debug("Received image with shape %s" % str(img.shape))

        # Drop alpha channel if it exists
        if img.shape[-1] == 4:
            img = img[:,:,:3]
            log.debug("Dropping alpha channel from image")

        fh, temp_file = tempfile.mkstemp('.jpg')
        os.close(fh)
        temp_file_no_ext = ".".join(temp_file.rsplit('.')[:-1])

        #source_pts = np.float32([[p.x, p.y] for p in header.source.point])
        #target_pts = np.float32([[p.x, p.y] for p in header.target.point])

        for bbox in header.source_bboxes:

            d = dlib.rectangle(bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h)

            #num_pts = len(source_pts)
            num_pts = 5
            try:
                landmark_predictor = landmark_predictors[str(num_pts)]
            except KeyError:
                raise Exception("Incorrect number of landmarks")

            detection_object = landmark_predictor(img, d)

            chip_size = 150
            border = 0.2
            dlib.save_face_chip(img, detection_object, temp_file_no_ext, chip_size, border)

            # Playing with OpenCVs geometric transforms - they don't work out of the box
            # due to faces not being a plane.
            #sample_idx = np.random.choice(source_pts.shape[0], 4, replace=False)
            #sample_source_pts = source_pts[sample_idx, :]
            #sample_target_pts = target_pts[sample_idx, :]

            #M = cv2.getPerspectiveTransform(sample_source_pts, sample_target_pts)
            #dst_img = cv2.warpPerspective(img, M, (300, 300))

            #H = cv2.findHomography(source_pts, target_pts, cv2.CV_RANSAC)
            #dst_img = cv2.warpPerspective(img, H, (300, 300))

            aligned_img = cv2.cvtColor(ioimg.imread(temp_file), cv2.COLOR_RGB2BGR)

            ret, buf = cv2.imencode('.jpg', aligned_img)
            raw_dst_img = buf.tobytes()

            chunk_size = 1024 * 64

            yield FaceAlignmentResponse(header=FaceAlignmentResponseHeader())

            for i in range(0, len(raw_dst_img), chunk_size):
                yield FaceAlignmentResponse(image_chunk=ImageRGB(content=raw_dst_img[i:i + chunk_size]))

        os.remove(temp_file)


def serve(max_workers=10, port=50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    services.grpc.face_alignment_pb2_grpc.add_FaceAlignmentServicer_to_server(
        FaceAlignmentServicer(), server)
    server.add_insecure_port('[::]:%d' % port)
    return server


@methods.add
async def ping():
    return 'pong'


@methods.add
async def align_face(**kwargs):
    image = kwargs.get("image", None)
    algorithm = kwargs.get("algorithm", "dlib_cnn")

    if image is None:
        raise InvalidParams("image is required")

    binary_image = base64.b64decode(image)
    img_data = io.BytesIO(binary_image)
    img = ioimg.imread(img_data)

    face_images = []

    return {'aligned_faces': face_images}


async def handle(request):
    request = await request.text()
    response = await methods.dispatch(request)
    if response.is_notification:
        return web.Response()
    else:
        return web.json_response(response, status=response.http_status)


if __name__ == '__main__':
    parser = services.common_parser(__file__)
    args = parser.parse_args(sys.argv[1:])
    serve_args = {}
    services.main_loop(serve, serve_args, handle, args)
