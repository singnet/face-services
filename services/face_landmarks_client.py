import grpc
import sys

import services.grpc.face_landmarks_pb2
import services.grpc.face_landmarks_pb2_grpc
from services.grpc.face_landmarks_pb2 import FaceLandmarkHeader, FaceLandmarkRequest, Empty
from services.grpc.face_common_pb2 import ImageUploadRequest, FaceDetections, BoundingBox

def read_in_chunks(filename, model="68", chunk_size=1024*64):
    # This bounding box currently matches a local test image, remove once landmarks server knows how to query upstream
    # service
    face_bbox = BoundingBox(**{
        'x': 711,
        'y': 525,
        'w': 507,
        'h': 557
    })
    header = FaceLandmarkHeader(landmark_model=model, faces=FaceDetections(face_bbox=[face_bbox]))
    flm = FaceLandmarkRequest(header=header)
    yield flm

    with open(filename, 'rb') as infile:
        while True:
            chunk = infile.read(chunk_size)
            if chunk:
                yield FaceLandmarkRequest(image_chunk=ImageUploadRequest(content=chunk))
            else:
                # The chunk was empty, which means we're at the end
                # of the file
                return

def get_face_landmarks(stub, image_fn):
    model = "68"
    img_iterator = read_in_chunks(image_fn, model=model)
    face_landmarks = stub.GetLandmarks(img_iterator)

    models = stub.GetLandmarkModels(Empty())
    landmark_names = None
    for m in models.model:
        if m.landmark_model == model:
            landmark_names = m.landmark_description

    if landmark_names is not None:
        for i, lm in enumerate(zip(landmark_names, face_landmarks.point)):
            print("%d - %s: %d, %d" % (i, lm[0], lm[1].x, lm[1].y))
    else:
        print("No descriptions")
        print(face_landmarks)

def run(image_fn):
    channel = grpc.insecure_channel('localhost:50051')
    stub = services.grpc.face_landmarks_pb2_grpc.FaceLandmarkStub(channel)
    print("-------------- GetLandmarks --------------")
    get_face_landmarks(stub, image_fn)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        print("Usage: python %s image.jpg" % __file__)