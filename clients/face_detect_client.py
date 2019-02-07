import grpc
import sys

import services.grpc.face_detect_pb2
import services.grpc.face_detect_pb2_grpc
from services.grpc.face_common_pb2 import ImageRGB

def make_request(filename):
    with open(filename, 'rb') as infile:
        data = infile.read()
        return ImageRGB(content=data)

def find_faces(stub, image_fn):
    return stub.FindFace(make_request(image_fn))

def run(image_fn):
    channel = grpc.insecure_channel('localhost:50051')
    stub = services.grpc.face_detect_pb2_grpc.FaceDetectStub(channel)
    print("-------------- FindFaces --------------")
    faces = find_faces(stub, image_fn)
    print(faces)

if __name__ == '__main__':
    if len(sys.argv) == 2:
        run(sys.argv[1])
    else:
        print("Usage: python %s image.jpg" % __file__)