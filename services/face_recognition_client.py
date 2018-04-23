import grpc
import sys
import numpy as np

import services.grpc.face_recognition_pb2
import services.grpc.face_recognition_pb2_grpc
from services.grpc.face_recognition_pb2 import FaceRecognitionRequest, FaceRecognitionResponse, FaceRecognitionHeader
from services.grpc.face_common_pb2 import ImageRGB, FaceDetections, BoundingBox

from tests.test_images import pre_calculated_faces

def read_in_chunks(filename, face_bboxes, model="68", chunk_size=1024*64):
    bboxes = []
    for bbox in face_bboxes:
        bboxes.append(BoundingBox(**bbox))
    header = FaceRecognitionHeader(faces=bboxes)
    flm = FaceRecognitionRequest(header=header)
    yield flm

    with open(filename, 'rb') as infile:
        while True:
            chunk = infile.read(chunk_size)
            if chunk:
                yield FaceRecognitionRequest(image_chunk=ImageRGB(content=chunk))
            else:
                # The chunk was empty, which means we're at the end
                # of the file
                return


def recognise_face(stub, image_fn, face_bboxes):
    img_iterator = read_in_chunks(image_fn, face_bboxes)
    face_identities = stub.RecogniseFace(img_iterator)
    return face_identities


def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = services.grpc.face_recognition_pb2_grpc.FaceRecognitionStub(channel)

    bbox = pre_calculated_faces['adele_2016.jpg'][0]
    result = recognise_face(stub, r'tests\test_images\adele_2016.jpg', [bbox])

    for identity in result.identities:
        identity_vector = np.array([x for x in identity.identity])
        print(identity_vector)



if __name__ == '__main__':
    run()