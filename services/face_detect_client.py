import grpc

import services.grpc.face_detect_pb2
import services.grpc.face_detect_pb2_grpc
from services.grpc.face_detect_pb2 import ImageUploadRequest

def read_in_chunks(filename, chunk_size=1024*64):
    with open(filename, 'rb') as infile:
        while True:
            chunk = infile.read(chunk_size)
            if chunk:
                yield ImageUploadRequest(content=chunk, image_type=filename)
            else:
                # The chunk was empty, which means we're at the end
                # of the file
                return

def find_faces(stub):
    img_iterator = read_in_chunks("headshot.jpg")
    faces = stub.FindFace(img_iterator)
    print(faces)

def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = services.grpc.face_detect_pb2_grpc.FaceDetectStub(channel)
    print("-------------- GetFeature --------------")
    find_faces(stub)


if __name__ == '__main__':
    run()