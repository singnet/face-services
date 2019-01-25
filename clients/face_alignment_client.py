import grpc
import sys
import numpy as np

import services.grpc.face_alignment_pb2
import services.grpc.face_alignment_pb2_grpc
from services.grpc.face_common_pb2 import ImageRGB, BoundingBox, Point2D, FaceLandmarks

from tests.test_images import pre_calculated_faces


def read_in_chunks(filename, source_bboxes, chunk_size=1024*64):
    boxes = [BoundingBox(**b) for b in source_bboxes]
    #source_p2d = [Point2D(x=int(p[0]), y=int(p[1])) for p in source_pts]
    #target_p2d = [Point2D(x=int(p[0]), y=int(p[1])) for p in target_pts]
    #source = FaceLandmarks(point=source_p2d)
    #target = FaceLandmarks(point=target_p2d)
    header = services.grpc.face_alignment_pb2.FaceAlignmentHeader(
        source_bboxes=boxes,
    )
    yield services.grpc.face_alignment_pb2.FaceAlignmentRequest(header=header)

    with open(filename, 'rb') as infile:
        while True:
            chunk = infile.read(chunk_size)
            if chunk:
                yield services.grpc.face_alignment_pb2.FaceAlignmentRequest(image_chunk=ImageRGB(content=chunk))
            else:
                # The chunk was empty, which means we're at the end
                # of the file
                return


def align_face(stub, image_fn, source_bboxes):
    img_iterator = read_in_chunks(image_fn, source_bboxes)
    images = []
    image_data = None
    for data in stub.AlignFace(img_iterator):
        if data.HasField("header"):
            if image_data is not None:
                images.append(image_data)
            image_data = bytearray()
        else:
            image_data.extend(data.image_chunk.content)
    if image_data is not None:
        images.append(image_data)
    return images


def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = services.grpc.face_alignment_pb2_grpc.FaceAlignmentStub(channel)
    #source_keypoints = np.array([
    #    [341, 186], [277, 181], [185, 176], [223, 178], [253, 236]], dtype='float32')
    #target_keypoints = np.array([
    #    [0.8595674595992, 0.2134981538014],
    #    [0.6460604764104, 0.2289674387677],
    #    [0.1205750620789, 0.2137274526848],
    #    [0.3340850613712, 0.2290642403242],
    #    [0.4901123135679, 0.6277975316475]], dtype='float32')
    img_list_data = align_face(stub, r'tests\test_images\adele_2016.jpg',
                       #source_pts=source_keypoints,
                       source_bboxes=pre_calculated_faces['adele_2016.jpg'],
                       #target_pts=target_keypoints
                       )
    for i in range(0, len(pre_calculated_faces['adele_2016.jpg'])):
        with open('test_align_%d.jpg' % i, 'wb') as f:
            f.write(img_list_data[i])


if __name__ == '__main__':
    run()