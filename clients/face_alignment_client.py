import grpc
import sys
import numpy as np

import services.grpc.face_alignment_pb2
import services.grpc.face_alignment_pb2_grpc
from services.grpc.face_common_pb2 import ImageRGB, BoundingBox, Point2D, FaceLandmarks

from tests.test_images import pre_calculated_faces


def make_request(filename, source_bboxes):
    boxes = [BoundingBox(**b) for b in source_bboxes]
    #source_p2d = [Point2D(x=int(p[0]), y=int(p[1])) for p in source_pts]
    #target_p2d = [Point2D(x=int(p[0]), y=int(p[1])) for p in target_pts]
    #source = FaceLandmarks(point=source_p2d)
    #target = FaceLandmarks(point=target_p2d)
    header = services.grpc.face_alignment_pb2.FaceAlignmentHeader(
        source_bboxes=boxes,
    )
    
    with open(filename, 'rb') as infile:
        chunk = infile.read()
        return services.grpc.face_alignment_pb2.FaceAlignmentRequest(header=header, image_chunk=ImageRGB(content=chunk))


def align_face(stub, image_fn, source_bboxes):
    request = make_request(image_fn, source_bboxes)
    images = []
    
    result = stub.AlignFace(request)
    
    for i in result.image_chunk:
        images.append(bytes(i.content))
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