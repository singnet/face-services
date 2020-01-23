import unittest
import logging
import os

import grpc

from services import registry

from services.grpc.face_common_pb2 import ImageRGB, BoundingBox
from services.grpc.face_alignment_pb2 import FaceAlignmentHeader, FaceAlignmentRequest
from services.grpc.face_alignment_pb2_grpc import FaceAlignmentStub

from tests.test_images import one_face, multiple_faces, no_faces, pre_calculated_faces


class TestFaceAlignmentGRPC(unittest.TestCase):
    def setUp(self):
        service_name = "face_align_server"
        port = registry[service_name]["grpc"]
        self.channel = grpc.insecure_channel('localhost:{}'.format(port))
        self.stub = FaceAlignmentStub(self.channel)

    def tearDown(self):
        pass

    def test_align_single_face(self):
        for img_fn in one_face:
            bboxes = pre_calculated_faces[os.path.basename(img_fn)]
            logging.debug("Testing face alignment on file with a single face %s" % (img_fn,))

            boxes = [BoundingBox(**b) for b in bboxes]
            header = FaceAlignmentHeader(source_bboxes=boxes)
            with open(img_fn, 'rb') as infile:
                chunk = infile.read()
                request = FaceAlignmentRequest(header=header, image_chunk=ImageRGB(content=chunk))
            result = self.stub.AlignFace(request)
            images = []
            for i in result.image_chunk:
                images.append(bytes(i.content))
            self.assertEqual(len(images), len(bboxes))

    def test_align_multiple_faces(self):
        for img_fn in multiple_faces:
            bboxes = pre_calculated_faces[os.path.basename(img_fn)]
            logging.debug("Testing face alignment on file with multiple faces %s" % (img_fn,))
            boxes = [BoundingBox(**b) for b in bboxes]
            header = FaceAlignmentHeader(source_bboxes=boxes)
            with open(img_fn, 'rb') as infile:
                chunk = infile.read()
                request = FaceAlignmentRequest(header=header, image_chunk=ImageRGB(content=chunk))
            result = self.stub.AlignFace(request)
            images = []
            for i in result.image_chunk:
                images.append(bytes(i.content))
            self.assertEqual(len(images), len(bboxes))

    def test_align_no_faces(self):
        for img_fn in no_faces:
            # When there is no face, then this checks things don't explode when we give it a face bbox with no face
            bboxes = list(pre_calculated_faces.values())[0]
            logging.debug("Testing face alignment on file with no faces %s" % (img_fn,))
            boxes = [BoundingBox(**b) for b in bboxes]
            header = FaceAlignmentHeader(source_bboxes=boxes)
            with open(img_fn, 'rb') as infile:
                chunk = infile.read()
                request = FaceAlignmentRequest(header=header, image_chunk=ImageRGB(content=chunk))
            result = self.stub.AlignFace(request)
            images = []
            for i in result.image_chunk:
                images.append(bytes(i.content))
            # Should still have the right number of responses, even if they are meaningless
            self.assertEqual(len(images), len(bboxes))


if __name__ == '__main__':
    unittest.main()
