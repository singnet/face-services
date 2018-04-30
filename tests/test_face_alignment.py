import unittest
import logging
import os.path
import cv2
import dlib

import grpc
import services.face_alignment_server
from services.face_alignment_client import align_face
import services.grpc.face_alignment_pb2_grpc

from tests.test_images import one_face, multiple_faces, no_faces, pre_calculated_faces

import numpy as np


class TestFaceAlignmentGRPC(unittest.TestCase):
    test_port = 50003
    server = None

    @classmethod
    def setUpClass(cls):
        cls.server = services.face_alignment_server.serve(max_workers=2, port=cls.test_port)
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop(0)

    def setUp(self):
        self.channel = grpc.insecure_channel('localhost:' + str(self.test_port))
        self.stub = services.grpc.face_alignment_pb2_grpc.FaceAlignmentStub(self.channel)

    def tearDown(self):
        pass

    def test_align_single_face(self):
        for img_fn in one_face:
            bboxes = pre_calculated_faces[os.path.basename(img_fn)]
            logging.debug(
                "Testing face alignment on file with a single face %s" % (img_fn,))
            result = align_face(self.stub, img_fn, bboxes)
            self.assertEqual(len(result), len(bboxes))

    def test_align_multiple_faces(self):
        for img_fn in multiple_faces:
            bboxes = pre_calculated_faces[os.path.basename(img_fn)]
            logging.debug(
                "Testing face alignment on file with multiple faces %s" % (img_fn,))
            result = align_face(self.stub, img_fn, source_bboxes=bboxes)
            self.assertEqual(len(result), len(bboxes))

    def test_align_no_faces(self):
        for img_fn in no_faces:
            # When there is no face, then this checks things don't explode when we give it a face bbox with no face
            bboxes = list(pre_calculated_faces.values())[0]
            logging.debug("Testing face alignment on file with no faces %s" % (img_fn,))
            result = align_face(self.stub, img_fn, source_bboxes=bboxes)
            # Should still have the right number of responses, even if they are meaningless
            self.assertEqual(len(result), len(bboxes))


if __name__ == '__main__':
    unittest.main()