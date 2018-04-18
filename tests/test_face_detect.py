import unittest
import logging

import grpc
import services.face_detect_server
from services.face_detect_client import find_faces
import services.grpc.face_detect_pb2_grpc

from tests.test_images import one_face, multiple_faces, no_faces

class TestFaceDetectGRPCService(unittest.TestCase):
    test_port = 50051
    server = None

    @classmethod
    def setUpClass(cls):
        cls.server = services.face_detect_server.serve(port=cls.test_port, blocking=False)

    @classmethod
    def tearDownClass(cls):
        cls.server.stop(0)

    def setUp(self):
        self.channel = grpc.insecure_channel('localhost:' + str(self.test_port))
        self.stub = services.grpc.face_detect_pb2_grpc.FaceDetectStub(self.channel)

    def tearDown(self):
        pass

    def test_find_single_face(self):
        for img_fn in one_face:
            logging.debug("Testing face detect on file with single face %s" % (img_fn,))
            find_faces(self.stub, img_fn)

    def test_find_multiple_faces(self):
        for img_fn in multiple_faces:
            logging.debug("Testing face detect on file with multiple face %s" % (img_fn,))
            find_faces(self.stub, img_fn)

    def test_find_no_faces(self):
        for img_fn in no_faces:
            logging.debug("Testing face detect on file with no face %s" % (img_fn,))
            find_faces(self.stub, img_fn)

if __name__ == '__main__':
    unittest.main()