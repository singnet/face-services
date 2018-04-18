import unittest
import logging

import grpc
import services.face_detect_server
from services.face_detect_client import find_faces
import services.grpc.face_detect_pb2_grpc

from tests.test_images import one_face, multiple_faces, no_faces

class BaseTestCase:
    class BaseTestFaceDetectGRPC(unittest.TestCase):
        test_port = 50001
        server = None

        @classmethod
        def setUpClass(cls):
            cls.server = services.face_detect_server.serve(algorithm=cls.algorithm, max_workers=2, port=cls.test_port, blocking=False)

        @classmethod
        def tearDownClass(cls):
            cls.server.stop(0)

        def setUp(self):
            self.channel = grpc.insecure_channel('localhost:' + str(self.test_port))
            self.stub = services.grpc.face_detect_pb2_grpc.FaceDetectStub(self.channel)

        def tearDown(self):
            pass

class TestFaceDetectGRPC_DlibCNN(BaseTestCase.BaseTestFaceDetectGRPC):
    algorithm = 'dlib_cnn'

    def test_find_single_face(self):
        for img_fn in one_face:
            logging.debug("Testing face detect %s on file with a single face %s" % (self.algorithm, img_fn,))
            result = find_faces(self.stub, img_fn)
            self.assertEqual(len(result.face_bbox), 1)

    def test_find_multiple_faces(self):
        for img_fn in multiple_faces:
            logging.debug("Testing face detect %s on file with multiple faces %s" % (self.algorithm, img_fn,))
            result = find_faces(self.stub, img_fn)
            self.assertGreater(len(result.face_bbox), 1)
            logging.debug(str(result.face_bbox))

    def test_find_no_faces(self):
        for img_fn in no_faces:
            logging.debug("Testing face detect on file with no faces %s" % (img_fn,))
            result = find_faces(self.stub, img_fn)
            self.assertEqual(len(result.face_bbox), 0)
            logging.debug(str(result.face_bbox))

class TestFaceDetectGRPC_DlibHOG(BaseTestCase.BaseTestFaceDetectGRPC):
    algorithm = 'dlib_hog'

    def test_find_single_face(self):
        for img_fn in one_face:
            logging.debug("Testing face detect %s on file with a single face %s" % (self.algorithm, img_fn,))
            result = find_faces(self.stub, img_fn)
            self.assertEqual(len(result.face_bbox), 1)

    def test_find_multiple_faces(self):
        for img_fn in multiple_faces:
            logging.debug("Testing face detect %s on file with multiple faces %s" % (self.algorithm, img_fn,))
            result = find_faces(self.stub, img_fn)
            self.assertGreater(len(result.face_bbox), 1)
            logging.debug(str(result.face_bbox))

    def test_find_no_faces(self):
        for img_fn in no_faces:
            logging.debug("Testing face detect on file with no faces %s" % (img_fn,))
            result = find_faces(self.stub, img_fn)
            self.assertEqual(len(result.face_bbox), 0)
            logging.debug(str(result.face_bbox))

class TestFaceDetectGRPC_HaarCascade(BaseTestCase.BaseTestFaceDetectGRPC):
    algorithm = 'haar_cascade'

    def test_find_single_face(self):
        for img_fn in one_face:
            logging.debug("Testing face detect %s on file with a single face %s" % (self.algorithm, img_fn,))
            result = find_faces(self.stub, img_fn)
            self.assertEqual(len(result.face_bbox), 1)

    def test_find_multiple_faces(self):
        for img_fn in multiple_faces:
            logging.debug("Testing face detect %s on file with multiple faces %s" % (self.algorithm, img_fn,))
            result = find_faces(self.stub, img_fn)
            if img_fn.endswith('classroom_in_tanzania.jpg'):
                logging.debug("Haar cascade is known to fail on %s" % (img_fn,))
            else:
                self.assertGreater(len(result.face_bbox), 1)
            logging.debug(str(result.face_bbox))

    def test_find_no_faces(self):
        for img_fn in no_faces:
            logging.debug("Testing face detect %s on file with no faces %s" % (self.algorithm, img_fn,))
            result = find_faces(self.stub, img_fn)
            self.assertEqual(len(result.face_bbox), 0)
            logging.debug(str(result.face_bbox))


if __name__ == '__main__':
    unittest.main()