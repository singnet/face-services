import unittest
import logging

import grpc

from services import registry

from services.grpc.face_detect_pb2_grpc import FaceDetectStub
from services.grpc.face_common_pb2 import ImageRGB

from faceutils import render_face_detect_debug_image
from tests.test_images import one_face, multiple_faces, no_faces

log = logging.getLogger("test.face_detect")


class BaseTestCase:
    class BaseTestFaceDetectGRPC(unittest.TestCase):
        def setUp(self):
            service_name = "face_detect_server"
            port = registry[service_name]["grpc"]
            self.channel = grpc.insecure_channel('localhost:{}'.format(port))
            self.stub = FaceDetectStub(self.channel)

        def tearDown(self):
            pass


class TestFaceDetectGRPC_DlibCNN(BaseTestCase.BaseTestFaceDetectGRPC):
    algorithm = 'dlib_cnn'
    test_port = 7001

    def test_find_single_face(self):
        for img_fn in one_face:
            log.debug("Testing face detect %s on file with a single face %s" % (self.algorithm, img_fn,))
            with open(img_fn, 'rb') as infile:
                data = infile.read()
                request = ImageRGB(content=data)
            result = self.stub.FindFace(request)
            log.debug(str(result.face_bbox))
            self.assertEqual(len(result.face_bbox), 1)

    def test_find_multiple_faces(self):
        for img_fn in multiple_faces:
            log.debug("Testing face detect %s on file with multiple faces %s" % (self.algorithm, img_fn,))
            with open(img_fn, 'rb') as infile:
                data = infile.read()
                request = ImageRGB(content=data)
            result = self.stub.FindFace(request)
            self.assertGreater(len(result.face_bbox), 1)
            log.debug(str(result.face_bbox))

    def test_find_no_faces(self):
        for img_fn in no_faces:
            log.debug("Testing face detect %s on file with no faces %s" % (self.algorithm, img_fn,))
            with open(img_fn, 'rb') as infile:
                data = infile.read()
                request = ImageRGB(content=data)
            result = self.stub.FindFace(request)
            self.assertEqual(len(result.face_bbox), 0)
            log.debug(str(result.face_bbox))


class TestFaceDetectGRPC_DlibHOG(BaseTestCase.BaseTestFaceDetectGRPC):
    algorithm = 'dlib_hog'
    test_port = 7001

    def test_find_single_face(self):
        for img_fn in one_face:
            log.debug("Testing face detect %s on file with a single face %s" % (self.algorithm, img_fn,))
            with open(img_fn, 'rb') as infile:
                data = infile.read()
                request = ImageRGB(content=data)
            result = self.stub.FindFace(request)
            log.debug("%s - %s - %s" % (self.algorithm, img_fn, str(result.face_bbox)))
            self.assertEqual(len(result.face_bbox), 1)
            render_face_detect_debug_image(self, img_fn, result)

    def test_find_multiple_faces(self):
        for img_fn in multiple_faces:
            log.debug("Testing face detect %s on file with multiple faces %s" % (self.algorithm, img_fn,))
            with open(img_fn, 'rb') as infile:
                data = infile.read()
                request = ImageRGB(content=data)
            result = self.stub.FindFace(request)
            self.assertGreater(len(result.face_bbox), 1)
            log.debug("%s - %s - %s" % (self.algorithm, img_fn, str(result.face_bbox)))
            render_face_detect_debug_image(self, img_fn, result)

    def test_find_no_faces(self):
        for img_fn in no_faces:
            log.debug("Testing face detect %s on file with no faces %s" % (self.algorithm, img_fn,))
            with open(img_fn, 'rb') as infile:
                data = infile.read()
                request = ImageRGB(content=data)
            result = self.stub.FindFace(request)
            self.assertEqual(len(result.face_bbox), 0)
            log.debug("%s - %s - %s" % (self.algorithm, img_fn, str(result.face_bbox)))
            render_face_detect_debug_image(self, img_fn, result)


class TestFaceDetectGRPC_HaarCascade(BaseTestCase.BaseTestFaceDetectGRPC):
    algorithm = 'haar_cascade'
    test_port = 7001

    def test_find_single_face(self):
        for img_fn in one_face:
            log.debug("Testing face detect %s on file with a single face %s" % (self.algorithm, img_fn,))
            with open(img_fn, 'rb') as infile:
                data = infile.read()
                request = ImageRGB(content=data)
            result = self.stub.FindFace(request)
            log.debug(str(result.face_bbox))
            self.assertEqual(len(result.face_bbox), 1)
            render_face_detect_debug_image(self, img_fn, result)

    def test_find_multiple_faces(self):
        for img_fn in multiple_faces:
            log.debug("Testing face detect %s on file with multiple faces %s" % (self.algorithm, img_fn,))
            with open(img_fn, 'rb') as infile:
                data = infile.read()
                request = ImageRGB(content=data)
            result = self.stub.FindFace(request)
            if img_fn.endswith('classroom_in_tanzania.jpg'):
                log.debug("Haar cascade is known to fail on %s" % (img_fn,))
            else:
                self.assertGreater(len(result.face_bbox), 1)
            log.debug(str(result.face_bbox))
            render_face_detect_debug_image(self, img_fn, result)

    def test_find_no_faces(self):
        for img_fn in no_faces:
            log.debug("Testing face detect %s on file with no faces %s" % (self.algorithm, img_fn,))
            with open(img_fn, 'rb') as infile:
                data = infile.read()
                request = ImageRGB(content=data)
            result = self.stub.FindFace(request)
            self.assertEqual(len(result.face_bbox), 0)
            log.debug(str(result.face_bbox))
            render_face_detect_debug_image(self, img_fn, result)


if __name__ == '__main__':
    unittest.main()
