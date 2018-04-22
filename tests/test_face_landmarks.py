import unittest
import logging
import os.path
import cv2
import dlib

import grpc
import services.face_landmarks_server
from services.face_landmarks_client import get_face_landmarks
import services.grpc.face_landmarks_pb2_grpc

from faceutils import render_face_landmarks_debug_image

from tests.test_images import one_face, multiple_faces, no_faces

pre_calculated_faces = {
    'classroom_in_tanzania.jpg': [
        {'x': 633, 'y': 245, 'w': 90, 'h': 90},
        {'x': 155, 'y': 216, 'w': 90, 'h': 89},
        {'x': 59, 'y': 36, 'w': 44, 'h': 43},
        {'x': 218, 'y': 55, 'w': 43, 'h': 43},
        {'x': 118, 'y': 101, 'w': 52, 'h': 52},
        {'x': 519, 'y': 196, 'w': 75, 'h': 75},
        {'x': 259, 'y': 331, 'w': 107, 'h': 107},
    ],
    'laos.jpg': [
        {'x': 22, 'y': 260, 'w': 268, 'h': 268},
        {'x': 538, 'y': 217, 'w': 223, 'h': 223},
    ],
    'adele_2016.jpg': [
        {'x': 142, 'y': 118, 'w': 223, 'h': 223},
    ],
    "elon_musk_2015.jpg": [
        {'x': 117, 'y': 142, 'w': 223, 'h': 223},
    ],
    "waaah.jpg": [
        {'x': 315, 'y': 142, 'w': 223, 'h': 223}
    ],
    "city_portrait.jpg": [
        {'x': 233, 'y': 172, 'w': 555, 'h': 554}
    ],
    "benedict_cumberbatch_2014.png": [
        {'x': 97, 'y': 118, 'w': 186, 'h': 186}
    ],
    "model_face.jpg": [
        {'x': 206, 'y': 280, 'w': 666, 'h': 666}
    ],
    "sophia.jpg": [
        {'x': 221, 'y': 77, 'w': 186, 'h': 186}
    ],
    "woman_portrait.jpg": [
        {'x': 85, 'y': 298, 'w': 958, 'h': 958}
    ]
}



class BaseTestCase:
    class BaseTestFaceLandmarksGRPC(unittest.TestCase):
        test_port = 50001
        server = None

        @classmethod
        def setUpClass(cls):
            cls.server = services.face_landmarks_server.serve(max_workers=2, port=cls.test_port, blocking=False)

        @classmethod
        def tearDownClass(cls):
            cls.server.stop(0)

        def setUp(self):
            self.channel = grpc.insecure_channel('localhost:' + str(self.test_port))
            self.stub = services.grpc.face_landmarks_pb2_grpc.FaceLandmarkStub(self.channel)

        def tearDown(self):
            pass

        def test_get_landmarks_single_face(self):
            for img_fn in one_face:
                bboxes = pre_calculated_faces[os.path.basename(img_fn)]
                logging.debug(
                    "Testing face landmark prediction %s on file with a single face %s" % (self.algorithm, img_fn,))
                result = get_face_landmarks(self.stub, img_fn, bboxes, model=self.algorithm)
                self.assertEqual(len(result.landmarked_faces), len(bboxes))
                self.assertEqual(len(result.landmarked_faces[0].point), int(self.algorithm))
                render_face_landmarks_debug_image(self, img_fn, result)

        def test_get_landmarks_multiple_faces(self):
            for img_fn in multiple_faces:
                bboxes = pre_calculated_faces[os.path.basename(img_fn)]
                logging.debug(
                    "Testing face landmark prediction %s on file with multiple faces %s" % (self.algorithm, img_fn,))
                result = get_face_landmarks(self.stub, img_fn, bboxes, model=self.algorithm)
                self.assertEqual(len(result.landmarked_faces), len(bboxes))
                for i in range(0, len(bboxes)):
                    self.assertEqual(len(result.landmarked_faces[i].point), int(self.algorithm))
                render_face_landmarks_debug_image(self, img_fn, result)

        def test_get_landmarks_no_faces(self):
            for img_fn in no_faces:
                # When there is no face, then this checks things don't explode when we give it a face bbox with no face
                bboxes = list(pre_calculated_faces.values())[0]
                logging.debug("Testing face detect on file with no faces %s" % (img_fn,))
                result = get_face_landmarks(self.stub, img_fn, bboxes, model=self.algorithm)
                # Should still have the right number of responses, even if they are meaningless
                self.assertEqual(len(result.landmarked_faces), len(bboxes))
                for i in range(0, len(bboxes)):
                    self.assertEqual(len(result.landmarked_faces[i].point), int(self.algorithm))
                render_face_landmarks_debug_image(self, img_fn, result)


class TestFaceLandmarksGRPC_Dlib68(BaseTestCase.BaseTestFaceLandmarksGRPC):
    algorithm = '68'


class TestFaceLandmarksGRPC_Dlib5(BaseTestCase.BaseTestFaceLandmarksGRPC):
    algorithm = '5'


if __name__ == '__main__':
    unittest.main()