import unittest
import logging
import os.path

import grpc

from services import registry

from services.grpc.face_recognition_pb2_grpc import FaceRecognitionStub
from services.grpc.face_recognition_pb2 import FaceRecognitionRequest, FaceRecognitionHeader
from services.grpc.face_common_pb2 import ImageRGB, FaceDetections, BoundingBox

from tests.test_images import one_face, multiple_faces, no_faces, pre_calculated_faces


class TestFaceRecognitionGRPC(unittest.TestCase):
    def setUp(self):
        service_name = "face_identity_server"
        port = registry[service_name]["grpc"]
        self.channel = grpc.insecure_channel('localhost:{}'.format(port))
        self.stub = FaceRecognitionStub(self.channel)

    def tearDown(self):
        pass

    def test_recognise_single_face(self):
        for img_fn in one_face:
            bboxes = pre_calculated_faces[os.path.basename(img_fn)]
            logging.debug("Testing face recognition on file with a single face %s" % (img_fn,))
            boxes = [BoundingBox(**b) for b in bboxes]
            header = FaceRecognitionHeader(faces=boxes)
            with open(img_fn, 'rb') as infile:
                chunk = infile.read()
                request = FaceRecognitionRequest(header=header, image_chunk=ImageRGB(content=chunk))
            result = self.stub.RecogniseFace(request)
            # print(np.array([f for f in result.identities[0].identity]))
            self.assertEqual(len(result.identities), len(bboxes))

    def test_recognise_multiple_faces(self):
        for img_fn in multiple_faces:
            bboxes = pre_calculated_faces[os.path.basename(img_fn)]
            logging.debug("Testing face recognition on file with multiple faces %s" % (img_fn,))
            boxes = [BoundingBox(**b) for b in bboxes]
            header = FaceRecognitionHeader(faces=boxes)
            with open(img_fn, 'rb') as infile:
                chunk = infile.read()
                request = FaceRecognitionRequest(header=header, image_chunk=ImageRGB(content=chunk))
            result = self.stub.RecogniseFace(request)
            # for i in range(0, len(bboxes)):
            #    print(repr(np.array([f for f in result.identities[i].identity])))
            self.assertEqual(len(result.identities), len(bboxes))

    def test_recognise_no_faces(self):
        for img_fn in no_faces:
            # When there is no face, then this checks things don't explode when we give it a face bbox with no face
            bboxes = list(pre_calculated_faces.values())[0]
            logging.debug("Testing face recognition on file with no faces %s" % (img_fn,))
            boxes = [BoundingBox(**b) for b in bboxes]
            header = FaceRecognitionHeader(faces=boxes)
            with open(img_fn, 'rb') as infile:
                chunk = infile.read()
                request = FaceRecognitionRequest(header=header, image_chunk=ImageRGB(content=chunk))
            result = self.stub.RecogniseFace(request)
            # Should still have the right number of responses, even if they are meaningless
            # print(np.array([f for f in result.identities[0].identity]))
            self.assertEqual(len(result.identities), len(bboxes))


if __name__ == '__main__':
    unittest.main()
