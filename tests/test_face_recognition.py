import unittest
import logging
import os.path
import cv2
import dlib

import grpc
import services.face_recognition_server
from services.face_recognition_client import recognise_face
import services.grpc.face_recognition_pb2_grpc
import numpy as np

from tests.test_images import one_face, multiple_faces, no_faces, pre_calculated_faces

from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop
from aiohttp import web


class TestFaceRecognitionJSONRPC(AioHTTPTestCase):

    async def get_application(self):
        app = web.Application()
        app.router.add_post('/', services.face_recognition_server.handle)
        return app

    @unittest_run_loop
    async def test_recognise_face(self):
        import base64
        img_fn = one_face[0]
        with open(img_fn, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('ascii')

        rpc_dict = {
            "jsonrpc": "2.0",
            "method": "recognise_face",
            "id": "1",
            "params": {
                "faces": pre_calculated_faces[os.path.basename(img_fn)],
                "image": img_base64
            }
        }

        resp = await self.client.post('/', json=rpc_dict)
        assert resp.status == 200
        json = await resp.json()
        assert "face_identities" in json['result']
        assert len(json['result']['face_identities']) == 1
        assert len(json['result']['face_identities'][0]) == 128


class TestFaceRecognitionGRPC(unittest.TestCase):
    test_port = 6008
    server = None

    @classmethod
    def setUpClass(cls):
        cls.server = services.face_recognition_server.serve(max_workers=2, port=cls.test_port)
        cls.server.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop(0)

    def setUp(self):
        self.channel = grpc.insecure_channel('localhost:' + str(self.test_port))
        self.stub = services.grpc.face_recognition_pb2_grpc.FaceRecognitionStub(self.channel)

    def tearDown(self):
        pass

    def test_recognise_single_face(self):
        for img_fn in one_face:
            bboxes = pre_calculated_faces[os.path.basename(img_fn)]
            logging.debug(
                "Testing face recognition on file with a single face %s" % (img_fn,))
            result = recognise_face(self.stub, img_fn, bboxes)
            self.assertEqual(len(result.identities), len(bboxes))

    def test_recognise_multiple_faces(self):
        for img_fn in multiple_faces:
            bboxes = pre_calculated_faces[os.path.basename(img_fn)]
            logging.debug(
                "Testing face recognition on file with multiple faces %s" % (img_fn,))
            result = recognise_face(self.stub, img_fn, bboxes)
            self.assertEqual(len(result.identities), len(bboxes))

    def test_recognise_no_faces(self):
        for img_fn in no_faces:
            # When there is no face, then this checks things don't explode when we give it a face bbox with no face
            bboxes = list(pre_calculated_faces.values())[0]
            logging.debug("Testing face recognition on file with no faces %s" % (img_fn,))
            result = recognise_face(self.stub, img_fn, bboxes)
            # Should still have the right number of responses, even if they are meaningless
            self.assertEqual(len(result.identities), len(bboxes))


if __name__ == '__main__':
    unittest.main()