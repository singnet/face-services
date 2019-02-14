import grpc
import sys
import argparse
from getpass import getpass

import services.grpc.face_detect_pb2
import services.grpc.face_detect_pb2_grpc
from services.grpc.face_common_pb2 import ImageRGB

from snet_sdk import Snet

def make_request(filenamem, request_cls=None):
    if request_cls is None:
        request_cls = ImageRGB
    with open(filename, 'rb') as infile:
        data = infile.read()
        return request_cls(content=data)

def find_faces(stub, image_fn):
    return stub.FindFace(make_request(image_fn))

def run_snet(image_fn, private_key):
    snet = Snet(private_key=private_key)
    client = snet.client("snet", "face-detect")
    stub = client.grpc.face_detect_pb2_grpc.FaceDetectStub(client.grpc_channel)
    request = make_request(image_fn, client.grpc.translate_pb2.Request)
    return stub.FindFace(request)
    

def run_local(image_fn, endpoint):
    channel = grpc.insecure_channel(endpoint)
    stub = services.grpc.face_detect_pb2_grpc.FaceDetectStub(channel)
    print("-------------- FindFaces --------------")
    return find_faces(stub, image_fn)
    

def main():
    script_name = sys.argv[0]
    parser = argparse.ArgumentParser(prog=script_name)
    subcommand = parser.add_subparsers(dest='subcommand')

    default_endpoint = 'http://localhost:50051'
    parser.add_argument("--endpoint", help="jsonrpc server to connect to", default=default_endpoint,
                        type=str, required=False)
    parser.add_argument("--snet", help="call service on SingularityNet - requires configured snet CLI",
                        action='store_true')
    parser.add_argument("--image", help="path to image to apply face detection on",
                        type=str, required=True)
    parser.add_argument("--algorithm", help="face detection algorithm to request",
                        type=str, default="dlib_cnn", action='store',
                        choices=['dlib_cnn','dlib_hog','haar_cascade'])
    parser.add_argument("--out-image", help="Render bounding box on image and save",
                        type=str, required=False)
    args = parser.parse_args(sys.argv[1:])

    endpoint = args.endpoint
    if args.snet:
        private_key = getpass("Enter private key: ")
        response = run_snet(args.image, private_key)
    else:
        response = run_local(args.image, args.endpoint)
    print(response)

    if args.out_image:
        print("Rendering bounding box and saving to {}".format(args.out_image))
        import cv2
        image = cv2.imread(args.image)
        for d in response['faces']:
            cv2.rectangle(image, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 255, 0), 2)
        cv2.imwrite(args.out_image, image)

if __name__ == '__main__':
    main()