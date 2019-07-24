import grpc
import sys
import argparse
import os
from services import registry

import services.grpc.face_landmarks_pb2
import services.grpc.face_landmarks_pb2_grpc
from services.grpc.face_landmarks_pb2 import FaceLandmarkHeader, FaceLandmarkRequest, Empty
from services.grpc.face_common_pb2 import ImageRGB, FaceDetections, BoundingBox

def make_request(filename, face_bboxes, model="68", request_cls=None):
    if request_cls is None:
        request_cls = FaceLandmarkRequest
    bboxes = []
    for bbox in face_bboxes:
        bboxes.append(BoundingBox(**bbox))
    header = FaceLandmarkHeader(landmark_model=model, faces=FaceDetections(face_bbox=bboxes))
    with open(filename, 'rb') as infile:
        chunk = infile.read()
        return request_cls(header=header, image_chunk=ImageRGB(content=chunk))

def get_face_landmarks(stub, image_fn, face_bboxes, model="68"):
    img_iterator = make_request(image_fn, face_bboxes, model=model)
    face_landmarks = stub.GetLandmarks(img_iterator)
    return face_landmarks

def get_face_landmark_models(stub):
    return stub.GetLandmarkModels(Empty())

def get_landmark_names_for_model(stub, model):
    models = get_face_landmark_models(stub)
    landmark_names = None
    for m in models.model:
        if m.landmark_model == model:
            landmark_names = m.landmark_description
            break
    return landmark_names


def run_snet(image_fn, face_bboxes, model, private_key):
    from snet_sdk import Snet
    snet = Snet(private_key=private_key)
    client = snet.client("snet", "face-landmarks")
    stub = client.grpc.face_landmarks_pb2_grpc.FaceLandmarkStub(client.grpc_channel)
    landmark_names = get_landmark_names_for_model(stub, model)
    request = make_request(image_fn, client.grpc.face_landmarks_pb2.Request)
    return landmark_names, stub.GetLandmarks(request)

def run_local(image_fn, face_bboxes, model, endpoint):
    channel = grpc.insecure_channel(endpoint)
    stub = services.grpc.face_landmarks_pb2_grpc.FaceLandmarkStub(channel)
    print("-------------- GetLandmarks --------------")
    landmark_names = get_landmark_names_for_model(stub, model)
    return landmark_names, get_face_landmarks(stub, image_fn, face_bboxes, model)


def main():
    script_name = sys.argv[0]
    parser = argparse.ArgumentParser(prog=script_name)
    subcommand = parser.add_subparsers(dest='subcommand')

    server_name = "_".join(os.path.splitext(os.path.basename(script_name))[0].split('_')[:2]) + "_server"
    default_endpoint = "127.0.0.1:{}".format(registry[server_name]['grpc'])
    parser.add_argument("--endpoint", help="grpc server to connect to", default=default_endpoint,
                        type=str, required=False)
    parser.add_argument("--snet", help="call service on SingularityNet - requires configured snet CLI",
                        action='store_true')
    parser.add_argument("--image", help="path to image to apply face detection on",
                        type=str, required=True)
    parser.add_argument("--model", help="face landmark algorithm to request",
                        type=str, action='store',
                        choices=['5', '68'], default='68')
    parser.add_argument("--face-bb", help='Specify face bounding box in "x,y,w,h" format',
                        type=str, required=True, action='append')                        
    parser.add_argument("--out-image", help="Render bounding box on image and save",
                        type=str, required=False)
    args = parser.parse_args(sys.argv[1:])

    bboxes = []
    for b in args.face_bb:
        b = [int(x) for x in b.split(',')]
        assert len(b) == 4
        bboxes.append(dict(x=b[0], y=b[1], w=b[2], h=b[3]))
    
    endpoint = args.endpoint
    if args.snet:
        private_key = getpass("Enter private key: ")
        landmark_names, response = run_snet(args.image, bboxes, args.model, private_key)
    else:
        landmark_names, response = run_local(args.image, bboxes, args.model, args.endpoint)
    
    for fl in response.landmarked_faces:
        if landmark_names is not None:
            for i, lm in enumerate(zip(landmark_names, fl.point)):
                print("%d - %s: %d, %d" % (i, lm[0], lm[1].x, lm[1].y))
        else:
            print("No descriptions")
            print(fl)

    if args.out_image:
        import cv2
        import numpy as np
        print("Rendering landmarks and saving to {}".format(args.out_image))
        image = cv2.imread(args.image)
        
        for l, d in zip(response.landmarked_faces, bboxes):
            
            cv2.rectangle(image, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 255, 0), 2)

            landmarks = np.matrix([[p.x, p.y] for p in l.point])
            for idx, point in enumerate(landmarks):
                pos = (point[0, 0], point[0, 1])

                # annotate the positions
                cv2.putText(image, str(idx), pos,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4,
                            color=(0, 0, 255))

                # draw points on the landmark positions
                cv2.circle(image, pos, 3, color=(0, 255, 255))
        cv2.imwrite(args.out_image, image)

    

if __name__ == '__main__':
    main()
