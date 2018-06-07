import jsonrpcclient
import sys
import os
import argparse
import base64

from services import registry
from .snet import snet_setup


def main():
    script_name = sys.argv[0]
    parser = argparse.ArgumentParser(prog=script_name)
    server_name = "_".join(os.path.splitext(os.path.basename(script_name))[0].split('_')[:2]) + "_server"
    default_endpoint = "http://127.0.0.1:{}".format(registry[server_name]['jsonrpc'])
    parser.add_argument("--endpoint", help="jsonrpc server to connect to", default=default_endpoint,
                        type=str, required=False)
    parser.add_argument("--snet", help="call service on SingularityNet - requires configured snet CLI",
                        action='store_true')
    parser.add_argument("--image", help="path to image to apply face landmark prediction on",
                        type=str, required=True)
    parser.add_argument("--model", help="face landmark algorithm to request",
                        type=str, action='store',
                        choices=['5', '68'])
    parser.add_argument("--out-image", help="Render landmarks on image and save",
                        type=str, required=False)
    parser.add_argument("--face-bb", help='Specify face bounding box in "x,y,w,h" format',
                        type=str, required=True, action='append')
    args = parser.parse_args(sys.argv[1:])

    with open(args.image, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('ascii')

    endpoint = args.endpoint
    bboxes = []
    for b in args.face_bb:
        b = [int(x) for x in b.split(',')]
        assert len(b) == 4
        bboxes.append(dict(x=b[0], y=b[1], w=b[2], h=b[3]))

    params = {'model': args.model, "image": img_base64, "face_bboxes": bboxes}
    if args.snet:
        endpoint, job_address, job_signature = snet_setup(service_name="face_landmarks")
        params['job_address'] = job_address
        params['job_signature'] = job_signature

    response = jsonrpcclient.request(endpoint, "get_landmarks", **params)

    if args.out_image:
        import cv2
        import numpy as np
        print("Rendering landmarks and saving to {}".format(args.out_image))
        image = cv2.imread(args.image)
        for l in response['landmarks']:
            landmarks = np.matrix([[p['x'], p['y']] for p in l['points']])
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



