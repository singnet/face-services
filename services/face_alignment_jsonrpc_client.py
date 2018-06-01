import jsonrpcclient
import sys
import os
import argparse
import base64

from services import registry
from faceutils import snet_setup


def main():
    script_name = sys.argv[0]
    parser = argparse.ArgumentParser(prog=script_name)
    server_name = "_".join(os.path.splitext(os.path.basename(script_name))[0].split('_')[:2]) + "_server"
    default_endpoint = "http://127.0.0.1:{}".format(registry[server_name]['jsonrpc'])
    parser.add_argument("--endpoint", help="jsonrpc server to connect to", default=default_endpoint,
                        type=str, required=False)
    parser.add_argument("--snet", help="call service on SingularityNet - requires configured snet CLI",
                        action='store_true')
    parser.add_argument("--image", help="path to image to apply face alignment on",
                        type=str, required=True)
    parser.add_argument("--out-image", help="Decode and save aligned imagery.",
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

    params = {"image": img_base64, "source_bboxes": bboxes}
    if args.snet:
        endpoint, job_address, job_signature = snet_setup(service_name="face_alignment")
        params['job_address'] = job_address
        params['job_signature'] = job_signature

    response = jsonrpcclient.request(endpoint, "align_face", **params)

    if args.out_image:
        import io
        import skimage.io as ioimg

        print("Saving aligned faces with suffix ...{}".format(args.out_image))

        for idx, result in enumerate(response['aligned_faces']):
            data = base64.b64decode(result)
            img_data = io.BytesIO(data)
            img = ioimg.imread(img_data)
            ioimg.imsave("aligned_face_" + str(idx) + "_" + args.out_image, img)


if __name__ == '__main__':
    main()



