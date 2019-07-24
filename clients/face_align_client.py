import grpc
import sys
import argparse
import os
from services import registry

from services.grpc.face_common_pb2 import ImageRGB, FaceDetections, BoundingBox
from services.grpc.face_alignment_pb2 import FaceAlignmentHeader, FaceAlignmentRequest, FaceAlignmentResponse
from services.grpc.face_alignment_pb2_grpc import FaceAlignmentStub

def make_request(filename, source_bboxes):
    boxes = [BoundingBox(**b) for b in source_bboxes]
    
    header = FaceAlignmentHeader(source_bboxes=boxes)
    
    with open(filename, 'rb') as infile:
        chunk = infile.read()
        return services.grpc.face_alignment_pb2.FaceAlignmentRequest(header=header, image_chunk=ImageRGB(content=chunk))


def align_face(stub, image_fn, source_bboxes):
    request = make_request(image_fn, source_bboxes)
    images = []
    
    result = stub.AlignFace(request)
    
    for i in result.image_chunk:
        images.append(bytes(i.content))
    return images


def run():
    img_list_data = align_face(stub, r'tests\test_images\adele_2016.jpg',
                       source_bboxes=pre_calculated_faces['adele_2016.jpg'],
                       )
    for i in range(0, len(pre_calculated_faces['adele_2016.jpg'])):
        with open('test_align_%d.jpg' % i, 'wb') as f:
            f.write(img_list_data[i])


def make_request(filename, face_bboxes, request_cls=None):
    if request_cls is None:
        request_cls = FaceAlignmentRequest
    bboxes = []
    for bbox in face_bboxes:
        bboxes.append(BoundingBox(**bbox))
    header = FaceAlignmentHeader(source_bboxes=bboxes)
    with open(filename, 'rb') as infile:
        chunk = infile.read()
        return request_cls(header=header, image_chunk=ImageRGB(content=chunk))

def run_snet(image_fn, face_bboxes, private_key):
    from snet_sdk import Snet
    snet = Snet(private_key=private_key)
    client = snet.client("snet", "face-align")
    stub = client.grpc.face_alignment_pb2_grpc.FaceAlignmentStub(client.grpc_channel)
    request = make_request(image_fn, client.grpc.face_alignment_pb2.Request)
    return stub.AlignFace(request)

def run_local(image_fn, face_bboxes, endpoint):
    channel = grpc.insecure_channel(endpoint)
    stub = FaceAlignmentStub(channel)
    print("-------------- AlignFace --------------")
    return align_face(stub, image_fn, face_bboxes)


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
        response = run_snet(args.image, bboxes, private_key)
    else:
        response = run_local(args.image, bboxes, args.endpoint)
    
    if args.out_image:
        import io
        import skimage.io as ioimg

        print("Saving aligned faces with prefix ...{}".format(args.out_image))

        for idx, result in enumerate(response):
            img_data = io.BytesIO(result)
            img = ioimg.imread(img_data)
            ioimg.imsave(args.out_image + "_aligned_face_" + str(idx) + ".jpg", img)


if __name__ == '__main__':
    main()
