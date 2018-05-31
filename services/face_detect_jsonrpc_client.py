import jsonrpcclient
import sys
import os
import argparse
import base64
import yaml

from subprocess import Popen, PIPE

from services import registry


SERVER_PORT=8080

def main():
    script_name = sys.argv[0]
    parser = argparse.ArgumentParser(prog=script_name)
    server_name = "_".join(os.path.splitext(os.path.basename(script_name))[0].split('_')[:2]) + "_server"
    default_endpoint = "http://127.0.0.1:{}".format(registry[server_name]['jsonrpc'])
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

    with open(args.image, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('ascii')

    endpoint = args.endpoint
    params = {'algorithm': args.algorithm, "image": img_base64}
    if args.snet:
        endpoint, job_address, job_signature = snet_setup()
        params['job_address'] = job_address
        params['job_signature'] = job_signature

    response = jsonrpcclient.request(endpoint, "find_face", **params)

    if args.out_image:
        print("Rendering bounding box and saving to {}".format(args.out_image))
        import cv2
        image = cv2.imread(args.image)
        for d in response['faces']:
            cv2.rectangle(image, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 255, 0), 2)
        cv2.imwrite(args.out_image, image)



def snet_setup():
    service_name = "face_detect"
    print("Get {} service details from SingularityNET".format(service_name))

    p = Popen("snet registry query {}".format(service_name).split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    agent_address = yaml.load(output)['record']['agent']

    print("  - agent address is {}".format(agent_address))

    endpoint_cmd_str = "snet contract Agent --at {} endpoint".format(agent_address)
    p = Popen(endpoint_cmd_str.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    endpoint = yaml.load(output)

    print("  - agent endpoint is {}".format(endpoint))

    print("Funding job on SingularityNET")
    job_cmd_str = "snet agent --at {} create-jobs --number 1 --max-price 100 --funded --signed --no-confirm".format(agent_address)
    p = Popen(job_cmd_str.split(" "), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    jobs = yaml.load(output)['jobs']
    job = jobs[0]
    job_address = job['job_address']
    job_signature = job['job_signature']

    print("  - job funded and at address {}".format(job_address))

    return endpoint, job_address, job_signature

if __name__ == '__main__':
    main()



