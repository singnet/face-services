import argparse
import os.path

from services import registry
import time


def common_parser(script_name):
    parser = argparse.ArgumentParser(prog=script_name)
    server_name = os.path.splitext(os.path.basename(script_name))[0]
    parser.add_argument("--grpc-port", help="port to bind grpc service to", default=registry[server_name]['grpc'], type=int, required=False)
    return parser


def main_loop(grpc_serve_function, grpc_args, args):
    server = grpc_serve_function(port=args.grpc_port, **grpc_args)
    server.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        server.stop(0)

