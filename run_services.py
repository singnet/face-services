import pathlib
import subprocess
import signal
import time
import sys
import os
import argparse

from services import registry


def main():
    #parser = argparse.ArgumentParser(prog=__file__)
    #parser.add_argument("--daemon-config-path", help="Path to daemon configuration file", required=False)
    #args = parser.parse_args(sys.argv[1:])

    service_processes = []

    def handle_signal(signum, frame):
        for service, p in service_processes:
            if sys.platform.startswith('win'):
                p.kill()
            else:
                p.send_signal(signum)
        for service, p in service_processes:
            p.wait()
        exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    root_path = pathlib.Path(__file__).absolute().parent

    service_modules = [
        'services.face_detect_server', 'services.face_landmarks_server',
        'services.face_alignment_server', 'services.face_recognition_server'
    ]

    service_processes = start_face_services(root_path, service_modules)

    while True:
        for i, (service_module, service_p) in enumerate(service_processes):
            if service_p.poll() is not None:
                service_processes[i] = start_face_services(root_path, [service_module])[0]
        time.sleep(5)


def start_face_services(cwd, service_modules):
    services = []

    for i, service_module in enumerate(service_modules):
        server_name = service_module.split('.')[-1]
        grpc_port = registry[server_name]["grpc"]
        jsonrpc_port = registry[server_name]["jsonrpc"]

        print("Launching", service_module, "on ports", str(registry[server_name]))
        services.append((
            service_module,
            subprocess.Popen([
                "python", "-m", service_module,
                '--grpc-port', str(grpc_port),
                '--json-rpc-port', str(jsonrpc_port)
            ], cwd=cwd)
        ))

    return services


def restart_service(service_p):
    pass


if __name__ == "__main__":
    main()