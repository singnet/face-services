import pathlib
import subprocess
import signal
import time
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(prog=__file__)
    #parser.add_argument("--daemon-config-path", help="Path to daemon configuration file", required=False)
    args = parser.parse_args(sys.argv[1:])

    service_processes = []

    def handle_signal(signum, frame):
        for p in service_processes:
            p.send_signal(signum)
        for p in service_processes:
            p.wait()
        exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    root_path = pathlib.Path(__file__).absolute().parent

    service_p = start_face_services(root_path)

    while True:
        for service_p in service_processes:
            if service_p.poll() is not None:
                service_p = restart_service(service_p)
        time.sleep(5)


def start_face_services(cwd):
    services = []
    #p = cwd / 'services'
    #for service_module in p.glob('face_*_server.py'):

    service_modules = [
        'services.face_detect_server', 'services.face_landmarks_server',
        'services.face_alignment_server', 'services.face_recognition_server'
    ]

    for i, service_module in enumerate(service_modules):
        print(service_module)
        services.append(
            subprocess.Popen(
                ["python", "-m", service_module, '--port', str(50051 + i)], cwd=cwd
            )
        )
        print("launched")

    return services

def restart_service(service_p):
    pass

if __name__ == "__main__":
    main()