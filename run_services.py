import pathlib
import subprocess
import signal
import time
import sys
import os
import argparse

from services import registry


def main():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument("--daemon-config-path",
                        help="Path to directory of daemon configuration files, without config it won't be started",
                        required=False
                        )
    args = parser.parse_args(sys.argv[1:])

    service_processes = []

    def handle_signal(signum, frame):
        for service, p, snetd_conf, snetd_p in service_processes:
            if sys.platform.startswith('win'):
                p.kill()
                if snetd_p:
                    snetd_p.kill()
            else:
                p.send_signal(signum)
                if snetd_p:
                    snetd_p.send_signal(signum)
        for service, p, snetd_conf, snetd_p in service_processes:
            p.wait()
            if snetd_p:
                snetd_p.wait()
        exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    root_path = pathlib.Path(__file__).absolute().parent

    service_modules = [
        'services.face_detect_server', 'services.face_landmarks_server',
        'services.face_align_server', 'services.face_identity_server'
    ]

    service_processes = start_face_services(root_path, service_modules, args.daemon_config_path)

    while True:
        for i, (service_module, service_p, snetd_conf, snetd_p) in enumerate(service_processes):
            if service_p.poll() is not None:
                service_processes[i] = start_face_services(root_path, [service_module], None)[1]
            if snetd_p is not None and snetd_p.poll() is not None:
                service_processes[i][3] = start_snetd(root_path, snetd_conf)
        time.sleep(5)


def start_snetd(cwd, daemon_config_path=None):
    cmd = ["snetd"]
    if daemon_config_path is not None:
        cmd.extend(["--config", str(daemon_config_path)])
        return subprocess.Popen(cmd, cwd=str(cwd))
    return None


def start_face_services(cwd, service_modules, daemon_config_path):
    services = []

    for i, service_module in enumerate(service_modules):
        server_name = service_module.split('.')[-1]

        grpc_port = registry[server_name]["grpc"]
        
        print("Launching", service_module, "on ports", str(registry[server_name]))
        snetd_p = None
        snetd_config = None
        if daemon_config_path:
            sub = "_server"
            server_name_chop = server_name[:-len(sub)] if server_name.endswith(sub) else server_name
            snetd_config = pathlib.Path(daemon_config_path) / ('snetd_' + server_name_chop + '_config.json')
            snetd_p = start_snetd(str(cwd), daemon_config_path=snetd_config)

        services.append([
            service_module,
            subprocess.Popen([
                sys.executable, "-m", service_module,
                '--grpc-port', str(grpc_port),
            ], cwd=str(cwd)),
            snetd_config,
            snetd_p
        ])

    return services


def restart_service(service_p):
    pass


if __name__ == "__main__":
    main()