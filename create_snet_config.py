import os
import sys

import json
import pathlib
import argparse
import getpass


from services import registry

config_template = {
    "DAEMON_LISTENING_PORT": None,
    "ETHEREUM_JSON_RPC_ENDPOINT": "https://kovan.infura.io/",
    "AGENT_CONTRACT_ADDRESS": None,
    "PASSTHROUGH_ENDPOINT": None,
    "PASSTHROUGH_ENABLED": True,
    "BLOCKCHAIN_ENABLED": True,
    "LOG_LEVEL": 10,
    "PRIVATE_KEY": None,
}

agent_contracts = {
    "kovan": {
        "face_detect_server": "0xA",
        "face_landmarks_server": "0xB",
        "face_alignment_server": "0xC",
        "face_recognition_server": "0xD",
    }
}

def make_config(service, snetd_port, local_jsonrpc_port, contract_address, private_key):
    service_config = dict(config_template)

    service_config['DAEMON_LISTENING_PORT'] = str(snetd_port)
    service_config['PASSTHROUGH_ENDPOINT'] = "http://localhost:" + str(local_jsonrpc_port)
    service_config['AGENT_CONTRACT_ADDRESS'] = contract_address
    service_config['PRIVATE_KEY'] = private_key

    config_file = pathlib.Path('config') / ('snetd_' + service + '_config.json')
    os.makedirs("config", exist_ok=True)
    with open(str(config_file), 'w') as f:
        json.dump(service_config, f, indent=2)

def main():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument("--network",
                        help="Which ethereum network to use",
                        default="kovan",
                        type=str
                        )
    parser.add_argument("--detect-address",
                        help="Agent contract address for face detection",
                        type=str
                        )
    parser.add_argument("--landmarks-address",
                        help="Agent contract address for face landmarks",
                        type=str
                        )
    parser.add_argument("--alignment-address",
                        help="Agent contract address for face alignment",
                        type=str
                        )
    parser.add_argument("--recognition-address",
                        help="Agent contract address for face recognition",
                        type=str
                        )
    args = parser.parse_args(sys.argv[1:])

    if args.network != 'kovan':
        raise Exception("Only kovan network currently supported")
    private_key = getpass.getpass('Private key used for creating agent contract:')

    service_to_arg = {
        'face_detect_server': 'detect_address',
        'face_landmarks_server': 'landmarks_address',
        'face_alignment_server': 'alignment_address',
        'face_recognition_server': 'recognition_address',
    }

    for service, local_ports in registry.items():
        manual_address = getattr(args, service_to_arg[service])
        if manual_address:
            agent_contracts[args.network][service] = manual_address
        make_config(service, local_ports['snetd'], local_ports['jsonrpc'], agent_contracts[args.network][service], private_key)

if __name__ == "__main__":
    main()