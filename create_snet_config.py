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
        "face_detect_server": "0x4cBe33Aa28eBBbFAa7d98Fa1c65af2FEf6885EF2",
        "face_landmarks_server": "0x88DeC961e30F973b6DeDbae35754a3c557380BEE",
        "face_alignment_server": "0xCB58410EE3B8E99ABd9774aB98951680E637b5F3",
        "face_recognition_server": "0x8f3c5F4B522803DA8B07a257b6a558f61100452C",
    }
}

def make_config(service, snetd_port, local_jsonrpc_port, contract_address, private_key):
    service_config = dict(config_template)

    service_config['DAEMON_LISTENING_PORT'] = str(snetd_port)
    service_config['PASSTHROUGH_ENDPOINT'] = "http://localhost:" + str(local_jsonrpc_port)
    service_config['AGENT_CONTRACT_ADDRESS'] = contract_address
    service_config['PRIVATE_KEY'] = private_key
    if private_key is None:
        service_config['BLOCKCHAIN_ENABLED'] = False

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
    if not private_key:
        private_key = None

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