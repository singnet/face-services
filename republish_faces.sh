#!/bin/bash
set -e
set -o xtrace


regenerate_service() {
    snet service metadata-init --metadata-file $MD_FILE $PROTODIR $SERVICE_NAME $PAYMENT_ADDRESS
    snet service metadata-set-fixed-price --metadata-file $MD_FILE $PRICE
    snet service metadata-add-endpoints --metadata-file $MD_FILE http://$HOST:$PORT
    snet service publish -y --metadata-file $MD_FILE $ORG $SERVICE_NAME

    cat << EOF > $SNETD_CONFIG
{
"DAEMON_END_POINT": "http://$HOST:$PORT",
"ETHEREUM_JSON_RPC_ENDPOINT": "https://kovan.infura.io",
"IPFS_END_POINT": "http://ipfs.singularitynet.io:80",
"REGISTRY_ADDRESS_KEY": "0xe331bf20044a5b24c1a744abc90c1fd711d2c08d",
"PASSTHROUGH_ENABLED": true,
"PASSTHROUGH_ENDPOINT": "http://localhost:$PORT",
"ORGANIZATION_ID": "$ORG",
"SERVICE_ID": "$SERVICE_NAME",

"payment_channel_storage_client": {
    "connection_timeout": "5s",
    "request_timeout": "3s",
    "endpoints": ["http://127.0.0.1:$ETCD_CLIENT_PORT"]
},
"payment_channel_storage_server": {
    "id": "storage-${ETCD_ID}",
    "host" : "127.0.0.1",
    "client_port": $ETCD_CLIENT_PORT,
    "peer_port": $ETCD_PEER_PORT,
    "token": "unique-token",
    "cluster": "storage-${ETCD_ID}=http://127.0.0.1:$ETCD_PEER_PORT",
    "data_dir": "etcd/storage-data-dir-${ETCD_ID}.etcd",
    "enabled": true
},

"LOG": {
    "LEVEL": "debug",
        "output": {
                        "current_link": "./snetd-${SERVICE_NAME}.log",
                        "file_pattern": "./snetd-${SERVICE_NAME}.%Y%m%d.log",
                        "rotation_count": 0,
                        "rotation_time_in_sec": 86400,
                        "type": "file"
        }
    }
} 
EOF

}

# common config
PAYMENT_ADDRESS="0x464c564e427fA7A715922D9E0373a5D90589E021"
HOST=34.216.72.29
PROTODIR=services/grpc
ORG=snet


MD_FILE=service_metadata_face_detect.json
SNETD_CONFIG=snetd_face_detect_config.json
SERVICE_NAME=face-detect
PORT=6201
PRICE=0.000001
ETCD_ID=1
ETCD_CLIENT_PORT=2379
ETCD_PEER_PORT=2380
#regenerate_service



MD_FILE=service_metadata_face_landmarks.json
SNETD_CONFIG=snetd_face_landmarks_config.json
SERVICE_NAME=face-landmarks
PORT=6202
PRICE=0.000001
ETCD_ID=2
ETCD_CLIENT_PORT=2381
ETCD_PEER_PORT=2382
regenerate_service

MD_FILE=service_metadata_face_align.json
SNETD_CONFIG=snetd_face_align_config.json
SERVICE_NAME=face-align
PORT=6203
PRICE=0.000001
ETCD_ID=3
ETCD_CLIENT_PORT=2383
ETCD_PEER_PORT=2384
regenerate_service

MD_FILE=service_metadata_face_identity.json
SNETD_CONFIG=snetd_face_identity_config.json
SERVICE_NAME=face-identity
PORT=6204
PRICE=0.000001
ETCD_ID=4
ETCD_CLIENT_PORT=2385
ETCD_PEER_PORT=2386
regenerate_service

snet organization list-services snet

