#!/bin/bash
set -e
set -o xtrace



# port ranges
# 6201-6210 - kovan
# 6301-6310 - ropsten
# 6401-6410 - mainnet

publish_service() {
    mkdir -p services/grpc/snet_hack/$SERVICE_NAME
    cp $PROTO_SERVICE_FILE services/grpc/face_common.proto services/grpc/snet_hack/$SERVICE_NAME/.

    snet service metadata-init --metadata-file $MD_FILE services/grpc/snet_hack/$SERVICE_NAME $SERVICE_NAME $PAYMENT_ADDRESS
    snet service metadata-set-fixed-price --metadata-file $MD_FILE $PRICE
    snet service metadata-add-endpoints --metadata-file $MD_FILE $HOST:$PORT
    snet service publish -y --metadata-file $MD_FILE $ORG $SERVICE_NAME

    cat << EOF > $SNETD_CONFIG
{
"DAEMON_END_POINT": "0.0.0.0:$SNETD_PORT",
"ETHEREUM_JSON_RPC_ENDPOINT": "$ETHEREUM_ENDPOINT",
"IPFS_END_POINT": "http://ipfs.singularitynet.io:80",
"REGISTRY_ADDRESS_KEY": "0xe331bf20044a5b24c1a744abc90c1fd711d2c08d",
"PASSTHROUGH_ENABLED": true,
"PASSTHROUGH_ENDPOINT": "http://localhost:$SERVICE_PORT",
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

publish_kovan()
{
    snet network kovan
    mkdir -p config/kovan
    # common config
    ETHEREUM_ENDPOINT="https://kovan.infura.io"
    PAYMENT_ADDRESS="0x464c564e427fA7A715922D9E0373a5D90589E021"
    HOST=http://34.216.72.29
    ORG=snet
    
    MD_FILE=config/kovan/service_metadata_face_detect.json
    SNETD_CONFIG=config/kovan/snetd_face_detect_config.json
    PROTO_SERVICE_FILE=services/grpc/face_detect.proto
    SERVICE_NAME=face-detect
    SNETD_PORT=6201
    SERVICE_PORT=6001
    PRICE=0.000001
    ETCD_ID=1
    ETCD_CLIENT_PORT=2379
    ETCD_PEER_PORT=2380
    publish_service

    MD_FILE=config/kovan/service_metadata_face_landmarks.json
    SNETD_CONFIG=config/kovan/snetd_face_landmarks_config.json
    PROTO_SERVICE_FILE=services/grpc/face_landmarks.proto
    SERVICE_NAME=face-landmarks
    SNETD_PORT=6202
    SERVICE_PORT=6002
    PRICE=0.000001
    ETCD_ID=2
    ETCD_CLIENT_PORT=2381
    ETCD_PEER_PORT=2382
    publish_service

    MD_FILE=config/kovan/service_metadata_face_align.json
    SNETD_CONFIG=config/kovan/snetd_face_align_config.json
    PROTO_SERVICE_FILE=services/grpc/face_alignment.proto
    SERVICE_NAME=face-align
    SNETD_PORT=6203
    SERVICE_PORT=6003
    PRICE=0.000001
    ETCD_ID=3
    ETCD_CLIENT_PORT=2383
    ETCD_PEER_PORT=2384
    publish_service

    MD_FILE=config/kovan/service_metadata_face_identity.json
    SNETD_CONFIG=config/kovan/snetd_face_identity_config.json
    PROTO_SERVICE_FILE=services/grpc/face_recognition.proto
    SERVICE_NAME=face-identity
    SNETD_PORT=6204
    SERVICE_PORT=6004
    PRICE=0.000001
    ETCD_ID=4
    ETCD_CLIENT_PORT=2385
    ETCD_PEER_PORT=2386
    publish_service
}

publish_ropsten()
{
    snet network ropsten
    mkdir -p config/ropsten
    # common config
    PAYMENT_ADDRESS="0x464c564e427fA7A715922D9E0373a5D90589E021"
    HOST=https://services-1.snet.sh
    ORG=snet
    ETHEREUM_ENDPOINT="https://ropsten.infura.io"

    MD_FILE=config/ropsten/service_metadata_face_detect.json
    SNETD_CONFIG=config/ropsten/snetd_face_detect_config.json
    PROTO_SERVICE_FILE=services/grpc/face_detect.proto
    SERVICE_NAME=face-detect
    SNETD_PORT=6301
    SERVICE_PORT=6001
    PRICE=0.000001
    ETCD_ID=1
    ETCD_CLIENT_PORT=2379
    ETCD_PEER_PORT=2380
    publish_service

    MD_FILE=config/ropsten/service_metadata_face_landmarks.json
    SNETD_CONFIG=config/ropsten/snetd_face_landmarks_config.json
    PROTO_SERVICE_FILE=services/grpc/face_landmarks.proto
    SERVICE_NAME=face-landmarks
    SNETD_PORT=6302
    SERVICE_PORT=6002
    PRICE=0.000001
    ETCD_ID=2
    ETCD_CLIENT_PORT=2381
    ETCD_PEER_PORT=2382
    publish_service

    MD_FILE=config/ropsten/service_metadata_face_align.json
    SNETD_CONFIG=config/ropsten/snetd_face_align_config.json
    PROTO_SERVICE_FILE=services/grpc/face_alignment.proto
    SERVICE_NAME=face-align
    SNETD_PORT=6303
    SERVICE_PORT=6003
    PRICE=0.000001
    ETCD_ID=3
    ETCD_CLIENT_PORT=2383
    ETCD_PEER_PORT=2384
    publish_service

    MD_FILE=config/ropsten/service_metadata_face_identity.json
    SNETD_CONFIG=config/ropsten/snetd_face_identity_config.json
    PROTO_SERVICE_FILE=services/grpc/face_recognition.proto
    SERVICE_NAME=face-identity
    SNETD_PORT=6304
    SERVICE_PORT=6004
    PRICE=0.000001
    ETCD_ID=4
    ETCD_CLIENT_PORT=2385
    ETCD_PEER_PORT=2386
    publish_service
}



if [ "$1" = "kovan" ]; then
    publish_kovan
elif [ "$1" = "ropsten" ]; then
    publish_ropsten
else
    echo "usage $0 [kovan/ropsten]"
fi

snet organization list-services snet