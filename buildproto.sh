#!/bin/sh
GRPC_DIR=services/grpc
python3.6 -m grpc_tools.protoc -I$GRPC_DIR --python_out=$GRPC_DIR --grpc_python_out=$GRPC_DIR services/grpc/face_common.proto
python3.6 -m grpc_tools.protoc -I$GRPC_DIR --python_out=$GRPC_DIR --grpc_python_out=$GRPC_DIR services/grpc/face_detect.proto
python3.6 -m grpc_tools.protoc -I$GRPC_DIR --python_out=$GRPC_DIR --grpc_python_out=$GRPC_DIR services/grpc/face_landmarks.proto
python3.6 -m grpc_tools.protoc -I$GRPC_DIR --python_out=$GRPC_DIR --grpc_python_out=$GRPC_DIR services/grpc/face_recognition.proto
python3.6 -m grpc_tools.protoc -I$GRPC_DIR --python_out=$GRPC_DIR --grpc_python_out=$GRPC_DIR services/grpc/face_alignment.proto