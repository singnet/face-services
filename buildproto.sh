#!/bin/sh
python3.6 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. services/grpc/face_common.proto
python3.6 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. services/grpc/face_detect.proto
python3.6 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. services/grpc/face_landmarks.proto
python3.6 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. services/grpc/face_recognition.proto
python3.6 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. services/grpc/face_alignment.proto