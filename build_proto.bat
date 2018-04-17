python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. services\grpc\face_common.proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. services\grpc\face_detect.proto
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. services\grpc\face_landmarks.proto
