#!/bin/bash
set -e
set -o xtrace

mkdir -p services/grpc/snet_hack/face_detect
cp services/grpc/face_{detect,common}.proto services/grpc/snet_hack/face_detect/.
snet service metadata-set-model --metadata-file service_metadata_face_detect.json services/grpc/snet_hack/face_detect
snet service update-metadata --yes --metadata-file service_metadata_face_detect.json snet face-detect

mkdir -p services/grpc/snet_hack/face_landmarks
cp services/grpc/face_{landmarks,common}.proto services/grpc/snet_hack/face_landmarks/.
snet service metadata-set-model --metadata-file service_metadata_face_landmarks.json services/grpc/snet_hack/face_landmarks
snet service update-metadata --yes --metadata-file service_metadata_face_landmarks.json snet face-landmarks

mkdir -p services/grpc/snet_hack/face_align
cp services/grpc/face_{alignment,common}.proto services/grpc/snet_hack/face_align/.
snet service metadata-set-model --metadata-file service_metadata_face_align.json services/grpc/snet_hack/face_align
snet service update-metadata --yes --metadata-file service_metadata_face_align.json snet face-align

mkdir -p services/grpc/snet_hack/face_identity
cp services/grpc/face_{recognition,common}.proto services/grpc/snet_hack/face_identity/.
snet service metadata-set-model --metadata-file service_metadata_face_identity.json services/grpc/snet_hack/face_identity
snet service update-metadata --yes --metadata-file service_metadata_face_identity.json snet face-identity