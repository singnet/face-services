import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.joinpath('grpc')))

registry = {
    'face_detect_server': {
        'grpc': 6001,
        'snetd': 6201,
    },
    'face_landmarks_server': {
        'grpc': 6002,
        'snetd': 6202,
    },
    'face_align_server': {
        'grpc': 6003,
        'snetd': 6203,
    },
    'face_identity_server': {
        'grpc': 6004,
        'snetd': 6204,
    },
}