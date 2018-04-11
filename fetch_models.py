import requests
import os
import sys
import bz2
import math
import pathlib

from tqdm import tqdm

model_urls = [
    "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
    "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2",
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
]

my_path = os.path.dirname(__file__)

pathlib.Path(os.path.join(my_path, "models")).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    for url in model_urls:

        local_name = os.path.join(my_path, "models", url.split('/')[-1])
        print("%s => %s" % (url, local_name))

        response = requests.get(url, stream=True)

        decompressor = bz2.BZ2Decompressor()
        local_name_unzip = ".".join(local_name.split('.')[:-1])

        total_size = int(response.headers.get('content-length', 0));

        block_size = 1024
        wrote = 0
        with open(local_name_unzip, "wb") as handle:
            for data in tqdm(response.iter_content(block_size), total=math.ceil(total_size//block_size), unit='KB'):
                if data:
                    wrote = wrote + len(data)
                    uncompress_data = decompressor.decompress(data)
                    handle.write(uncompress_data)

        if total_size != 0 and wrote != total_size:
            print("ERROR, something went wrong")