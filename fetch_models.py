import requests
import os
import sys
import json
import math
import pathlib

from tqdm import tqdm

model_urls = []
with open("models.json", "r") as f:
    model_urls = json.load(f)["model_urls"]

# TODO maybe make models.json a parameter and use it's location as the root/my_path
my_path = os.getcwd()

if __name__ == "__main__":
    pathlib.Path(os.path.join(my_path, "models")).mkdir(parents=True, exist_ok=True)
    download_dir = os.path.join(my_path, "downloads")
    if len(sys.argv) > 1:
        download_dir = sys.argv[1]
    pathlib.Path(download_dir).mkdir(parents=True, exist_ok=True)

    for url, file_filter in model_urls:
        model_dir = os.path.join(my_path, "models")

        local_name = os.path.join(download_dir, url.split('/')[-1])
        print("Downloading %s => %s" % (url, local_name))

        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        if os.path.exists(local_name) and ('content-length' not in response.headers or int(response.headers.get('content-length')) == os.path.getsize(local_name)):
            print(local_name + " is already downloaded.")
            del response
        else:
            block_size = 1024
            wrote = 0

            with open(local_name, 'wb') as handle:
                for data in tqdm(response.iter_content(block_size), total=math.ceil(total_size//block_size), unit='KB'):
                    if data:
                        wrote = wrote + len(data)
                        handle.write(data)

            if total_size != 0 and wrote != total_size:
                raise Exception("ERROR, total size does not match total written bytes")

        if local_name.endswith('.bz2'):
            if '.tar.' in local_name:
                import tarfile
                tar = tarfile.open(local_name, "r:bz2")  
                for tar_item in tar:
                    if file_filter is None or tar_item.name in file_filter:
                        print("Extracting %s : %s => %s" % (local_name, tar_item.name, os.path.join(model_dir, tar_item.name)))
                        tar.extract(tar_item, path=model_dir)
                tar.close()
            else:
                import bz2
                local_name_unzip = os.path.join(model_dir, '.'.join(url.split('/')[-1].split('.')[:-1]))
                print("Extracting %s => %s" % (local_name, local_name_unzip))
                with bz2.BZ2File(local_name) as f:
                    with open(local_name_unzip, 'wb') as dest:
                        dest.write(f.read())
        elif local_name.endswith('.zip'):
            from zipfile import ZipFile
            zfile = ZipFile(local_name)
            for filename in zfile.namelist():
                if file_filter is None or filename in file_filter:
                    print("Extracting %s : %s => %s" % (local_name, filename, os.path.join(model_dir, filename)))
                    zfile.extract(filename, path=model_dir)
        else:
            import shutil
            dest_local_name = os.path.join(model_dir, url.split('/')[-1])
            print("Copying %s => %s" % (local_name, dest_local_name))
            shutil.copyfile(local_name, dest_local_name)