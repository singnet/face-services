import jsonrpcclient
import sys
import base64
from tests.test_images import one_face

SERVER_PORT=8080

if __name__ == '__main__':
    response = jsonrpcclient.request("http://127.0.0.1:{}".format(SERVER_PORT), "ping")
    print(response)

    with open(one_face[0], "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('ascii')

    response = jsonrpcclient.request("http://127.0.0.1:{}".format(SERVER_PORT), "find_face",
                                     algorithm="dlib_hog",
                                     image=img_base64
                                     )

    with open(one_face[1], "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode('ascii')

    response = jsonrpcclient.request("http://127.0.0.1:{}".format(SERVER_PORT), "find_face",
                                     algorithm="dlib_hog",
                                     image=img_base64
                                     )