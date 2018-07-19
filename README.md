# Face Services

This repository contains a number of prototype services related to the detection, tracking, identification, and manipulation
of faces.

## Overview

Initially, services will be implemented that support the following tasks:

- *Face localization* - provides bounding boxes where faces are detected.
- *Face landmark detection* - provides a set of face keypoints based on a landmark model.
- *Face alignment* - transforms face (rotate, translates, and scales) to a template landmark layout.
- *Face recognition* - return a vector representing the faces identity mapped to N-dimensional manifold.

There are different techniques to solve these tasks. The goal is to provide multiple implementations of some of these,
so that upstream tasks can swap implementations depending on availability, price, their impact on reputation and performance,
or other factors. To begin with, this repository is mostly a wrapping of dlib and opencv algorithms. 

## Dependencies

Install Anaconda3, then open terminal with the Anaconda environment

```
conda create --name face-services python=3.6
conda activate face-services
pip install -r requirements.txt
```

You also need to download various pretrained models and generate the grpc code from the proto definitions:

```
python fetch_models.py
build_proto.bat
```

This repo has been developed on Windows, but is deployed on linux and I've run it on my Macbook.
If you run into any cross-platform issues (or any other type of issue!) please report and I'll fix.

## Webcam test

![alt text](example_webcam.jpg)

Eventually all services will call each other via some RPC mechanism, but while attempting to get each part working
there is a `webcam_test.py` script.

This will activate your webcam and overlay outputs from each stage of processing, run with `python webcam_test.py`
from the conda `face-services` environment you created above. It will open two windows, one with an overlay of the
original webcam image, and a smaller one with the aligned and cropped face.

There are several hot-keys you can use:
- `l` - change landmark detection model.
- `d` - change face detection model.
- `q` - quit.

## Run Face Services

To run the services:

```
python run_services.py
```

To run the services, each one fronted by an instance of the SingularityNET daemon (`snetd`), make sure
you have created your own agent contracts and then use them to generate a directory of config files. One for each
service. It will ask you for the private key for the identity that created the agent contracts:

```
python create_snet_config.py --network kovan --detect-address 0x1234 --landmarks-address 0x2345 --alignment-address 0x3456 --recognition-address 0x4567
python run_services.py --daemon-config-path config/
```

There are also Dockerfiles for gpu or cpu deployments. Runtime selection isn't possible because dlib choses
the execution method at compilation time. 

## Calling Services on SingularityNet

The clients directory has command line tools for calling each via SingularityNET. Use Kovan, and make sure you
have snet-cli configured to use an identity with KETH and AGI tokens.

```
pip install -r clients/requirements.txt # only needed if you haven't installed the root requirements.txt
python -m clients.face_detect_jsonrpc_client --image tests/test_images/laos.jpg --snet --out-image ~/laos_face_detect.jpg
```

In the terminal output it should tell you the bounding boxes, which you can then use for the other services, e.g.: 
```
python -m clients.face_landmarks_jsonrpc_client --image tests/test_images/laos.jpg --snet --out-image ~/laos_face_landmarks.jpg --face-bb 511,170,283,312 --face-bb 61,252,236,259
```

## Service description

Each service defines both a grpc and jsonrpc server. grpc is better formalised and client code is generated,
but SingularityNet currently only supports jsonrpc. I've tried to keep the method names the same, except
that grpc uses CamelCase (e.g. FindFace) whereas jsonrpc method names use underscores (e.g. find_face). 

### Face localization

Implementations:
- ✓ [opencv haar cascade](https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html),
- ✓ [dlib HOG and SVM](https://github.com/davisking/dlib/blob/master/python_examples/face_detector.py),
- ✓ [dlib CNN](https://github.com/davisking/dlib/blob/master/python_examples/cnn_face_detector.py)

Calls:
- `FindFace` -> expects rgb image, return a number of bounding boxes where faces are detected.

### Face landmark detection

Implementations:
- ✓ [dlib 68 point CNN](https://github.com/davisking/dlib/blob/master/python_examples/face_landmark_detection.py)
- ✓ [dlib 5 point CNN](http://blog.dlib.net/2017/09/fast-multiclass-object-detection-in.html)
- possibly [clmtrack](https://github.com/auduno/clmtrackr) to show a js service working in conjunction with python/c++?

Calls:
- `GetLandmarkModels` -> no arguments, return list of landmark models, including description of each landmark,
  e.g. "tip of nose", optionally also return rgb image showing the layout 
- `GetLandmarks` -> expects rgb image, a list of face detection bboxes.
  For each face bbox, return x,y locations for each landmark

### Face alignment

Implementations:
- ✓ dlib `save_face_chips`
- opencv `getAffineTransform` or `getPerspectiveTransform`.

Calls:
- `AlignFace` -> expects rgb image and detected face bounding boxes. Return aligned rgb image.

### Face recognition

Implementations:
- ✓ [dlib CNN](https://github.com/davisking/dlib/blob/master/python_examples/face_recognition.py) 

Calls:
- `RecogniseFace` -> expects rgb image, and list of face detections bounding boxes
  Return 128D vector of floats representing identity

The 128D vector of floats has no shared meaning to other services, i.e. one can't compare it from one
recognition service to another.