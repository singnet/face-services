# Face Services

This repository contains a number of services related to the detection, tracking, identification, and manipulation
of faces.

## Overview

Initially, services will be implemented that support the following tasks:

- Face localization - provides bounding boxes where faces are detected.
- Face landmark detection - provides a set of face keypoints based on a keypoint model.
- Face alignment - transforms face (rotate, translates, and scales) to a template landmark layout.
- Face recognition - return a vector representing the faces identity mapped to N-dimensional manifold.

There are different techniques to solve these tasks, and the goal is provide multiple implementations so that
upstream tasks can swap implementations depending on availability, price, their impact on reputation and performance,
or other factors. To begin with, this is a simple wrapping of dlib and opencv algorithms.

## Dependencies

Install Anaconda3, then open terminal with the Anaconda environment

```
conda create --name face-services python=3.6
conda activate face-services
pip install -r requirements.txt
```

You also need to download various pretrained models

```
python fetch_models.py
```

## Webcam test

![alt text](example_webcam.jpg)

Eventually all services will call each other via some RPC mechanism, but while attempting to get each part working well
there is a `webcam_test.py` script.

This will activate your webcam and overlay outputs from each stage of processing, run with `python webcam_test.py`
from the conda `face-services` environment you created above.

## Face Service Details

The aim is that any component can be swapped out with another, and be able to gracefully handle different inputs.
This section aims to describe those interfaces, and responsibilities. It will become more specific as the
implementation progresses.

### Face localization

Implementations: opencv haar cascade, dlib HOG, dlib CNN

Calls:
- `get_bounding_boxes` -> expects rgb image, return a number bounding boxes where faces are detected 0..N,
optionally return rgb image with bounding box annotations.

### Face landmark detection

Implementations: dlib 68 point CNN, dlib 5 point CNN, clmtrack

Calls:
- `get_landmark_models` -> no arguments, return list of landmark models, including description of each landmark,
  e.g. "tip of nose", optionally also include
- `get_landmarks` -> expects rgb image. Find faces, then return x,y locations for each landmark.
- `get_landmarks_for_faces` -> expects rgb image, a list of face detection bboxes.
  For each face bbox, return x,y locations for each landmark

### Face alignment

Implementations: dlib `save_face_chips`, opencv `getAffineTransform` or `getPerspectiveTransform`.

Calls:
- `align_image` -> expects rgb image, landmarks in image, destination locations for landmarks, optional
  specify type of transform and how to handle borders (mirror, zero, etc). Return aligned rgb image (with error?)

### Face recognition

Calls:
- `get_landmark_models` -> no arguments, return list of landmark models recognition algorithm uses for alignment, including
mean locations, so callers can pre-align the face if they want.
- `recognise_face` -> expects rgb image, and list of face detections using one of the landmark models.
  Return 128D vector of floats representing identity
- `recognise_face_prealigned` -> expects a pre-aligned image of a face, 150x150px.
  Return 128D vector of floats representing identity

The 128D vector of floats has no shared meaning to other services, i.e. one can't compare it from one
recognition service to another. Not sure if the API should describe this or

## Support Service Details

A couple of generally useful services which may be worth splitting off into their own repo eventually.

### Mapping frames in video to an image service

Video processing can smooth the noise out of frame-by-frame
predictions or use interpolation to avoid processing every frame (if the upstream service is too slow/expensive).

A video service could provide a bridge from image-based services to video.

For localizations, shapes, or segmentation, there are different ways this could be approached:
- Take a video and send key frames to a image-based service, interpolate between them using the motion vectors
  inherent in the video stream.
- Take a video and send every Nth frame to image-based service. Use optical-flow and template-based matching
  to interpolate between frames.

For something like face recognition, we'd want to assign a label to each face descriptor tracked through time.
Since the face descriptor will have noise between frames, we will want to cluster identities and assign a
unique (within the video) label to each. Basic graph clustering (dlib provides easy hooks to the chinese whispers algorithm)
of identity vectors is one approach, but weighting their edges by the spatial/temporal distance of face descriptors
should improve it.

### Image converter

We could reimplement image conversion and manipulation in every service, but until
it is bundled as part of a library, this will be annoying extra work for service authors.

Thus, an image converter service could be helpful:

- load a wide variety of image formats, including those that are less known outside of
  computer-vision/rendering - e.g. exr.
- load these from different storage systems, S3, ipfs, bytestream, URL
- normalise images
- channel swapping, merging, or dropping

## Launch Services

**TODO** Once individual steps work within the live webcam demo, they'll be wrapped up with grpc service definitions and
an automatic launcher launcher.