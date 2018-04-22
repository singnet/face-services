import numpy as np
import cv2
import dlib
import os

from tests import DEBUG_IMAGE_PATH


def render_landmarks(frame, detected_landmarks):
    for l in detected_landmarks:
        landmarks = np.matrix([[p.x, p.y] for p in l])
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])

            # annotate the positions
            cv2.putText(frame, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.4,
                        color=(0, 0, 255))

            # draw points on the landmark positions
            cv2.circle(frame, pos, 3, color=(0, 255, 255))


def render_bounding_boxes(frame, detected_boxes):
    for i, d in enumerate(detected_boxes):
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)


def debug_image_file_name(test, img_fn):
    base_img_fn = os.path.basename(img_fn)
    parts = test.id().rsplit(".", 2)[1:]
    parts.append(base_img_fn)
    return os.path.join(DEBUG_IMAGE_PATH, "_".join(parts))


def convert_face_detections_proto_to_dlib_rect(detections):
    return [dlib.rectangle(d.x, d.y, d.x + d.w, d.y + d.h) for d in detections.face_bbox]


def render_face_detect_debug_image(test, img_fn, detections):
    if DEBUG_IMAGE_PATH is None:
        return
    image = cv2.imread(img_fn)
    render_bounding_boxes(image, convert_face_detections_proto_to_dlib_rect(detections))
    cv2.imwrite(debug_image_file_name(test, img_fn), image)


def render_face_landmarks_debug_image(test, img_fn, result):
    if DEBUG_IMAGE_PATH is None:
        return
    image = cv2.imread(img_fn)
    for landmarks in result.landmarked_faces:
        render_landmarks(image, [landmarks.point])
    cv2.imwrite(debug_image_file_name(test, img_fn), image)
