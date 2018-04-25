import os.path

DEBUG_IMAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "debug_images"))

if DEBUG_IMAGE_PATH is not None:
    os.makedirs(DEBUG_IMAGE_PATH, exist_ok=True)
