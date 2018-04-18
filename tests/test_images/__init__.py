import os

__one_face = [
    "adele_2016.jpg",
    "elon_musk_2015.jpg",
    "waaah.jpg",
    "city_portrait.jpg",
    "benedict_cumberbatch_2014.png",
    "model_face.jpg",
    "sophia.jpg",
    "woman_portrait.jpg",
]

__multiple_faces = [
    "classroom_in_tanzania.jpg",
    "laos.jpg",
]

__no_faces = [
    "zebra_stripes.jpg",
]

def abs_path(fn):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), fn))

one_face = [abs_path(fn) for fn in __one_face]
multiple_faces = [abs_path(fn) for fn in __multiple_faces]
no_faces = [abs_path(fn) for fn in __no_faces]