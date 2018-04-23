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

pre_calculated_faces = {
    'classroom_in_tanzania.jpg': [
        {'x': 633, 'y': 245, 'w': 90, 'h': 90},
        {'x': 155, 'y': 216, 'w': 90, 'h': 89},
        {'x': 59, 'y': 36, 'w': 44, 'h': 43},
        {'x': 218, 'y': 55, 'w': 43, 'h': 43},
        {'x': 118, 'y': 101, 'w': 52, 'h': 52},
        {'x': 519, 'y': 196, 'w': 75, 'h': 75},
        {'x': 259, 'y': 331, 'w': 107, 'h': 107},
    ],
    'laos.jpg': [
        {'x': 22, 'y': 260, 'w': 268, 'h': 268},
        {'x': 538, 'y': 217, 'w': 223, 'h': 223},
    ],
    'adele_2016.jpg': [
        {'x': 142, 'y': 118, 'w': 223, 'h': 223},
    ],
    "elon_musk_2015.jpg": [
        {'x': 117, 'y': 142, 'w': 223, 'h': 223},
    ],
    "waaah.jpg": [
        {'x': 315, 'y': 142, 'w': 223, 'h': 223}
    ],
    "city_portrait.jpg": [
        {'x': 233, 'y': 172, 'w': 555, 'h': 554}
    ],
    "benedict_cumberbatch_2014.png": [
        {'x': 97, 'y': 118, 'w': 186, 'h': 186}
    ],
    "model_face.jpg": [
        {'x': 206, 'y': 280, 'w': 666, 'h': 666}
    ],
    "sophia.jpg": [
        {'x': 221, 'y': 77, 'w': 186, 'h': 186}
    ],
    "woman_portrait.jpg": [
        {'x': 85, 'y': 298, 'w': 958, 'h': 958}
    ]
}

def abs_path(fn):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), fn))

one_face = [abs_path(fn) for fn in __one_face]
multiple_faces = [abs_path(fn) for fn in __multiple_faces]
no_faces = [abs_path(fn) for fn in __no_faces]