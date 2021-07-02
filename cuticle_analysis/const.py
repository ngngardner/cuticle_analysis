DATASET = 'dataset'
DATASET_ALL = 'all'
DATASET_BG = 'background'
DATASET_RS = 'rough_smooth'

LABEL_MAP = {
    '_background_': 0,
    'antenna_base': 1,
    'cuticle': 2,
    'cuticle_extra': 3,
    'eye': 4
}

DS_MAP = {
    DATASET_BG: 'bg',
    DATASET_RS: 'rs'
}

BG_LABEL_MAP = {
    '_background_': 0,
    'cuticle': 1,
}

RS_LABEL_MAP = {
    '_background_': 0,
    'rough': 1,
    'smooth': 2,
}

INT_RS_LABEL_MAP = {v: k for k, v in RS_LABEL_MAP.items()}
