LABEL_MAP = {
    '_background_': 0,
    'antenna_base': 1,
    'cuticle': 2,
    'cuticle_extra': 3,
    'eye': 4
}

INT_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

NUM_CLASSES = 2

DATASET = 'dataset'
DATASET_BG = 'background'
DATASET_RS = 'rough_smooth'
DATASET_SI = 'subimage'

# collections of dataset types
DATASETS = [DATASET, DATASET_BG, DATASET_RS, DATASET_SI]
SUBIMAGE_DATASETS = [DATASET_BG, DATASET_RS, DATASET_SI]

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
