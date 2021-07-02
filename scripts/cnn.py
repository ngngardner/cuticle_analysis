
import logging

import numpy as np

from cuticle_analysis.datasets import RoughSmoothFull
from cuticle_analysis.models import CNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

d = RoughSmoothFull((128, 128), save=True)
m = CNN(d)
m.train(20, 500)
