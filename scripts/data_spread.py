
import re

import numpy as np

from cuticle_analysis.datasets import RoughSmoothSub

data = RoughSmoothSub((16, 16))
print(np.uniques(data.labels))
