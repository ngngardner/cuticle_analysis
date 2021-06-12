
import numpy as np

from .cnn import CNN
from .kviews import KViews
from . import const


class EndToEnd():
    def __init__(
        self,
        bg_model: CNN,
        rs_model: KViews
    ) -> None:
        self.name = 'EndToEnd'
        self.bg_model = bg_model
        self.rs_model = rs_model

    def metadata(self):
        return self.bg_model.metadata() + self.rs_model.metadata()

    def predict(self, image: np.ndarray) -> np.ndarray:
        # first find background
        preds = self.bg_model.predict(image)

        # cuticle detected, so use rs_model
        if preds.any() == const.BG_LABEL_MAP['cuticle']:
            idx = np.where(preds == 1)
            rs_preds = self.rs_model.predict(image[idx])

            # remap (0, 1) to (1, 2)
            mp = {0: 1, 1: 2}
            rs_preds = np.array([mp[i] for i in rs_preds])
            preds[idx] = rs_preds

        return preds
