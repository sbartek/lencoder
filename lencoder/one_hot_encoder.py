import numpy as np

from .encoder import Encoder

class OneHotEncoder(Encoder):

    @property
    def encoder(self):
        return super()
    
    def encode(self, itmes_to_encode):
        return one_hot_encoding_eye(
            super().encode(itmes_to_encode),
            max_number=self.max_number)

    def decode(self, array_to_decode):
        items_to_decode = array_to_decode.argmax(axis=1)
        return super().decode(items_to_decode)

def one_hot_encoding_eye(nums, max_number=None, dtype=np.bool):
    if max_number is None:
        max_number = max(nums)
    return np.eye(max_number + 1, dtype=dtype)[nums]
