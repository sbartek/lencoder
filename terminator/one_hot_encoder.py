import numpy as np

from .encoder import Encoder

class OneHotEncoder(Encoder):

    def encode(self, itmes_to_encode):
        print(self.max_number)
        return one_hot_encoding_eye(
            super().encode(itmes_to_encode),
            max_number=self.max_number)
    
def one_hot_encoding_eye(nums, max_number=None, dtype=np.bool):
    if max_number is None:
        max_number = max(nums)
    return np.eye(max_number + 1, dtype=dtype)[nums]
