import pandas as pd
import numpy as np

def dict2fun(dictionary):
    def fun(key):
        return dictionary[key]
    return fun
