import pandas as pd
import numpy as np

def dict2fun(dictionary):
    def fun(key):
        return dictionary[key]
    return fun

def dict2fun_with_nan_replacement(dictionary, nan_replacement):
    def fun(key):
        return dictionary.get(key, dictionary[nan_replacement])
    return fun
