import os

import numpy as np
import pandas as pd

MODELS_DIR = 'data/models/'

def save_y_tilde(dir_name, name, y_tilde):
    pd.DataFrame({name: y_tilde})\
        .to_feather('{dir_name}/{name}.feather'.format(dir_name=dir_name, name=name))

def save_y_train_dev_val_test_tildes(name, y_tildes):
    sets = ['train', 'dev', 'val', 'test']
    for i in range(len(sets)):
        save_y_tilde('data/models', name + '_' + sets[i], y_tildes[i])

def read_train_dev_val_test():
    train = pd.read_feather('data/train.feather')
    dev = pd.read_feather('data/dev.feather')
    val = pd.read_feather('data/val.feather')
    test = pd.read_feather('data/test.feather')
    train_dev = pd.concat([train, dev]).reset_index(drop=True)
    train_dev_val = pd.concat([train_dev, val]).reset_index(drop=True)
    return train, train_dev, train_dev_val, dev, val, test

def get_Xs_ys_read_train_dev_val_test():
    train, train_dev, train_dev_val, dev, val, test = read_train_dev_val_test()
    X_train, y_train = train.drop('target', axis=1), train['target']
    X_train_dev, y_train_dev = train_dev.drop('target', axis=1), train_dev['target']
    X_train_dev_val, y_train_dev_val = train_dev_val.drop('target', axis=1),train_dev_val['target']
    X_dev, y_dev = dev.drop('target', axis=1), dev['target']
    X_val, y_val = val.drop('target', axis=1), val['target']
    X_test, y_test = test.drop('target', axis=1), test['target']
    return X_train, y_train, X_train_dev, y_train_dev, X_train_dev_val, y_train_dev_val, X_dev, y_dev, X_val, y_val, X_test, y_test

def get_ys_train_dev_val_test():
    train, train_dev, train_dev_val, dev, val, test = read_train_dev_val_test()
    y_train = train['target']
    y_train_dev = train_dev['target']
    y_train_dev_val = train_dev_val['target']
    y_dev = dev['target']
    y_val = val['target']
    y_test = test['target']
    return y_train, y_train_dev, y_train_dev_val, y_dev, y_val, y_test

def get_models_fns(name):
    return [fn for fn in os.listdir(MODELS_DIR) if fn.endswith("{}.feather".format(name))]

def get_y_tildes(name):
    fns = get_models_fns(name)
    return pd.concat([pd.read_feather(os.path.join(MODELS_DIR, fn)).reset_index(drop=True) for fn in fns], axis=1)

def get_y_tildes_train_dev_val_test():
    y_train_tilde = get_y_tildes('train')
    y_dev_tilde = get_y_tildes('dev')
    y_val_tilde = get_y_tildes('val')
    y_test_tilde = get_y_tildes('test')
    return y_train_tilde, y_dev_tilde, y_val_tilde, y_test_tilde
