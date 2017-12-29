# lencoder


Transformations of python's dataframes that I can understand.

## One hot emcoder

Add nans:

```
from lencoder import OneHotEncoder

items = ['a', 'b', 'c']
ohenc = OneHotEncoder(items).create_dicts()
encoded = ohenc.encode(np.array(['a', 'b']))
print(encoded)
```

```
print(ohenc.decode(encoded))
```

```
ohenc = OneHotEncoder(items, add_nan=False).create_dicts()
encoded = ohenc.encode(np.array(['a', 'b']))
print(encoded)
```

