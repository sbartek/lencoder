
# LENCODER

Transformations of python's dataframes and arraya that I can understand.


```python
import numpy as np
import pandas as pd

items = ['a', 'b', 'c']
```

## Encoder


```python
from lencoder import Encoder

enc = Encoder(items).create_dicts()
encoded = enc.encode(np.array(['a', 'b']))
print(encoded)
```

    [1 2]


One number (here `0`) is reserved for `nan`s or not encoded labels.


```python
enc.item2num
```




    {'<NAN>': 0, 'a': 1, 'b': 2, 'c': 3}




```python
enc.encode(np.array(['not existing', 'b']))
```




    array([0, 2])



If you do not want to reserve something for `nan`s use:


```python
enc = Encoder(items, add_nan=False).create_dicts()
enc.item2num
```




    {'a': 0, 'b': 1, 'c': 2}



## One-Hot-Encoding

### One-Hot-Encoding adding nans


```python
from lencoder import OneHotEncoder

ohenc = OneHotEncoder(items).create_dicts()
encoded = ohenc.encode(np.array(['a', 'b']))
print(encoded)
```

    [[False  True False False]
     [False False  True False]]


#### Encoding new items gives the same result as nan


```python
ohenc.encode(np.array(['something new', 'hehehe']))
```




    array([[ True, False, False, False],
           [ True, False, False, False]], dtype=bool)



### One-Hot-Encoding no nans


```python
ohenc = OneHotEncoder(items, add_nan=False).create_dicts()
encoded = ohenc.encode(np.array(['a', 'b']))
print(encoded)
```

    [[ True False False]
     [False  True False]]



```python
ohenc.decode(encoded)
```




    array(['a', 'b'],
          dtype='<U1')



#### Save and load from disk


```python
ohenc.dump_dicts('ohe_nonans_')
```


```python
ohenc_from_saved = OneHotEncoder.create_from_saved_dicts('ohe_nonans_')
```


```python
ohenc_from_saved.encode(np.array(['a', 'b']))
```




    array([[ True, False, False],
           [False,  True, False]], dtype=bool)



### One-Hot-Encoding of columns, directly


```python
items_df = pd.DataFrame({'col': items})
```


```python
from lencoder import ColumnOneHotEncoder
```


```python
cohenc = ColumnOneHotEncoder(items=items_df['col'], colname='col', add_nan=False)
cohenc.create_dicts().encode(items_df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_0</th>
      <th>col_1</th>
      <th>col_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
items_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
    </tr>
  </tbody>
</table>
</div>




```python
cohenc = ColumnOneHotEncoder(items=items_df['col'], colname='col')
cohenc.create_dicts().encode(items_df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_0</th>
      <th>col_1</th>
      <th>col_2</th>
      <th>col_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



#### Save and load from disk


```python
cohenc.dump_dicts("column_ohe_with_nans")
```


```python
saved_cohenc = ColumnOneHotEncoder.create_from_saved_dicts("column_ohe_with_nans")
```


```python
saved_cohenc.encode(items_df)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col_0</th>
      <th>col_1</th>
      <th>col_2</th>
      <th>col_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>


