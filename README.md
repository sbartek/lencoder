
# LENCODER

Transformations of python's dataframes and arraya that I can understand.

# One-Hot-Encoding


```python
import numpy as np
import pandas as pd

items = ['a', 'b', 'c']
```

### One-Hot-Encoding adding nans


```python
from lencoder import OneHotEncoder

ohenc = OneHotEncoder(items).create_dicts()
encoded = ohenc.encode(np.array(['a', 'b']))
print(encoded)
```

    [[False  True False False]
     [False False  True False]]


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


