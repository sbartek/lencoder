
# LENCODER

Transformations of python's dataframes and arraya that I can understand.

## Install

```
pip install git+https://github.com/sbartek/lencoder
```


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
           [ True, False, False, False]])



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




    array(['a', 'b'], dtype='<U1')



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
           [False,  True, False]])



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



# Value Encoding


```python
df = pd.DataFrame({
            'days': sorted(list(range(4)) * 4),
            'group': sorted(['A', 'B'] * 2) * 4,
            'value1': sorted(list(range(8)) * 2),
            'value2': list(range(8)) * 2
        })
df
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
      <th>days</th>
      <th>group</th>
      <th>value1</th>
      <th>value2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>A</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>A</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>B</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>B</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>A</td>
      <td>2</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>A</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>B</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>B</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>A</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>A</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2</td>
      <td>B</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2</td>
      <td>B</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3</td>
      <td>A</td>
      <td>6</td>
      <td>4</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3</td>
      <td>A</td>
      <td>6</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3</td>
      <td>B</td>
      <td>7</td>
      <td>6</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3</td>
      <td>B</td>
      <td>7</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
from lencoder.value_encoder import ValueEncoder
venc = ValueEncoder(
            df, ['days'], ['value1', 'value2'],
            aggregations=['mean', 'sum'])
encoded_df = venc.encode()
encoded_df
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
      <th>days</th>
      <th>days:value1:mean</th>
      <th>days:value1:sum</th>
      <th>days:value2:mean</th>
      <th>days:value2:sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.5</td>
      <td>2</td>
      <td>1.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.5</td>
      <td>10</td>
      <td>5.5</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4.5</td>
      <td>18</td>
      <td>1.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>6.5</td>
      <td>26</td>
      <td>5.5</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.value_encodigs(
            ['days'], ['value1', 'value2'],
            aggregations=['mean', 'sum'])
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
      <th>days</th>
      <th>days:value1:mean</th>
      <th>days:value1:sum</th>
      <th>days:value2:mean</th>
      <th>days:value2:sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.5</td>
      <td>2</td>
      <td>1.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2.5</td>
      <td>10</td>
      <td>5.5</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4.5</td>
      <td>18</td>
      <td>1.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>6.5</td>
      <td>26</td>
      <td>5.5</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>



## with lags


```python
import lencoder.value_encoder_with_lags

df.value_encodigs_with_lags(
            'days', ['group'], ['value1', 'value2'],
            aggregations=['mean', 'sum'], lags=[1, 2])
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
      <th>days</th>
      <th>group</th>
      <th>days_group:value1:mean</th>
      <th>days_group:value1:sum</th>
      <th>days_group:value2:mean</th>
      <th>days_group:value2:sum</th>
      <th>days_group:value1:mean_1</th>
      <th>days_group:value1:sum_1</th>
      <th>days_group:value2:mean_1</th>
      <th>days_group:value2:sum_1</th>
      <th>days_group:value1:mean_2</th>
      <th>days_group:value1:sum_2</th>
      <th>days_group:value2:mean_2</th>
      <th>days_group:value2:sum_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>A</td>
      <td>0</td>
      <td>0</td>
      <td>0.5</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>B</td>
      <td>1</td>
      <td>2</td>
      <td>2.5</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>A</td>
      <td>2</td>
      <td>4</td>
      <td>4.5</td>
      <td>9</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>B</td>
      <td>3</td>
      <td>6</td>
      <td>6.5</td>
      <td>13</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>A</td>
      <td>4</td>
      <td>8</td>
      <td>0.5</td>
      <td>1</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>B</td>
      <td>5</td>
      <td>10</td>
      <td>2.5</td>
      <td>5</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>6.5</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.5</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>A</td>
      <td>6</td>
      <td>12</td>
      <td>4.5</td>
      <td>9</td>
      <td>4.0</td>
      <td>8.0</td>
      <td>0.5</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>4.5</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>B</td>
      <td>7</td>
      <td>14</td>
      <td>6.5</td>
      <td>13</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>6.5</td>
      <td>13.0</td>
    </tr>
  </tbody>
</table>
</div>


