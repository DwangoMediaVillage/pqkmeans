# PQk-means

PQk-means is a Python library for efficient clustering of large-scale data.
By first compressing input vectors into short product-quantized (PQ) codes,
PQk-means achieves fast and memory-efficient clustering, even for
high-dimensional vectors.
Similar to k-means, PQk-means repeats the assignment and update steps,
both of which can be performed in the PQ-code domain.

For a comparison, we also provide the ITQ encoding for the binary conversion and 
[Binary k-means [Gong+, CVPR 15]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Gong_Web_Scale_Photo_2015_CVPR_paper.html) for the clustering of binary codes.

The library is written in C++ for the main algorithm with wrappers for Python.
All encoding/clustering codes are implemented based on the sklearn design.



## Install
### Requisites
- CMake
    - `brew install cmake` for OS X
    - `sudo apt-get install cmake` for Ubuntu

### Build & install
```
git clone https://github.com/DwangoMediaVillage/pqkmeans.git
git submodule init
git submodule update
pip install -r requirements.txt
python setup.py install
```

## Test
```
python setup.py test
```


## Usage
For PQk-means
```python
from pqkmeans encoder import *
from pqkmeans clustering import *
import numpy as np
# [todo] can we include a one line script to download siftsmall?
X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0])

# Train a PQ encoder
encoder = PQEncoder(num_dim = 2)
encoder.fit(X)

# Convert input vectors to PQ codes
X_pqcode = encoder.transform(X)

# Run clustering
kmeans = PQKmeans(encoder=encoder, k=5)
clusterd = kmeans.fit_predicted(X_pqcode)
```
Then, `clusterd[0]` is the id of assigned centroid for the first input code (`X_pqcode[0]`).

For Bk-means
```python
aaa
```


## Authors
- [Keisuke Ogaki](https://github.com/kogaki) designed the whole structure of the library, and implemented most of the Bk-means clustering
- [Yusuke Matsui](http://yusukematsui.me/) implemented most of the PQk-means clustering

## Reference

    @inproceedings{pqkmeans,
	    author = {Yusuke Matsui and Keisuke Ogaki and Toshihiko Yamasaki and Kiyoharu Aizawa},
	    title = {PQk-means: Billion-scale Clustering for Product-quantized Codes},
        booktitle = {ACM International Conference on Multimedia (ACMMM)},
        year = {2017},
    }
 

# pqkmeans-private

Build (For users)
------------------------

### On OSX

```
brew install cmake
```

```
python setup.py install
```


Build (For developers)
------------------------

### On OSX

```
brew install cmake
```

```
python setup.py develop
```

```
python run_sample.py
```


Test
------------------------

```
python setup.py test
```
