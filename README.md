# PQk-means

[PQk-means [Matsui, Ogaki, Yamasaki, and Aizawa, ACMMM 17]](http://yusukematsui.me/project/pqkmeans/pqkmeans.html) is a Python library for efficient clustering of large-scale data.
By first compressing input vectors into short product-quantized (PQ) codes,
PQk-means achieves fast and memory-efficient clustering, even for
high-dimensional vectors.
Similar to k-means, PQk-means repeats the assignment and update steps,
both of which can be performed in the PQ-code domain.



For a comparison, we provide the ITQ encoding for the binary conversion and 
[Binary k-means [Gong+, CVPR 15]](http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Gong_Web_Scale_Photo_2015_CVPR_paper.html) for the clustering of binary codes.

The library is written in C++ for the main algorithm with wrappers for Python.
All encoding/clustering codes are compatible with scikit-learn.

## Summary of features
- Approximation of k-means
- Tens to hundreds of times faster than k-means
- Tens to hundreds of times more memory efficient than k-means
- Compatible with scikit-learn
- Portable; one-line installation

## Installation
### Requisites
- CMake
    - `brew install cmake` for OS X
    - `sudo apt install cmake` for Ubuntu
    
### Build & install
You can install the library by one line
```
pip install git+https://github.com/DwangoMediaVillage/pqkmeans.git
```
Or, equivalently, you can build and install the library one by one
```
git clone https://github.com/DwangoMediaVillage/pqkmeans.git
cd pqkmeans
git submodule init
git submodule update
pip install -r requirements.txt
python setup.py build
python setup.py install
```
## Run samples


```
# with aritficial data
python bin/run_experiment.py --dataset artificial --algorithm bkmeans pqkmeans --k 100
# with texmex dataset (http://corpus-texmex.irisa.fr/)
python bin/run_experiment.py --dataset siftsmall --algorithm bkmeans pqkmeans --k 100
```

## Test
```
python setup.py test
```



## Usage
### For PQk-means

```python
import pqkmeans
import numpy as np
X = np.random.random((1000, 100)) # 100 dimentional 1000 samples

# Train a PQ encoder
encoder = pqkmeans.encoder.PQEncoder(num_subdim = 2)
encoder.fit(X)

# Convert input vectors to PQ codes
X_pqcode = encoder.transform(X)

# Run clustering
kmeans = pqkmeans.clustering.PQKMeans(encoder=encoder, k=5)
clusterd = kmeans.fit_predict(X_pqcode)
```
Then, `clusterd[0]` is the id of assigned centroid for the first input code (`X_pqcode[0]`).

### For Bk-means

In almost the same manner as PQk-means,

```python
import pqkmeans
import numpy as np
X = np.random.random((1000, 100)) # 100 dimentional 1000 samples

# Train a ITQ binary encoder
encoder = pqkmeans.encoder.ITQEncoder(num_bit=32)
encoder.fit(X)

# Convert input vectors to binary codes
X_itq = encoder.transform(X)

# Run clustering
kmeans = pqkmeans.clustering.BKMeans(k=5, input_dim=32)
clustered = kmeans.fit_predict(X_itq)
```

## Note
- This repository contains the re-implemented version of the PQk-means with the Python interface. There can be the difference between this repo and the pure c++ implementation used in the paper.


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

## Links
- [Paper](https://arxiv.org/abs/1709.03708)
- [Project page](http://yusukematsui.me/project/pqkmeans/pqkmeans.html)


## Todo
- Evaluation script for billion-scale data
- Nearest neighbor search with PQTable
- References
