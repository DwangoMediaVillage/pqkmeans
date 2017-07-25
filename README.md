# pqkmeans-private

Build (For users)
------------------------

### On OSX

```
brew install cmake
brew install boost-python --with-python3
```

```
cmake .
make
python setup.py install
```

* If you want use custom python (like pyenv, virtualenv), you should set `PYTHON_LIBRARY` option
    * `cmake -DPYTHON_LIBRARY=/Users/keisuke_ogaki/.pyenv/versions/anaconda3-2.5.0/lib/libpython3.5m.dylib -DPYTHON_INCLUDE_DIRS=~/.pyenv/versions/anaconda3-2.5.0/include/python3.5m .`


Build (For developers)
------------------------

### On OSX

```
brew install cmake
brew install boost-python --with-python3
```

```
cmake .
make
```
