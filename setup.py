from setuptools import setup, find_packages

setup(
    name='pqkmeans',
    version='0.0.1',
    description='PQk-means',
    author='Yusuke Matsui, Keisuke OGAKI',
    author_email='keisuke_ogaki@dwango.co.jp',
    url='',
    packages=find_packages('.'),
    data_files=[(".", ["_pqkmeans.so"])],
    include_package_data=True,
    install_requires=[],
    test_suite='tests',
    entry_points='',
    zip_safe=False
)
