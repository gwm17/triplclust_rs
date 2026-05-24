# `triplclust_py` Documentation

`triplclust_py` is the Python package containing bindings for using [`triplclust_rs`](https://crates.io/crates/triplclust_rs)
from Python. It is an implementation of the `triplclust` algorithm developed by C. Dalitz
described in the paper

> C. Dalitz, J. Wilberg, L. Aymans: "TriplClust: An Algorithm for Curve Detection in
> 3D Point Clouds." Image Processing Online 9, pp. 26-46 (2019).
> https://doi.org/10.5201/ipol.2019.234

and implemented in C++ as the [`triplclust`](https://github.com/cdalitz/triplclust). This
documentation will be focused on how to install and use the Python bindings in your project,
rather than the details of the algorithm. You can see the paper for details on the algorithm,
as well as the `triplclust_rs` documentation for any changes relative to the original
implementation.

## Requirements

Python >= 3.8 for most Linux, MacOS
Python >= 3.10 for Windows 11.

## Install

`triplclust_py` can be installed using `pip` as

```bash
pip install triplclust_py
```

Most likely, there is a wheel built for your platform, and installation should be very smooth.
If there is not, you will also need the Rust compiler tool chain, which you can find 
[here](https://rust-lang.org/).

!!! warning
    Machines running Windows 11 + ARM are not currently supported

## Usage

Generally, using `triplclust_py` is very simple. Your code will look something like:

```python
from triplclust_py import smooth_pointcloud, calculate_dnn, triplet_clustering

# data in this case is a  Nx(3 or greater) numpy array representing a point cloud
# The first 3 columns should be (x,y,z) coordinates
dnn = calculate_dnn(data)
smooth_cloud = smooth_pointcloud(data, dnn, None)
cluster_labels, unique_labels = triplet_clustering(
    smooth_cloud, 19, 2, 0.03, dnn, None, None, 5, "single"
)
# cluster_labels is a length N numpy array of integer labels for each point
# in the point cloud. unique_labels is an array containing the unique label values

# Do some stuff here with the results...
```

For details on the parameters and return values see the [API](api.md) documentation.

## Performance considerations

Currently, `triplclust_rs` is benchmarked, however, `triplclust_py` is not. Eventually,
this will be setup to ensure `triplclust_py` does not impose significant overhead.
