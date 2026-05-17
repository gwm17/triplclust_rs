# triplclust_rs

Rust implementation of the [`triplclust`](https://github.com/cdalitz/triplclust) 
hierarchical clustering algorithm outlined in

> C. Dalitz, J. Wilberg, L. Aymans: "TriplClust: An Algorithm for Curve Detection in
> 3D Point Clouds." Image Processing Online 9, pp. 26-46 (2019).
> https://doi.org/10.5201/ipol.2019.234

used by AT-TPC, with Python bindings using pyo3.

## Installation

Eventually get this hosted on PyPI but until then...

### From source

Requires the [Rust](https://rust-lang.org) compiler tool chain.

Download the repository from GitHub and then create a Python virtual environment.
Activate the virtual environment and install maturin using

```bash
pip install maturin
```

Then from the top level of the repository run

```bash
maturin develop
```

This will compile `triplclust_rs` and install the Python bindings `triplclust_py`.

## Usage

Docs in progress.

## Benchmarks

To run the benchmarks run

```bash
cargo bench
```
