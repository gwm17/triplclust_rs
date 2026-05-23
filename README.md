# triplclust_rs

![Rust CI](https://github.com/gwm17/triplclust_rs/actions/workflows/rust_build.yaml/badge.svg)
![Python CI](https://github.com/gwm17/triplclust_rs/actions/workflows/python_build.yaml/badge.svg)
[![PyPI - version](https://img.shields.io/pypi/v/triplclust_py)](https://pypi.org/projects/triplclust_py)
[![Crates.io - version](https://img.shields.io/crates/v/triplclust_rs)](https://crates.io/crates/triplclust_rs)
[![Rust Docs](https://img.shields.io/docsrs/triplclust_rs)](https://docs.rs/crates/triplclust_rs)

Rust implementation of the [`triplclust`](https://github.com/cdalitz/triplclust) 
hierarchical clustering algorithm outlined in

> C. Dalitz, J. Wilberg, L. Aymans: "TriplClust: An Algorithm for Curve Detection in
> 3D Point Clouds." Image Processing Online 9, pp. 26-46 (2019).
> https://doi.org/10.5201/ipol.2019.234

used by AT-TPC, with Python bindings using pyo3.

## Requirements

Python > 3.8 if creating Python bindings.
[Rust](https://rust-lang.org) compiler, if compiling from source.

## Basic Installation

### Python

To install the Python bindings, create a virtual environment, activate it, and then use

```bash
pip install triplclust_py
```

Most OS-architecture-python targets have a prebuilt wheel, so you should not need to
compile from source. Some caveats:

- windows11-aarch is not supported. [kiddo](https://crates.io/crates/kiddo) does not support
windows11-aarch through one of it's dependencies.
- Python prior to 3.8 is not supported on any platform. Python prior to 3.10 is not
supported on Windows.
- A wide range of linux distributions is covered via the manylinux spec. See the spec
for [details](https://github.com/pypa/manylinux) if you have questions.

### Rust

Requires the Rust compiler tool chain.

To add the Rust library as a dependency to your repository, use cargo

```bash
cargo add triplclust_rs
```

### From source

Requires the Rust compiler tool chain.

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

For a complete description of the library, see the Rust [documentation](https://docs.rs/crates/triplclust_rs).
For a high level overview and an outline of the Python bindings, see the Python documentation.

## Benchmarks

To run the benchmarks run

```bash
cargo bench
```
