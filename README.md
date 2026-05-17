# triplclust_rs

Rust implementation of the [`triplclust`](https://github.com/cdalitz/triplclust) hierarchical clustering algorithm used by 
AT-TPC, with Python bindings using pyo3.

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
