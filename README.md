[![Python Version](https://img.shields.io/pypi/pyversions/brainglobe-heatmap.svg)](https://pypi.org/project/brainglobe-heatmap)
[![PyPI](https://img.shields.io/pypi/v/brainglobe-heatmap.svg)](https://pypi.org/project/brainglobe-heatmap)
[![Downloads](https://pepy.tech/badge/brainglobe-heatmap)](https://pepy.tech/project/brainglobe-heatmap)
[![Wheel](https://img.shields.io/pypi/wheel/brainglobe-heatmap.svg)](https://pypi.org/project/brainglobe-heatmap)
[![Development Status](https://img.shields.io/pypi/status/brainglobe-heatmap.svg)](https://github.com/brainglobe/brainglobe-heatmap)
[![Tests](https://img.shields.io/github/actions/workflow/status/brainglobe/brainglobe-heatmap/test_and_deploy.yml?branch=main)](https://github.com/brainglobe/brainglobe-heatmap/actions)
[![codecov](https://codecov.io/gh/brainglobe/brainglobe-heatmap/branch/main/graph/badge.svg?token=nx1lhNI7ox)](https://codecov.io/gh/brainglobe/brainglobe-heatmap)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://brainglobe.info/community/developers/index.html)
[![Twitter](https://img.shields.io/twitter/follow/brain_globe?style=social)](https://twitter.com/brain_globe)

# brainglobe-heatmap
`brainglobe-heatmap` allows you to create heatmaps, mapping scalar values for each brain region (e.g., number of labelled cells in each region) to a color and creating visualizations in 2D (using [matplotlib](https://matplotlib.org/) or 3D (using [brainrender](https://brainglobe.info/documentation/brainrender/index.html)).

<img width="947" alt="Hansen_2025_Fig1" src="https://github.com/user-attachments/assets/38e93939-aa3a-4f94-8edf-a6a470260de9" />


**2D heatmap generated using matplotlib - adapted from Fig 1. from [Hansen et al (2025)](https://doi.org/10.1101/2025.01.24.634803)**

<img width="700" alt="heatmap_3d" src="https://github.com/user-attachments/assets/06f634aa-351b-4399-b84f-01107805e80e" />

**3D heatmap generated using brainrender**

## Usage
For full documentation, please see the [BrainGlobe documentation](https://brainglobe.info/documentation/brainglobe-heatmap/index.html). Examples can be found in the [examples](https://github.com/brainglobe/brainglobe-heatmap/tree/main/examples) directory of this repository.

## Quickstart
### Installation
`pip install brainglobe-heatmap`

### 2D heatmap
```python

import brainglobe_heatmap as bgh

values = dict(  # scalar values for each region
    TH=1,
    RSP=0.2,
    AI=0.4,
    SS=-3,
    MO=2.6,
    PVZ=-4,
    LZ=-3,
    VIS=2,
    AUD=0.3,
    RHP=-0.2,
    STR=0.5,
    CB=0.5,
    FRP=-1.7,
    HIP=3,
    PA=-4,
)


f = bgh.Heatmap(
    values,
    position=5000,
    orientation="frontal",
    vmin=-5,
    vmax=3,
    format="2D",
).show()
```
### 3D heatmap
```python

import brainglobe_heatmap as bgh

values = dict(  # scalar values for each region
    TH=1,
    RSP=0.2,
    AI=0.4,
    SS=-3,
    MO=2.6,
    PVZ=-4,
    LZ=-3,
    VIS=2,
    AUD=0.3,
    RHP=-0.2,
    STR=0.5,
    CB=0.5,
    FRP=-1.7,
    HIP=3,
    PA=-4,
)


scene = bgh.Heatmap(
    values,
    position=(8000, 5000, 5000),
    orientation="frontal",
    thickness=1000,
    vmin=-5,
    vmax=3,
    format="3D",
).show()

```

## Seeking help or contributing
We are always happy to help users of our tools, and welcome any contributions. If you would like to get in contact with us for any reason, please see the [contact page of our website](https://brainglobe.info/contact.html).

## Citing `brainglobe-heatmap`
If you use `brainglobe-heatmap` in your work, please cite it as:

```
Federico Claudi, Adam Tyson, Luigi Petrucco, Mathieu Bourdenx, carlocastoldi, Rami Hamati, & Alessandro Felder. (2024). brainglobe/brainglobe-heatmap. Zenodo. https://doi.org/10.5281/zenodo.10375287
```

If you use `brainrender` via `brainglobe-heatmap` (i.e. for 3D visualisation), please also cite it:
```
Claudi, F., Tyson, A. L., Petrucco, L., Margrie, T.W., Portugues, R.,  Branco, T. (2021) "Visualizing anatomically registered data with Brainrender&quot; <i>eLife</i> 2021;10:e65751 [doi.org/10.7554/eLife.65751](https://doi.org/10.7554/eLife.65751)
```

BibTeX:

``` bibtex
@article{Claudi2021,
author = {Claudi, Federico and Tyson, Adam L. and Petrucco, Luigi and Margrie, Troy W. and Portugues, Ruben and Branco, Tiago},
doi = {10.7554/eLife.65751},
issn = {2050084X},
journal = {eLife},
pages = {1--16},
pmid = {33739286},
title = {{Visualizing anatomically registered data with brainrender}},
volume = {10},
year = {2021}
}
```
