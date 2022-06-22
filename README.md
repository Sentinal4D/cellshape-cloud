[![Python Version](https://img.shields.io/pypi/pyversions/cellshape-cloud.svg)](https://pypi.org/project/cellshape-cloud)
[![PyPI](https://img.shields.io/pypi/v/cellshape-cloud.svg)](https://pypi.org/project/cellshape-cloud)
[![Downloads](https://pepy.tech/badge/cellshape-cloud)](https://pepy.tech/project/cellshape-cloud)
[![Wheel](https://img.shields.io/pypi/wheel/cellshape-cloud.svg)](https://pypi.org/project/cellshape-cloud)
[![Development Status](https://img.shields.io/pypi/status/cellshape-cloud.svg)](https://github.com/Sentinal4D/cellshape-cloud)
[![Tests](https://img.shields.io/github/workflow/status/Sentinal4D/cellshape-cloud/tests)](
    https://github.com/Sentinal4D/cellshape-cloud/actions)
[![Coverage Status](https://coveralls.io/repos/github/Sentinal4D/cellshape-cloud/badge.svg?branch=master)](https://coveralls.io/github/Sentinal4D/cellshape-cloud?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<img src="https://github.com/DeVriesMatt/cellshape-cloud/blob/main/img/cellshape_cloud.png" 
     alt="Cellshape logo by Matt De Vries">
___
Cellshape-cloud is an easy-to-use tool to analyse the shapes of cells using deep learning and, in particular, graph-neural networks. The tool provides the ability to train popular graph-based autoencoders on point cloud data of 2D and 3D single cell masks as well as providing pre-trained networks for inference.



## To install
```bash
pip install cellshape-cloud
```

## Usage
```python
import torch
from cellshape_cloud import CloudAutoEncoder

model = CloudAutoEncoder(num_features=128, 
                         k=20,
                         encoder_type="dgcnn",
                         decoder_type="foldingnet")

points = torch.randn(1, 2048, 3)

recon, features = model(points)
```

## Parameters

- `num_features`: int.  
The size of the latent space of the autoencoder. If you have rectangular images, make sure your image size is the maximum of the width and height
- `k`: int.  
The number of neightbours to use in the k-nearest-neighbours graph construction.
- `encoder_type`: str.  
The type of encoder: 'foldingnet' or 'dgcnn'
- `decoder_type`: str.  
The type of decoder: 'foldingnet' or 'dgcnn'


## For developers
* Fork the repository
* Clone your fork
```bash
git clone https://github.com/USERNAME/cellshape-cloud 
```
* Install an editable version (`-e`) with the development requirements (`dev`)
```bash
cd cellshape-cloud
pip install -e .[dev] 
```
* To install pre-commit hooks to ensure formatting is correct:
```bash
pre-commit install
```

* To release a new version:

Firstly, update the version with bump2version (`bump2version patch`, 
`bump2version minor` or `bump2version major`). This will increment the 
package version (to a release candidate - e.g. `0.0.1rc0`) and tag the 
commit. Push this tag to GitHub to run the deployment workflow:

```bash
git push --follow-tags
```

Once the release candidate has been tested, the release version can be created with:

```bash
bump2version release
```

## References
[1] An Tao, 'Unsupervised Point Cloud Reconstruction for Classific Feature Learning', [GitHub Repo](https://github.com/AnTao97/UnsupervisedPointCloudReconstruction), 2020
