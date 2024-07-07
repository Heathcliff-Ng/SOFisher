<div align="center">
<img src="https://github.com/Heathcliff-Ng/SOFisher/blob/main/fig/logo.jpg" width="150px">

**A Python package for the Scalale and accurate identification condition-relevant niches from spatial omics data.**

---

<p align="center">
  <a href="" target="_blank">Preprint</a>
</p>

</div>

Taichi is able to automatically identify condition-relevant niches, and offers the downstream analysis based on obtained niches.
</p>
<p align="center">
  <img src="https://github.com/Heathcliff-Ng/SOFisher/blob/main/fig/pipeline.jpg" width="800px">
</p>


## Installation

1. Create a conda environment
```bash
conda create -n sofisher-env
conda activate sofisher-env
```
2. Install the SOFisher dependency
```bash
mamba install squidpy scanpy -c conda-forge (squidpy == 1.3.0 for reproducing CCI in manuscript)
pip insall pygsp ipykernel
```
3. Install the MENDER for batch-free niches representation:
```bash
cd MENDER
python setup.py install
```

Install the `pysodb` for efficient download processed Anndata in h5ad format. (https://pysodb.readthedocs.io/en/latest/)



## Getting started


Please refer to the  
- Results validation for cortex experiments. [Tutorial][link-tutorial_1] 
- Results validation for aging experiments. [Tutorial][link-tutorial_2]
- Results validation for disease experiments. [Tutorial][link-tutorial_3]

## Contribution

If you found a bug or you want to propose a new feature, please use the [issue tracker][issue-tracker].

[issue-tracker]: https://github.com/Heathcliff-Ng/SOFisher/issues
[link-tutorial_1]: https://github.com/Heathcliff-Ng/SOFisher/blob/main/visualize_cortex.ipynb
[link-tutorial_2]: https://github.com/Heathcliff-Ng/SOFisher/blob/main/visualize_aging.ipynb
[link-tutorial_3]: https://github.com/Heathcliff-Ng/SOFisher/blob/main/visualize_disease.ipynb
