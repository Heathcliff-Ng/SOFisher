<div align="center">
<img src="https://github.com/Heathcliff-Ng/SOFisher/blob/main/fig/logo.jpg" width="150px">

**An RL framework for guiding spatial omics FOVs selection.**

---

[//]: # (<p align="center">)

[//]: # (  <a href="" target="_blank">Preprint</a>)

[//]: # (</p>)

</div>

<p>
SOFisher a reinforcement learning-based framework that harnesses the knowledge gained from the sequence of previously sampled FOVs to guide the selection of the next FOV position.
</p>

<p align="center">
  <img src="https://github.com/Heathcliff-Ng/SOFisher/blob/main/fig/pipeline.jpg" width="800px">
</p>

## Directory Structure and Descriptions

```plaintext
.
├── README.md                 # Project overview and descriptions
├── AD_sets_2cell             # AD dataset extracted for 2 cell experiments
├── AD_sets_Aging             # AD dataset extracted for aging experiments
├── dataset_Aging             # cell dataset for aging experiments
├── dataset_cortex            # cell dataset for cortex experiments
├── dataset_disease           # cell dataset for disease experiments
├── fig                       # logo figs
├── model                     # trained models
├── data                      # test results
├── env                       # envs for different settings
│   ├── env_aging.py
│   ├── env_cortex_2cell.py
│   ├── env_cortex.py
│   ├── env_dis_both.py
│   └── env_dis_tau.py
├── src                       # RL algorithms
│   ├── args.py
│   ├── dqn.py
│   └── replay_dqn.py
├── main_aging.py             # training function for aging experiments
├── main_cortex.py            # training function for cortex experiments
├── main_disease.py           # training function for disease experiments
├── visualize_aging.ipynb     # results validation for aging experiments
├── visualize_cortex.ipynb    # results validation for cortex experiments
└── visualize_disease.ipynb   # results validation for disease experiments
├── requirements.txt          # environment setup
├── LICENSE
```

## Installation

1. Create a conda environment:

```bash
conda create -n sofisher-env
conda activate sofisher-env
```
2. Clone the SOFisher repository to your local machine:
```bash
git clone https://github.com/Heathcliff-Ng/SOFisher.git
cd SOFisher
```
3. Install dependencies:
```bash
pip install -r requirements.txt
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
