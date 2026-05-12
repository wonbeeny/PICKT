# PICKT: **P**ractical **I**ntegrated **C**ross-consistent **K**nowledge Tracing

Official implementation of **Practical Integrated Cross-consistent Knowledge Tracing (PICKT)** for the paper **"Enhancing Knowledge Tracing Robustness for New Question Cold Start in Intelligent Tutoring Systems."** This repository is intended for research code release accompanying a manuscript currently under review at *Computers and Education: Artificial Intelligence (CAEAI)*.

> **Review status notice**  
> The associated manuscript is currently under review. To preserve anonymous peer review, this repository may be updated after the review process with additional documentation, trained checkpoints, and reproducibility details. Please cite the repository rather than the unpublished manuscript until a formal publication record becomes available.


## Overview

PICKT is a practical Knowledge Tracing (KT) model for Intelligent Tutoring System (ITS) services. The model estimates learners' knowledge states by integrating diverse educational signals while preserving the characteristics of each variable.


## Motivation

Knowledge Tracing models often degrade when they encounter **new-question cold start** conditions, where newly introduced items have limited or no historical interaction data. PICKT is designed to improve robustness under this realistic setting by encoding multiple sources of information rather than relying only on question-level interaction history. <br><br>

This design is especially relevant for real-world ITS platforms where:

- new questions are continuously added,
- learner interaction histories are sparse for some items, and
- prediction models must remain stable under operational constraints.


## Installation

There are two ways to install PICKT.

#### 1. Install via pip
```console
pip install git+https://github.com/wonbeeny/PICKT.git
```

#### 2. Install from source
```console
git clone https://github.com/wonbeeny/PICKT.git
cd PICKT
pip install .
```


## Requirements

This code was developed with:
- CUDA 12.4
- Driver 550.163.01
- Python 3.10.16

Create and activate the conda environment from the YAML file:
```console
conda env update --name pickt --file requirements/environment.yaml
conda activate pickt
```

Or create a new environment and install dependencies from `requirements.txt`:
```console
conda create -n <your_env_name> python==3.10.16 -y
conda activate <your_env_name>
pip install -r ./requirements/requirements.txt
```

> PyTorch should be installed separately according to your CUDA and driver versions.


## Getting started

The following files are required to run PICKT. As an example, download the DBE‑KT22 dataset as instructed in `./PICKT/data/DBE-KT22/README.md`. Then preprocess the raw data by running:

```bash
sh ./PICKT/examples/preprocess/DBE-KT22/all_run.sh
```

After preprocessing, make sure the following files are generated:

- `data_args.json`
- `km_data.json`
- `train_dataset.json`
- `valid_dataset.json`

If needed, you can also generate `test_dataset.json` and `pred_dataset.json` using the same pipeline. <br><br>

After the data preparation, run the model training script:

1. Check the YAML configuration files in `./PICKT/examples/config/DBE-KT22` and select the desired model.
2. Update `./PICKT/examples/main/train.sh` to point to the selected YAML file.
3. Run:
    ```bash
    sh ./PICKT/examples/main/train.sh
    ```

This will start model training and evaluation on the processed DBE‑KT22 dataset.


## Reproducibility

To reproduce the results in the paper, follow the instructions for each research question below.

#### RQ1

The reproduction steps for **RQ1** are the same as those in the **Getting Started** section, except that the **Online** dataset should be used instead of **DBE-KT22**.

1. Download the **Our** dataset by running `./PICKT/data/download_datasets.ipynb`.
2. Preprocess the raw data by running:
    ```bash
    sh ./PICKT/examples/preprocess/Online/all_run.sh
    ```
4. After preprocessing, make sure the following files are generated:
   - `data_args.json`
   - `km_data.json`
   - `train_dataset.json`
   - `valid_dataset.json`
5. Check the YAML configuration files under `./PICKT/examples/config/Online` and select the desired model.
6. Update `./PICKT/examples/main/train.sh` to point to the selected YAML file.
7. Run the training script:
   ```bash
   sh ./PICKT/examples/main/train.sh
   ```

This will start model training and evaluation on the processed Online dataset.

#### RQ2

The reproduction code for **RQ2** will be provided in a separate branch: `repro-rq2`. <br><br>

Please switch to that branch and follow the instructions in its `README.md`.

```bash
git checkout repro-rq2
```



## Citation

If this repository is useful for your research, please use a repository citation for now.

```bibtex
@misc{pickt_code,
  title={PICKT: Practical Integrated Cross-consistent Knowledge Tracing},
  author={Anonymous},
  year={2026},
  howpublished={GitHub repository},
  note={Code release for the manuscript "Enhancing Knowledge Tracing Robustness for New Question Cold Start in Intelligent Tutoring Systems" under review}
}
```

After the review process, the citation information will be updated with the final bibliographic record if the paper is accepted.

## License

This repository is released under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY‑NC 4.0)** license.  <br>
You are free to share and adapt the code for non‑commercial purposes, provided you give appropriate credit to the original work.<br><br>

For full details, see:  
[https://creativecommons.org/licenses/by-nc/4.0/](https://creativecommons.org/licenses/by-nc/4.0/)