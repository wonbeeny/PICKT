# Online Data Usage

This directory stores **HME**‑related online datasets used for PICKT. <br>
The raw data is hosted on Hugging Face Datasets Hub and can be downloaded after granting appropriate access rights. <br><br>

Hosted dataset repo:
- https://huggingface.co/datasets/wonbeeny/milkt-hme-v1

## 1. Access rights from Hugging Face host

This dataset is hosted under the namespace `wonbeeny/milkt-hme-v1` and requires **manual access approval** from the host.<br><br>

To request access:

1. Go to the dataset page:
   - https://huggingface.co/datasets/wonbeeny/milkt-hme-v1
2. Click the button `Request access`.
3. Fill in the requested information (affiliation, intended use, etc.).
4. Wait until the dataset owner (host) accepts your access request.

Once the request is accepted, you will see the files and can start downloading the data via your Hugging Face account.

---

## 2. Download prerequisites

Before running the download script, you need:

- A **Hugging Face account**  
- Correct **HF access token (`HF_TOKEN`)** in your environment  
  - Create one at: https://huggingface.co/settings/tokens  
  - Make sure it has at least **Read access to repos**.

After creating the token, set it as an environment variable or configure `huggingface_hub`:

```bash
export HF_TOKEN=hf_xxxxxxxxxx
```

or (in a notebook, using `notebook_login()`):

```python
from huggingface_hub import notebook_login
notebook_login()
```

---

## 3. Download datasets to `./PICKT/data/Online/`

Use the notebook script:

`./PICKT/data/download_datasets.ipynb`

This notebook will:

- Download the following subsets from `wonbeeny/milkt-hme-v1`:
  - `MilkT-Question_Text`  
  - `MilkT-RQ1-Solved_History`  
  - `MilkT-Concept_List`  
  - `MilkT-Knowledge_Map`
  - `Total-Question_Meta`
- Save it as a local CSV file under `./PICKT/data/Online/`.

#### Files that will be downloaded

After successful execution, the `./PICKT/data/Online/` directory will contain:

- `MilkT-Question_Text.csv`
- `MilkT-RQ1-Solved_History.csv`  
- `MilkT-Concept_List.csv`  
- `MilkT-Knowledge_Map.csv`
- `Total-Question_Meta.csv`

Each file corresponds to one of the four MilkT‑related online datasets used in the PICKT pipeline.

---

## 4. How to verify

To check that the file is successfully downloaded:

```bash
ls ./PICKT/data/Online/
```

You should see the four CSV files listed above. <br> <br>

If you encounter `DatasetGenerationError` or schema mismatch, make sure the Hugging Face dataset schema has not changed and re‑run the notebook after clearing the cache:

```python
from datasets import Dataset
Dataset.cleanup_cache_files()
```

---

## 5. Summary

- Grant **access rights** on Hugging Face (`wonbeeny/milkt-hme-v1`).  
- Set up **HF access token** and `huggingface_hub` in your environment.  
- Run `./PICKT/data/download_datasets.ipynb` to download the data into `./PICKT/data/Online/`.


---

## License

These online datasets are provided for reproducibility of the PICKT paper experiments. <br>
They are released under the **Creative Commons Attribution‑NonCommercial 4.0 International (CC‑BY‑NC‑4.0)** license.

[![CC BY-NC 4.0 license](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

Under this license:

- You must **clearly credit the original authors and dataset** whenever you use, share, or modify the data.  
- You may use and redistribute this dataset for **non‑commercial** purposes, including research, education, and internal evaluation.  
- **Commercial use, resale, sublicensing, or embedding in paid products is not allowed** without prior written permission from the dataset owner.  
- Any modifications to the data should be clearly marked as such and not suggest that the original authors endorse your use.

For more details, see the official license deed:

- https://creativecommons.org/licenses/by-nc/4.0/