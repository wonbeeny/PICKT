# DBE-KT22

This directory contains the DBE-KT22 dataset files used in PICKT. <br>

The original data was downloaded from:
- https://dataverse.ada.edu.au/dataset.xhtml?persistentId=doi:10.26193/6DZWOH

## Original data

Only the following CSV files were moved from the downloaded dataset package:

- `KC_Relationships.csv`
- `KSs.csv`
- `Question_Choices.csv`
- `Question_KC_Relationships.csv`
- `Questions.csv`
- `Transaction.csv`

Place these files in:

```bash
./PICKT/data/DBE-KT22
```

## Preprocessed data

Preprocess the raw data by running:

```bash
sh ./PICKT/examples/preprocess/DBE-KT22/all_run.sh
```

Then preprocessed files are stored in:

```bash
./PICKT/data/DBE-KT22
```


## Notes

- The dataset is provided in website for reproducibility of the PICKT paper experiments.
- The dataset is used as provided for knowledge tracing experiments in PICKT.
