# e-SNLI Dataset

This directory is intentionally left empty in the repository.

Please download the e-SNLI dataset files manually using the following commands:

```bash
cd path/to/larev-research/dataset/esnli
curl -L -o esnli_train.csv \
  https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_train_1.csv
curl -L -o esnli_val.csv \
  https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_dev.csv
curl -L -o esnli_test.csv \
  https://raw.githubusercontent.com/OanaMariaCamburu/e-SNLI/master/dataset/esnli_test.csv
```

After downloading, this directory should contain:

- esnli_train.csv
- esnli_val.csv
- esnli_test.csv