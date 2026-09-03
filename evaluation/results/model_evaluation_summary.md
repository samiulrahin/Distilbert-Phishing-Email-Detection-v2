# Generated Model Evaluation Summary

All values below come from the untouched held-out test split. Each threshold was selected on validation data by maximum F1 before test evaluation.

| model                        |   accuracy |   precision |   recall |     f1 |   roc_auc |   threshold |   test_rows |
|:-----------------------------|-----------:|------------:|---------:|-------:|----------:|------------:|------------:|
| TF-IDF + Logistic Regression |     0.9855 |      0.9826 |   0.9786 | 0.9806 |    0.9989 |      0.5400 |        2627 |
| DistilBERT V1                |     0.9878 |      0.9847 |   0.9827 | 0.9837 |    0.9989 |      0.1000 |        2627 |
| DistilBERT V2                |     0.9855 |      0.9758 |   0.9857 | 0.9807 |    0.9988 |      0.1000 |        2627 |

These results describe this dataset split and experiment configuration; they are not production performance guarantees.
