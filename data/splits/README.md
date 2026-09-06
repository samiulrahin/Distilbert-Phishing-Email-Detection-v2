# Reproducible Splits

`train.csv`, `validation.csv`, and `test.csv` are generated with a fixed stratified seed of 42. Raw and split text is excluded from version control because emails may contain identifying header fields even when the source corpus is public.

The generated `data_quality_report.json` records row removals, class counts, and split sizes without storing message content.

