# Model Artefacts

The pipeline creates:

- `tfidf-logistic-regression.joblib`: classical baseline.
- `distilbert-phishing-v2/`: final tokenizer, configuration, fine-tuned transformer weights, calibration, and selective decision policy.

The portable transfer package includes the final V2 checkpoint but excludes intermediate training checkpoints and the obsolete V1 checkpoint. Version 1 metrics and hashes remain under `evaluation/versioned/v1/` so the development history is preserved without several gigabytes of unnecessary files.
