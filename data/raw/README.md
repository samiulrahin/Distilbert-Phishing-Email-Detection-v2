# Raw Data Boundary

Raw email data is intentionally excluded from version control. Run:

```powershell
.\.venv\Scripts\python.exe -m ml.download_dataset
```

The downloader records the exact URL, access time, byte count, and SHA-256 digest in `download_metadata.json`. Do not add personal Gmail data to this folder.

