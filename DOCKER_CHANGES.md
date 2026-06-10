# Docker Configuration Changes — Code Flow Reorganization

## Summary
The project code has been reorganized with Python source files moved to `src/` directory and data files moved to `data/` directory. The Docker configuration has been updated accordingly.

## Changes Made

### 1. **Dockerfile — Updated COPY Commands**
**Previous (Incorrect):**
```dockerfile
COPY inference.py .
COPY utils.py .
COPY id2label.json .
```

**Updated (Correct):**
```dockerfile
COPY src/inference.py .
COPY src/utils.py .
COPY data/id2label.json .
```

**Reason:** Source files are now in `src/` directory and data files in `data/` directory. The Dockerfile must reference the correct paths.

---

### 2. **inference.py — Enhanced Label File Path Resolution**
**Added imports:**
```python
import os
```

**Added auto-detection logic:**
```python
# auto-resolve label file path (check multiple locations)
if not os.path.exists(args.labels):
    # try common locations
    alt_paths = [
        os.path.join(os.path.dirname(__file__), args.labels),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'id2label.json'),
        os.path.join(os.getcwd(), 'data', 'id2label.json'),
        os.path.join(os.getcwd(), args.labels),
    ]
    for alt_path in alt_paths:
        if os.path.exists(alt_path):
            args.labels = alt_path
            break
```

**Reason:** Allows the script to work seamlessly both:
- **In Docker:** Where `id2label.json` is in `/app` (copied to root)
- **Locally:** Where `id2label.json` is in `data/` directory
- **CLI:** When explicitly specified with `--labels` argument

---

## File Structure After Changes

```
group13-assignment-mlops/
├── Dockerfile                 # Updated ✓
├── requirements.txt
├── src/
│   ├── inference.py          # Enhanced path resolution ✓
│   └── utils.py
├── data/
│   ├── id2label.json         # Now referenced from data/ ✓
│   ├── sample_data.json
│   └── inference_result.json
└── notebooks/
    ├── prepare_data.ipynb
    ├── train_model.ipynb
    └── model_inference.ipynb
```

---

## Testing the Docker Configuration

### Build the Docker Image:
```bash
cd group13-assignment-mlops
docker build -t ag-news-classifier:latest .
```

### Run the Docker Container (Demo Mode):
```bash
docker run ag-news-classifier:latest
```

### Run with Custom Text:
```bash
docker run ag-news-classifier:latest \
  --text "Stock market rises on strong earnings"
```

### Run with Custom Model:
```bash
docker run ag-news-classifier:latest \
  --demo \
  --model "YuvarajK-g25ait2054/ag-news-distilbert"
```

---

## Verification Checklist

✓ **Dockerfile COPY paths corrected** — Now references `src/` for Python files  
✓ **Dockerfile COPY paths corrected** — Now references `data/` for JSON files  
✓ **inference.py enhanced** — Auto-detects label file across multiple locations  
✓ **Backward compatible** — CLI `--labels` argument still works for explicit paths  
✓ **Both execution modes supported** — Docker container and local development

---

## Impact Analysis

| Aspect | Impact | Status |
|--------|--------|--------|
| Docker Build | Files now copied from correct paths | ✓ Fixed |
| Runtime (Docker) | Label file found in `/app/id2label.json` | ✓ Works |
| Runtime (Local Dev) | Label file auto-detected in `data/` | ✓ Enhanced |
| CLI Interface | `--labels` argument still functional | ✓ Preserved |
| Module Imports | Python imports from `utils.py` work (both in same dir) | ✓ Correct |

---

## Notes

1. **Import Dependencies:** `inference.py` imports from `utils.py` using relative import (`from utils import ...`). Both files are copied to `/app` in Docker, so the import works correctly.

2. **Label File Path:** The enhanced path resolution in `inference.py` makes the script more robust and eliminates the need to manually specify `--labels data/id2label.json` when running locally.

3. **Docker Command Line:** The default CMD in Dockerfile runs the demo, but can be overridden:
   ```bash
   docker run ag-news-classifier:latest --text "Your text here"
   ```

---

## Related Files

- [Dockerfile](./Dockerfile)
- [src/inference.py](./src/inference.py)
- [src/utils.py](./src/utils.py)
- [data/id2label.json](./data/id2label.json)
- [requirements.txt](./requirements.txt)
