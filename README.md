# Fake News Detection: XGBoost & LSTM Models

## Overview

This project implements and evaluates two machine learning models for binary classification of news articles as **fake (0) or real (1)**:

1. **XGBoost**: TF-IDF feature extraction + gradient boosting
2. **LSTM**: Tokenizer + embedding + recurrent neural network

The models are trained on the **WELFake dataset** (72,112 articles) with an 80/20 stratified split and evaluated on:
- **WELFake validation set (20%)** — in-domain performance
- **Recovery news dataset (2,029 articles)** — out-of-domain generalization

---

## System Requirements

- **Python 3.10+**
- **RAM**: 4GB minimum (8GB+ recommended for LSTM training)
- **GPU** (optional): CUDA 11.8+ for TensorFlow GPU acceleration

---

## Quick Start: Reproduce All Results

### 1. Clone Repository & Navigate
```bash
cd path/to/fake-news-detection
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Models (Generate All Outputs)
```bash
python src/run_tommy_models.py
```

This command creates the `results/` folder if it does not exist and generates all metrics/plots/report files automatically.

**Execution time**: ~5-15 minutes (depending on hardware)

### 5. Review Results
```bash
# Metrics and configurations
cat results/direct_metrics.json

# Detailed analysis
cat results/REPORT.txt

# View plots
# Open these in your preferred image viewer:
# - results/xgboost_model_metrics.png
# - results/lstm_model_metrics.png
```

---

## Detailed Setup Instructions

### Step 1: Verify Python Installation
```bash
python --version  # Should be 3.10 or higher
```

### Step 2: Install Package Dependencies
```bash
pip install -r requirements.txt
```

**Required packages**:
- `numpy==2.3.2` — Numerical computing
- `pandas==2.3.2` — Data manipulation
- `scikit-learn==1.7.1` — Machine learning (TF-IDF, metrics)
- `tensorflow==2.17.0` — Deep learning (LSTM, Keras)
- `keras==3.11.3` — Neural network API
- `xgboost==2.0.3` — Gradient boosting
- `matplotlib==3.10.7` — Plotting
- `seaborn==0.13.2` — Statistical visualization

### Step 3: Verify Data Files
Ensure the following datasets are present in one of these folders at project root:
- `News_Dataset/` (supported)
- `news_datasets/` (supported)

The dataset files are intentionally not tracked in this repository due GitHub file-size limits. Download them from:
- Recovery dataset: https://drive.google.com/file/d/1ck63CyypYRx3coXvL668xBPWb0sR9XQM/view
- WELFake dataset: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Required files:
```
News_Dataset/
  ├── WELFake_Dataset.csv        (72,112 rows, columns: title, text, label)
  └── recovery-news-data.csv      (2,029 rows, columns: title, body_text, reliability)
```

**Expected structure**:
- WELFake: label = 0 (fake) or 1 (real)
- Recovery: reliability = 0 (unreliable) or 1 (reliable)

### Step 4: Run Full Pipeline
```bash
python src/run_tommy_models.py \
  --max-features 25000 \
  --ngram-max 1 \
  --max-words 20000 \
  --max-len 200 \
  --lstm-epochs 6 \
  --lstm-batch-size 96
```

---

## Configuration Parameters

### TF-IDF Vectorizer (XGBoost)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-features` | 25000 | Maximum number of features extracted from text |
| `--ngram-max` | 1 | Maximum n-gram size (1=unigrams only) |

### LSTM Model
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-words` | 20000 | Vocabulary size for tokenizer |
| `--max-len` | 200 | Sequence padding length |
| `--lstm-epochs` | 6 | Number of training epochs |
| `--lstm-batch-size` | 96 | Batch size during training |

### Preprocessing
| Flag | Effect |
|------|--------|
| `--disable-keyword-removal` | Skip removing source/publisher keywords (reuters, cnn, etc.) |
| `--extra-blocked-terms [TERMS...]` | Additional terms to remove during text normalization |

### Example: Custom Configuration
```bash
python src/run_tommy_models.py \
  --max-features 30000 \
  --ngram-max 2 \
  --lstm-epochs 8 \
  --disable-keyword-removal
```

---

## Output Files

After running the pipeline, a `results/` directory is generated (if missing) and contains:

### 1. **direct_metrics.json**
Complete metrics and configurations in JSON format:
```json
{
  "train_setup": { ... },
  "dataset_sizes": { ... },
  "label_balance": { ... },
  "models": {
    "xgboost": {
      "welfake_validation_20pct": { accuracy, precision, recall, f1, confusion_matrix },
      "recovery_external_test": { ... },
      "config": { vectorizer, model hyperparameters }
    },
    "lstm": { ... }
  }
}
```

### 2. **REPORT.txt**
Human-readable analysis report covering:
- Problem statement and motivation
- Dataset description and preprocessing steps
- Model architectures and rationale
- Experimental setup
- Performance results (table format)
- Overfitting analysis and root causes
- Limitations and recommendations

### 3. **xgboost_model_metrics.png**
Bar plot comparing:
- Accuracy, Precision, Recall, F1 scores
- Side-by-side: WELFake validation vs Recovery test

### 4. **lstm_model_metrics.png**
Same format as XGBoost plot for easy comparison

---

## Model Configurations

### XGBoost

**Feature Extraction (TF-IDF)**:
- Vocabulary: 25,000 most frequent terms
- N-grams: Unigrams (1-word features)
- Sublinear scaling: Yes
- Min document frequency: 5
- Max document frequency: 90%
- Stop words: English (removed)

**Model Hyperparameters**:
- Estimators: 180 trees
- Learning rate: 0.05
- Max depth: 3
- Subsample: 0.65
- Min child weight: 5
- Regularization: L1=0.5, L2=2.0

### LSTM

**Tokenizer**:
- Vocabulary: 20,000 words
- Sequence length: 200 tokens (padded/truncated)
- Out-of-vocabulary handling: `<OOV>` token

**Architecture**:
```
Input (200 tokens)
  ↓
Embedding (96 dimensions)
  ↓
SpatialDropout1D (0.25)
  ↓
Bidirectional LSTM (48 units, dropout=0.3)
  ↓
Dense (32 units, ReLU, L2 regularization=1e-4)
  ↓
Dropout (0.35)
  ↓
Dense (1 unit, Sigmoid) → Binary output
```

**Training**:
- Optimizer: Adam (learning rate=8e-4, clip norm=1.0)
- Loss: Binary crossentropy
- Epochs: 6 (with early stopping if no improvement for 2 epochs)
- Batch size: 96
- Class weights: Balanced (inverse of class frequencies)
- Learning rate reduction: 0.5× if validation loss plateaus

---

## Key Results & Findings

### Expected Performance Summary

| Model | Dataset | Accuracy | Precision | Recall | F1 |
|-------|---------|----------|-----------|--------|-----|
| **XGBoost** | WELFake (validation) | ~97% | ~97% | ~96% | ~97% |
| **XGBoost** | Recovery (external) | ~73% | ~72% | ~77% | ~74% |
| **LSTM** | WELFake (validation) | ~85% | ~84% | ~86% | ~85% |
| **LSTM** | Recovery (external) | ~82% | ~81% | ~83% | ~82% |

### Key Insights

1. **XGBoost Overfitting**: Strong in-domain (97% F1) but degrades significantly on Recovery (74% F1)
   - Likely captures dataset-specific artifacts and publisher markers
   - More expressive model with larger feature space

2. **LSTM Generalization**: More modest in-domain (85% F1) but maintains better transfer (82% F1)
   - Lower capacity acts as regularization
   - Learns sequential patterns rather than bag-of-words artifacts

3. **Dataset Shift**:
   - WELFake: ~49% fake, 51% real
   - Recovery: ~33% fake, 67% real
   - Different class distributions affect both models

---

## Preprocessing Pipeline

1. **Text Normalization**:
   - Convert to lowercase
   - Remove URLs and email addresses
   - Remove non-alphanumeric characters
   - Remove extra whitespace

2. **Keyword Filtering** (default: enabled):
   - Remove source/publisher keywords: reuters, bbc, cnn, nyt, washington post, etc.
   - Prevents models from exploiting dataset-specific markers

3. **Feature Engineering**:
   - Combine title + text into single input
   - Remove empty records
   - Stratified 80/20 split on training data

---

## Reproducibility & Randomness

To ensure reproducible results, the following are fixed:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

**Note**: Some minor variation may occur due to:
- TensorFlow's non-deterministic GPU operations
- Floating-point arithmetic differences across hardware
- XGBoost's multi-threading behavior

For strict reproducibility on CPU-only environments, disable GPU:
```bash
export CUDA_VISIBLE_DEVICES=-1
python src/run_tommy_models.py
```

---

## Troubleshooting

### Error: `ModuleNotFoundError: No module named 'xgboost'`
```bash
pip install xgboost==2.0.3
```

### Error: `ModuleNotFoundError: No module named 'tensorflow'`
```bash
pip install tensorflow==2.17.0
```

### Error: `FileNotFoundError: News_Dataset/WELFake_Dataset.csv`
Ensure data files exist at project root in either `News_Dataset/` or `news_datasets/`.

### LSTM Training is Very Slow
- Ensure you're using GPU (check: `nvidia-smi`)
- Reduce `--lstm-batch-size` to 32 if out of memory
- Reduce `--lstm-epochs` to 3 for quick test run

### Plot Files Not Generated
Check for matplotlib backend issues:
```bash
python -c "import matplotlib; print(matplotlib.get_backend())"
```

---

## Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── News_Dataset/ or news_datasets/
│   ├── WELFake_Dataset.csv
│   └── recovery-news-data.csv
├── src/
│   ├── run_tommy_models.py            # Main entry point
│   ├── data/
│   │   ├── __init__.py
│   │   └── datasets.py                # Data loading & preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py              # LSTM architecture & training
│   │   ├── xgboost_model.py           # XGBoost pipeline
│   │   └── metrics.py                 # Evaluation metrics
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_preprocessing.py      # Text normalization
│   │   └── keywords.py                # Source keyword removal
│   └── results_utils/
│       ├── __init__.py
│       ├── plotting.py                # Generate metric plots
│       └── reporting.py               # Generate analysis report
└── results/
    ├── direct_metrics.json            # Model metrics & configs
    ├── xgboost_model_metrics.png      # XGBoost performance plot
    ├── lstm_model_metrics.png         # LSTM performance plot
    └── REPORT.txt                     # Detailed analysis report
```

---

## Citation

If you use this project, please cite:

```bibtex
@misc{fakenews_detection_2025,
  title={Fake News Detection: XGBoost vs LSTM Comparison},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/fake-news-detection}}
}
```

---

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

---

## Contact & Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Review `results/REPORT.txt` for analysis details
3. Inspect `results/direct_metrics.json` for raw metrics
4. Open an issue on GitHub with:
   - Full error message and traceback
   - Python version and OS
   - Output of `pip list | grep -E "(xgboost|tensorflow|keras|scikit)"`
