# ESMC-300m Model Predictions

This repository contains scripts to run predictions using ESMC-300m based models. It supports models utilizing **Mean** or **Median** pooling strategies, as well as combined **Mean+Median** architectures.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yoel-marcu/esmc-predictor.git
   cd esmc-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Place your input FASTA file in the directory (or anywhere accessible).

```bash
python predict.py \
  --fasta inputs.fasta \
  --output results.csv \
  --weights_dir weights
```

### Arguments

- `--fasta`: Path to the sequences file.
- `--weights_dir`: Folder containing `.pth` model files (default: `weights/`).
- `--output`: Name of the result file (CSV).
- `--batch_size`: Batch size for ESMC embedding (adjust based on GPU memory).
- `--chunk_size`: Number of sequences to process in memory at once (default: 100).
