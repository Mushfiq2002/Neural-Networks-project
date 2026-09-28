# Unsupervised Music Clustering with GTZAN Dataset

## 🎯 Project Overview

This project addresses music clustering tasks at three difficulty levels:

- **EASY**: MLP-VAE on MFCC features + KMeans clustering + t-SNE/UMAP visualization + Silhouette/Calinski-Harabasz metrics
- **MEDIUM**: Conv-VAE on log-mel spectrograms + multimodal fusion (audio + lyrics) + multiple clustering algorithms + Silhouette/Davies-Bouldin/ARI metrics
- **HARD**: CVAE/Beta-VAE + full multimodal fusion (audio + lyrics + genre) + comprehensive metrics (Silhouette, NMI, ARI, Purity) + reconstruction analysis + baseline comparisons

### Key Features

- ✅ Complete preprocessing pipeline for GTZAN (1000 tracks → 3000 clips)
- ✅ Whisper-based transcription for "lyrics" generation from audio
- ✅ Multi-modal feature extraction (MFCC, log-mel, TF-IDF)
- ✅ Multiple baselines (PCA+KMeans, Direct KMeans, AE, Spectral)
- ✅ Advanced VAE models (MLP-VAE, Conv-VAE, Beta-VAE, CVAE)
- ✅ Multimodal fusion strategies
- ✅ Comprehensive evaluation metrics and visualizations
- ✅ Reproducible pipeline with caching and resume capabilities

## 📁 Project Structure

```
project/
├── data/
│   ├── raw_gtzan/              # Place GTZAN dataset here
│   ├── clips_10s/              # Generated 10-second clips
│   ├── features/               # Extracted features
│   │   ├── mfcc_stats.npy
│   │   ├── logmel.npy
│   │   ├── lyrics_tfidf.npz
│   │   └── lyrics_tfidf_svd.npy
│   └── metadata.csv            # Clip metadata with splits
├── src/
│   ├── config.py               # Configuration management
│   ├── utils.py                # Utility functions
│   ├── preprocess.py           # Audio clip generation
│   ├── transcribe.py           # Whisper transcription
│   ├── features.py             # Feature extraction
│   ├── dataset.py              # PyTorch dataset
│   ├── clustering.py           # Clustering algorithms
│   ├── evaluation.py           # Evaluation metrics
│   ├── visualize.py            # Visualization functions
│   ├── baselines.py            # Baseline methods
│   ├── train.py                # VAE training pipeline
│   └── models/
│       ├── ae_mlp.py           # MLP Autoencoder
│       ├── vae_mlp.py          # MLP VAE (EASY)
│       ├── vae_conv.py         # Conv VAE (MEDIUM)
│       ├── vae_beta.py         # Beta-VAE (HARD)
│       └── cvae.py             # Conditional VAE (HARD)
├── notebooks/
│   ├── 00_exploration.ipynb    # Data exploration
│   └── 01_results_and_plots.ipynb  # Results visualization
├── results/
│   ├── metrics/                # Metrics CSV and latent features
│   ├── figures/                # Generated plots
│   └── checkpoints/            # Model checkpoints
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Setup

```bash
# Clone/setup the project
cd "D:\425 project"

# Install dependencies (CPU-only)
pip install -r requirements.txt

# For Colab (with GPU support for transcription/training):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faster-whisper sentence-transformers
```

### 2. Prepare GTZAN Dataset

Download the GTZAN dataset and place it in one of these structures:

```
data/raw_gtzan/genres_original/blues/*.wav
data/raw_gtzan/genres_original/classical/*.wav
...

OR

data/raw_gtzan/genres/blues/*.wav
data/raw_gtzan/genres/classical/*.wav
...

OR

data/raw_gtzan/blues/*.wav
data/raw_gtzan/classical/*.wav
...
```

Expected: 1000 audio files total (100 per genre × 10 genres).

### 3. Run the Pipeline

#### Step 1: Preprocess (Generate 10s clips)

```bash
python src/preprocess.py
```

This creates:
- 3000 clips (3 × 10s segments per track)
- `data/metadata.csv` with train/val/test splits
- Clips saved to `data/clips_10s/`

**Outputs:**
- `data/clips_10s/` - 3000 WAV files
- `data/metadata.csv` - Metadata with splits

**Time:** ~5-10 minutes (depends on ffmpeg availability)

#### Step 2: Extract Audio Features

```bash
python src/features.py --audio_only
```

This extracts:
- MFCC statistics (40-dim per clip)
- Log-mel spectrograms (128 × 431)

**Outputs:**
- `data/features/mfcc_stats.npy` (3000, 40)
- `data/features/logmel.npy` (3000, 128, 431)

**Time:** ~10-15 minutes

#### Step 3: Run Baselines

```bash
python src/baselines.py --visualize
```

**Outputs:**
- `results/metrics/metrics.csv` - Baseline results
- `results/figures/tsne_*.png` - Visualizations
- `results/metrics/confusion_*.csv` - Confusion matrices

**Time:** ~5-10 minutes

#### Step 4: Train EASY (MLP-VAE)

```bash
python src/train.py --mode vae_mlp --epochs 50 --device cpu
```

**For Colab (GPU):**
```bash
python src/train.py --mode vae_mlp --epochs 50 --device cuda
```

**Outputs:**
- `results/checkpoints/vae_mlp.pt` - Model checkpoint
- `results/metrics/latents_vae_mlp.npy` - Latent features
- `results/figures/tsne_vae_mlp.png` - t-SNE plot
- Updated `results/metrics/metrics.csv`

**Time:** ~5-10 minutes (CPU), ~1-2 minutes (GPU)

#### Step 5: Transcribe Audio (for MEDIUM/HARD)

**Run on Colab with GPU for speed:**

```bash
# On Colab (GPU):
python src/transcribe.py --model_size tiny --device cuda --compute_type float16

# On local CPU (slower):
python src/transcribe.py --model_size tiny --device cpu --compute_type int8
```

This adds `lyrics_text` column to `data/metadata.csv`.

**Time:** 
- CPU: ~2-4 hours for 3000 clips with `tiny` model
- GPU (Colab): ~15-30 minutes

**Note:** You can test with `--max_clips 100` first.

#### Step 6: Extract Text Features

```bash
python src/features.py --text_only
```

**Outputs:**
- `data/features/lyrics_tfidf.npz` - TF-IDF features
- `data/features/lyrics_tfidf_svd.npy` - Reduced TF-IDF

**Time:** ~1-2 minutes

#### Step 7: Train MEDIUM (Conv-VAE + Multimodal)

```bash
python src/train.py --mode vae_conv --epochs 50 --device cpu --multimodal
```

**Outputs:**
- `results/checkpoints/vae_conv.pt`
- `results/figures/tsne_vae_conv.png`
- `results/figures/recon_spectrograms_vae_conv.png`
- Metrics for both KMeans and Agglomerative clustering

**Time:** ~15-20 minutes (CPU), ~3-5 minutes (GPU)

#### Step 8: Train HARD (CVAE with Full Fusion)

```bash
python src/train.py --mode cvae --epochs 50 --device cpu
```

This trains a Conditional VAE with genre conditioning and fuses:
- Audio latent features from CVAE
- Lyrics TF-IDF embeddings
- Genre one-hot vectors

**Outputs:**
- `results/checkpoints/cvae.pt`
- `results/figures/tsne_cvae.png`
- `results/figures/recon_error_cvae.png`
- Comprehensive metrics (Silhouette, NMI, ARI, Purity)

**Time:** ~10-15 minutes (CPU), ~2-3 minutes (GPU)

## 📊 Evaluation Metrics

### Unsupervised Metrics
- **Silhouette Score** [-1, 1]: Cluster cohesion (higher is better)
- **Calinski-Harabasz Index**: Cluster separation (higher is better)
- **Davies-Bouldin Index**: Cluster compactness (lower is better)

### Supervised Metrics (using true genres)
- **Adjusted Rand Index (ARI)** [0, 1]: Clustering agreement (higher is better)
- **Normalized Mutual Information (NMI)** [0, 1]: Information overlap (higher is better)
- **Purity** [0, 1]: Dominant class per cluster (higher is better)

## 📈 Expected Results

Typical performance ranges (your results may vary):

| Method | Silhouette | ARI | NMI | Purity |
|--------|-----------|-----|-----|--------|
| Direct KMeans | 0.05-0.15 | 0.10-0.20 | 0.20-0.30 | 0.30-0.40 |
| PCA+KMeans | 0.10-0.20 | 0.15-0.25 | 0.25-0.35 | 0.35-0.45 |
| MLP-VAE | 0.15-0.25 | 0.20-0.30 | 0.30-0.40 | 0.40-0.50 |
| Conv-VAE (multimodal) | 0.20-0.30 | 0.25-0.35 | 0.35-0.45 | 0.45-0.55 |
| CVAE (full fusion) | 0.25-0.35 | 0.30-0.40 | 0.40-0.50 | 0.50-0.60 |

*Note: Music genre clustering is inherently challenging; moderate scores are expected.*

## 🔧 Advanced Options

### Running All Experiments

```bash
# Run all baselines
python src/baselines.py --methods pca_kmeans direct_kmeans ae_kmeans spectral --visualize

# Run all VAE models
python src/train.py --mode all --epochs 50 --device cpu --multimodal
```

### Custom Configuration

Create a custom config YAML:

```yaml
# custom_config.yaml
seed: 42
batch_size: 64
learning_rate: 0.001
epochs: 100
latent_dim: 32
beta: 4.0  # For beta-VAE
```

Run with custom config:
```bash
python src/train.py --mode vae_mlp --config custom_config.yaml
```

### Resume/Skip Existing

All scripts support caching:
- `--force` flag to re-run
- By default, skips existing outputs

```bash
# Re-extract features
python src/features.py --force

# Re-transcribe all clips
python src/transcribe.py --force
```

## 📓 Notebooks

### Exploration
```bash
jupyter notebook notebooks/00_exploration.ipynb
```

Explore:
- Dataset statistics
- Audio waveforms and spectrograms
- Feature distributions
- Transcription examples

### Results
```bash
jupyter notebook notebooks/01_results_and_plots.ipynb
```

View:
- All experiment results in tables
- Comparative plots
- Best model selection
- Saved visualizations

## 🔍 Debugging and Verification

### Verify Preprocessing
```bash
python src/preprocess.py --verify_only
```

### Verify Features
```bash
python src/features.py --verify_only
```

### Test Transcription (small sample)
```bash
python src/transcribe.py --max_clips 10
```

## 💻 System Requirements

### Local CPU (Minimum)
- Intel i5 8th gen or equivalent
- 8GB RAM
- 5GB disk space
- Python 3.8+

**Limitations:**
- Transcription is very slow on CPU (~2-4 hours)
- Training is slower but manageable

### Google Colab (Recommended for Heavy Tasks)
- Use Colab GPU for:
  - Whisper transcription (~15-30 min vs 2-4 hours)
  - Model training (~2-5 min vs 10-20 min per model)

## 🐛 Troubleshooting

### Issue: GTZAN dataset not found
```
Error: GTZAN dataset not found!
```
**Solution:** Ensure GTZAN files are in `data/raw_gtzan/genres_original/<genre>/*.wav`

### Issue: ffmpeg not found
```
Warning: ffmpeg not found, using librosa (slower)
```
**Solution (Windows):** Download ffmpeg from https://ffmpeg.org/ and add to PATH. Or just use librosa (slower but works).

### Issue: faster-whisper import error
```
ImportError: faster-whisper not installed
```
**Solution:** 
```bash
pip install faster-whisper
# Or use openai-whisper:
pip install openai-whisper
```

### Issue: CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size:
```bash
python src/train.py --mode vae_conv --batch_size 16
```

### Issue: Low clustering scores
This is expected! Music genre clustering is inherently subjective and challenging. Scores of 0.2-0.4 for ARI/NMI are reasonable.

## 📝 Implementation Details

### Dataset Splitting
- **Track-level splits** to prevent data leakage
- All 3 segments from same track go to same split
- 80% train, 10% val, 10% test

### Feature Engineering
- **MFCC**: 20 coefficients → mean + std → 40-dim
- **Log-mel**: 128 mel bins × 431 time frames
- **TF-IDF**: 5000 features → SVD to 128-dim
- **Lyrics**: Whisper `tiny` model (faster) or `base` (better)

### Model Architectures
- **MLP-VAE**: [40 → 128 → 64 → 16] encoder, symmetric decoder
- **Conv-VAE**: 4-layer conv encoder, 4-layer deconv decoder
- **CVAE**: Concatenates condition (genre) to encoder input and decoder input

### Reproducibility
- Fixed random seeds (numpy, torch, sklearn)
- Deterministic operations where possible
- All hyperparameters logged

## 📚 References

- GTZAN Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
- Whisper: https://github.com/openai/whisper
- Faster-Whisper: https://github.com/guillaumekln/faster-whisper
- VAE Paper: Kingma & Welling, "Auto-Encoding Variational Bayes" (2013)
- Beta-VAE: Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017)

## 🎓 Citation

If you use this pipeline for research or academic work, please cite:

```
@misc{gtzan-music-clustering-2024,
  title={Unsupervised Music Clustering with Multimodal VAEs},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/music-clustering}}
}
```

## 📄 License

This project is for educational and research purposes. The GTZAN dataset has its own license terms.

## 🤝 Contributing

Suggestions and improvements welcome! This is a research/educational project.

## ✅ Completed Tasks Checklist

- [x] EASY: MLP-VAE + KMeans + t-SNE + Silhouette/CH
- [x] MEDIUM: Conv-VAE + multimodal fusion + multiple clustering + Silhouette/DB/ARI
- [x] HARD: CVAE/Beta-VAE + full fusion (audio+lyrics+genre) + NMI/ARI/Purity + reconstructions + baselines
- [x] Preprocessing pipeline with splits
- [x] Whisper transcription
- [x] Feature extraction (MFCC, log-mel, TF-IDF)
- [x] Baseline methods (PCA, AE, Spectral)
- [x] Comprehensive evaluation metrics
- [x] Visualizations (t-SNE, UMAP, reconstructions)
- [x] Jupyter notebooks
- [x] Complete documentation

## 📞 Support

For questions or issues, please check:
1. This README
2. Code comments in `src/` files
3. Notebooks for examples

---

**Happy Clustering! 🎵🎶**

