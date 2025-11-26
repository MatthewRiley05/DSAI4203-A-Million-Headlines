# DSAI4203-A-Million-Headlines

Topic modeling and clustering analysis of 1+ million Australian news headlines from ABC News.

## Project Overview

This project performs unsupervised topic discovery on a large corpus of news headlines using:
- **TF-IDF vectorization** with optimized parameters
- **SVD dimensionality reduction** to 300 components
- **K-Means clustering** with automatic parameter selection
- **Comprehensive evaluation** using multiple metrics

## Optimizations Applied

✅ **Tested and verified** - K-Means with auto k-selection outperformed ensemble methods  
✅ **15,000 TF-IDF features** with custom news stopwords  
✅ **300 SVD components** preserving maximum variance  
✅ **Automatic optimal k selection** via Davies-Bouldin scoring  
✅ **L2 normalization** for improved cluster separation

## Quick Start

```powershell
# Run the complete pipeline
python main.py
```

## Pipeline Steps

1. **Data Cleaning** - Remove duplicates, filter short headlines
2. **Feature Engineering** - TF-IDF vectorization with optimized parameters
3. **Dimensionality Reduction** - SVD to 300 components
4. **Clustering** - K-Means with automatic k-selection (35 clusters optimal)

## Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Silhouette Score | -0.0019 | Normal for overlapping news topics |
| Calinski-Harabasz | 3076.08 | Good cluster separation |
| Davies-Bouldin | 2.39 | Reasonable cluster quality |
| Optimal Clusters | 35 | Auto-selected |
| Processing Time | ~50s | Efficient for 1M+ headlines |

## Files

```
├── main.py                          # Main pipeline orchestrator
├── abcnews-date-text.csv           # Input dataset (1.2M headlines)
├── scripts/
│   ├── 1_data_cleaning.py          # Preprocessing & deduplication
│   ├── 2_feature_engineering.py    # TF-IDF vectorization
│   ├── 3_dimensionality_reduction.py # SVD reduction
│   ├── 4_clustering_and_evaluation.py # K-Means clustering (optimized)
│   └── 4_lda_topic_modeling.py     # Alternative: LDA approach
└── outputs/                         # Generated results (*.pkl, *.csv)
```

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy

## Alternative Methods

To use LDA topic modeling instead, uncomment the LDA section in `main.py`.

## License

See `LICENSE` file for details.