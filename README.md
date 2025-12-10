# DSAI4203-A-Million-Headlines

Topic modeling and clustering analysis of 1+ million Australian news headlines from ABC News.

## Project Overview

This project performs unsupervised topic discovery and supervised (semi-supervised) classification of Fake/Real records on a large corpus of news headlines using:
- **TF-IDF vectorization** with optimized parameters
- **SVD dimensionality reduction** to 200 components
- **AE dimensionality reduction** to 10 components
- **K-Means clustering** with automatic parameter selection
- **Semi-supervized tehcnique** to label data
- **Classification** with different models 
- **Comprehensive evaluation** using multiple metrics

## Optimizations Applied

✅ **Tested and verified** - K-Means with auto k-selection outperformed ensemble methods  
✅ **10,000-15,000 TF-IDF features** with custom news stopwords  
✅ **200 SVD components** preserving maximum variance
✅ **10 Z latent size (AE)** reducing dimensionality burden
✅ **20 epochs** prevent from overtraining
✅ **Threshold for Self-training** including only good records
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
3. **Dimensionality Reduction** - SVD to 200 components
4. **Clustering** - LDA topic modeling (default; K-Means available as alternative)
5. **Autoencoder** - Dimensionality Reduction with AE on SVD to 10 components
6. **Plot AE Performance** - Comparing AE clustering results on different hyperparameters
7. **AE Clustering and Evaulation** - Clustering (K-means) and evaluation results
8. **AE Vizualization** - Plotting t-SNE on AE Clustering results
9. **Prepare FAKE/REAL Dataset** - Preparing labeled dataset
10. **Feature Engineering** - TF-IDF vectorization of labeled and unlabeled datasets
11. **Self-training** - Semi-supervised technique to pseudo-label data
12. **Evaluation of Self-training** - Check updated labeled + pseudo-labeled and unlabeled datasets
13. **Classification and Evaluation** - Classification model and evaluating their preformance

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
│   └── 4_lda_topic_modeling.py        # Alternative: LDA approach
│   ├── 5_autoencoder_on_svd.py        # AE dimensionality redcution
│   ├── 6_ae_clustering_and_evaluation.py    # K-Means clustering on Z latent space from AE
│   ├── 6_1_plot.py                    # Plotting 2 graph on AE performance on K-Means Clustering based on Z, epoch sizes
│   ├── 7_ae_vizualization.py          # t-SNE reduction to vizualize AE results on 2D space
│   ├── 8_1_fake_real_dataset_preparing.py   # Preparing, cleaning and mergind labeled (fake/real) news datasets
│   ├── 8_2_feature_engineering.py     # TF-IDF vectorization on labeled data, and unlabeled tranformed on that model 
│   ├── 8_3_evaluation.py              # Evaluating self-training result
│   ├── 8_3_self_training.py           # Semi-supervised technique: Self-training labeling
│   ├── 8_4_classification.py          # Classifacation trained on labeled + pseudolabeled data, and evaluated
│   ├── data_preprocessing.py          # Function to preprocess data and transform into better token for feature engineering 

    # @Yerasyk's experiments to improve models and scripts
│   ├── Experiment.txt                 # Assumption and experiments have done during work
│   ├── model_improvements.ipynb       # Testing and evaluation of models for further improvement
└── outputs/                         # Generated results (*.pkl, *.csv)
```

## Requirements

- Python 3.8+
- pandas
- scikit-learn
- numpy

## Alternative Methods

To use K-Means clustering instead, uncomment the K-Means section in `main.py`.

## License

See `LICENSE` file for details.
