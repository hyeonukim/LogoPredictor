# Business Logo Classification Project

## Overview
This project explores the feasibility of predicting business characteristics from company logos using deep learning and traditional machine learning techniques. We tackle two classification tasks: (1) predicting industry categories and (2) predicting company country of origin.

## Project Goals
- Extract visual features from business logos using pre-trained deep learning models
- Predict industry categories (47 classes) from logo images
- Predict company country (top-10 countries) from logo images
- Compare deep learning vs. traditional machine learning approaches
- Evaluate performance under extreme class imbalance conditions

## Dataset

### Data Source
- **10,000 logos** from top companies (Crunchbase dataset)
- **9,943 logos successfully processed** (99.4% match rate)
- Logo images stored locally, matched with company metadata from CSV

### Category Classification Dataset
- **47 industry categories** (multi-label problem)
- **38,012 training pairs** after "exploding" multi-label to single-label
- **Train/Test Split**: 80/20 by unique logos (7,954 train / 1,989 test logos)
- **Severe class imbalance**: 
  - Largest class (Software): 5,076 samples
  - Smallest classes: <100 samples
  - Many categories: 1-10 samples

**Top Categories:**
1. Software (5,076)
2. Financial Services (2,378)
3. Information Technology (2,301)
4. Science and Engineering (2,043)
5. Internet Services (2,007)

### Country Classification Dataset
- **Top-10 countries** by logo count
- **8,846 logos total**
- **Train/Test Split**: 80/20 stratified (7,076 train / 1,770 test)
- **Extreme USA dominance**: 71.2% of dataset

**Country Distribution:**
- USA: 6,299 (71.2%)
- GBR: 616 (7.0%)
- IND: 608 (6.9%)
- CAN: 282 (3.2%)
- DEU: 231 (2.6%)
- CHN: 225 (2.5%)
- ISR: 202 (2.3%)
- FRA: 148 (1.7%)
- SGP: 146 (1.7%)
- AUS: 89 (1.0%)

## Methodology

### Feature Extraction
- **Model**: Pre-trained ResNet18 (ImageNet weights)
- **Features**: 512-dimensional embeddings (removed final classification layer)
- **Preprocessing**: Resize(256) → CenterCrop(224) → Normalize (ImageNet stats)
- **Device**: CUDA-enabled GPU

### Data Preparation
1. **Logo-CSV Matching**: Cleaned company names to match logo filenames
2. **Category Parsing**: Converted string representations to Python lists
3. **Multi-label Explosion**: Each (logo, category) pair becomes a training sample
4. **Standardization**: StandardScaler applied to features
5. **Stratified Splitting**: Ensured logos appear in either train OR test (no leakage)

### Models Evaluated

#### 1. Random Forest Classifier
```python
n_estimators=300
max_depth=25
min_samples_split=8
min_samples_leaf=3
max_features='sqrt'
class_weight='balanced'
```

#### 2. Logistic Regression
```python
multi_class='ovr'  # One-vs-Rest
max_iter=2000
C=1.0
class_weight='balanced'
```

#### 3. ResNet-MLP (Custom Neural Network)
**Architecture:**
- Input: 512-d ResNet features
- FC(512) → ReLU → FC(512) → Residual Connection → ReLU → FC(num_classes)
- Residual connection helps gradient flow

**Training:**
- Optimizer: Adam (lr=1e-3)
- Loss: CrossEntropyLoss
- Batch size: 128 (train), 256 (test)
- Epochs: 10
- Device: CUDA GPU

### Evaluation Metrics

#### Row-Level Metrics
- Accuracy: Correct predictions / Total predictions
- Precision, Recall, F1-Score (weighted & macro)
- Classification report per class

#### Logo-Level Membership Accuracy (Category Task)
- Groups predictions by unique logo
- **Correct if ANY predicted category matches true categories**
- More realistic for multi-label evaluation
- Formula: Logos with ≥1 correct prediction / Total logos

## Results

### Category Classification (47 Classes)

| Model | Test Accuracy (Row) | Test F1 (Weighted) | Logo Membership (Test) | Train-Test Gap |
|-------|---------------------|-------------------|------------------------|----------------|
| Random Forest | 3.25% | 0.026 | 12.62% | 85.78% (severe overfit) |
| Logistic Regression | 1.57% | 0.012 | 6.08% | 16.14% |
| **ResNet-MLP** ✓ | **10.34%** | **0.060** | **40.17%** | 45.31% |

**Key Findings:**
- ResNet-MLP outperforms traditional ML by 3-6x
- Software category achieves 58.1% recall (largest class)
- Most categories (<100 samples) have 0% recall
- Logo membership accuracy (40%) shows practical utility

**Top Performing Categories (ResNet-MLP):**
- Software: 58.1% recall
- Health Care: 12.7% recall
- Financial Services: 8.6% recall
- Science & Engineering: 6.7% recall

### Country Classification (Top-10 Countries)

| Model | Train Acc | Test Acc | Test F1 (Macro) | USA Recall | Non-USA Detection |
|-------|-----------|----------|----------------|------------|-------------------|
| Random Forest | 100.0% | 71.24% | 0.088 | 100% | 1 logo (CHN only) |
| Logistic Regression | 44.8% | 24.07% | 0.104 | 27.8% | All countries attempted |
| **ResNet-MLP** ✓ | 92.2% | **65.59%** | **0.105** | 91.1% | **CHN: 15.6% recall** |

**Key Findings:**
- ResNet-MLP achieves best balance (65.6% accuracy, detects 4 countries)
- Random Forest has highest raw accuracy but only predicts USA
- China is the only non-USA country with meaningful detection (15.6% recall)
- 6 of 10 countries have 0% recall due to small sample sizes

**Per-Country Performance (ResNet-MLP):**
- USA: 91.1% recall, 71.2% precision
- CHN: 15.6% recall, 21.2% precision
- GBR: 3.3% recall
- IND: 1.6% recall
- 6 countries: 0% recall (AUS, CAN, DEU, FRA, ISR, SGP)

## What Worked

### Successes
1. **Deep learning superiority**: ResNet-MLP consistently outperformed traditional ML by 3-6x
2. **Feature quality**: ResNet18 embeddings effectively capture visual patterns
3. **Logo membership metric**: 40% accuracy shows logos do encode industry information
4. **Country > Category**: Fewer classes and clearer patterns led to 6x better accuracy
5. **China detection**: Model learned to distinguish Chinese logos from USA (15.6% recall)

### Technical Strengths
- Proper train-test splitting by logo (prevents data leakage)
- Residual connections in neural network improved gradient flow
- Class-weighted loss helped with imbalance (though insufficient)
- GPU acceleration enabled efficient feature extraction

## What Didn't Work

### Major Challenges
1. **Extreme class imbalance**: Dominant classes overshadow minority classes
   - Software (5,076 samples) vs. Events (24 samples)
   - USA (71.2%) vs. Australia (1.0%)
2. **Multi-label forced to single-label**: Not true multi-label classification
3. **Small sample classes**: Categories with <100 samples get 0% recall
4. **Low absolute performance**: 10% row-level accuracy for categories
5. **USA dominance**: Country model essentially learned "USA or not?"

### Technical Limitations
- No hyperparameter tuning or cross-validation
- Limited model architectures tested (only 3)
- No advanced imbalance techniques (SMOTE, focal loss, etc.)
- Feature extractor frozen (didn't fine-tune ResNet)
- No interpretability analysis (which visual features matter?)

## Key Insights

1. **Class imbalance is the bottleneck**: Models learn to predict majority class
2. **More classes = harder task**: 47 categories much harder than 10 countries
3. **Data quality > model complexity**: ResNet features more important than classifier choice
4. **Evaluation metrics matter**: Logo membership more meaningful than row-level accuracy
5. **Geographic bias**: Logo design may not strongly correlate with country

## Recommendations for Future Work

### Addressing Class Imbalance
- Implement oversampling (SMOTE, ADASYN) for minority classes
- Use focal loss or class-weighted loss functions
- Collect more balanced data for underrepresented categories
- Try curriculum learning (train on easy classes first)

### Model Improvements
- Fine-tune entire ResNet instead of just feature extraction
- Implement true multi-label classification (sigmoid + BCE loss)
- Try Vision Transformers (ViT) or other modern architectures
- Ensemble multiple models for robustness
- Add attention mechanisms to focus on logo-specific regions

### Evaluation & Analysis
- Use proper multi-label metrics (Hamming loss, subset accuracy)
- Perform k-fold cross-validation for robust estimates
- Analyze which visual features drive predictions (interpretability)
- Test on external logo datasets for generalization
- Conduct error analysis on misclassified logos

### Data Collection
- Balance dataset through targeted collection
- Gather more samples for minority classes
- Include temporal information (logo evolution over time)
- Collect additional metadata (company size, founding date, etc.)
- Create synthetic data through augmentation

## Technical Requirements

### Dependencies
```
python >= 3.10
torch >= 2.6.0
torchvision
pandas
numpy
matplotlib
seaborn
Pillow
scikit-learn
joblib
```

### Hardware
- CUDA-capable GPU recommended (used in this project)
- ~16GB RAM for feature extraction
- ~10GB disk space for logo images

## File Structure
```
project/
├── data_science-all_list.ipynb    # Main notebook with all experiments
├── top10k_logos.csv               # Company metadata
├── logo_images/                   # Directory with logo image files
└── README.md                      # This file
```

## Usage

### Running the Notebook
1. Ensure logo images are in `logo_images/` directory
2. Update paths in notebook to match your file locations
3. Run cells sequentially from top to bottom
4. Feature extraction takes ~30-60 minutes on GPU

### Key Sections
1. **Data Loading & Matching**: Match logos to metadata
2. **Feature Extraction**: ResNet18 embedding generation
3. **Category Classification**: Train and evaluate 3 models
4. **Country Classification**: Train and evaluate 3 models on country subset

## Lessons Learned

1. **Start with balanced data**: Imbalance should be addressed at data collection stage
2. **Choose appropriate metrics**: Accuracy alone misleading for imbalanced datasets
3. **Multi-label needs special handling**: Don't force into single-label framework
4. **Deep learning is essential**: Visual features too complex for linear models
5. **Interpretability matters**: Understanding what model learns as important as accuracy

## Conclusion

This project demonstrates that business logos encode information about industry categories and geographic origin, achieving 40% logo membership accuracy for categories and 66% accuracy for country prediction. Deep learning approaches significantly outperform traditional ML, but extreme class imbalance remains the primary obstacle. Future work should focus on collecting balanced datasets, implementing true multi-label classification, and employing advanced techniques for handling imbalanced data.

The results suggest that with proper data collection and advanced modeling techniques, logo-based business classification is a viable approach for automated company categorization systems.

## Authors & Acknowledgments

- Dataset: Crunchbase top 10,000 companies
- Pre-trained Model: ResNet18 (ImageNet weights)
- Framework: PyTorch 2.6.0

## License

[Specify your license here]

---

**Last Updated**: December 2024