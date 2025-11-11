# Heart Attack Predictor - Code Reproduction Guide

This repository contains a complete, reproducible machine learning pipeline for heart attack prediction using clinical data. All results can be regenerated from scratch with fixed random seeds.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Defaultt512/MLBA.git
cd MLBA

# Install dependencies
pip install -r requirements.txt

# Run the complete analysis
python run_ml_analysis.py

# Generate LaTeX paper
python generate_latex.py
```

## Requirements

- **Python**: 3.11 or higher
- **RAM**: Minimum 4GB recommended
- **Time**: ~2-3 minutes for complete execution
- **GPU**: Optional (gracefully falls back to CPU)

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/Defaultt512/MLBA.git
cd MLBA
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

All dependencies are pinned to specific versions for reproducibility.

### Step 3: Verify Installation
```bash
python -c "import sklearn, xgboost, tensorflow; print('All libraries installed successfully')"
```

## Running the Analysis

### Option 1: Direct Python Execution (Recommended)

Run the complete pipeline in one command:

```bash
python run_ml_analysis.py
```

This will:
- Load and preprocess the dataset
- Apply SMOTE for class imbalance
- Train all 5 models (Logistic Regression, SVM, Random Forest, XGBoost, Neural Network)
- Generate all evaluation metrics (ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1)
- Perform threshold optimization and cost analysis
- Run ablation studies
- Calculate business impact
- Generate all plots (8 high-quality PNG files)
- Save all results to `outputs/` directory

**Expected Output:**
```
✓ 10 CSV files in outputs/
✓ 8 PNG plots in outputs/plots/
✓ Runtime: ~2 minutes
```

### Option 2: Interactive Jupyter Notebook

For step-by-step exploration:

```bash
jupyter notebook heart_attack_ml_analysis.ipynb
```

Then run all cells in order. The notebook contains the same analysis with detailed explanations.

### Option 3: Generate LaTeX Paper

After running the analysis:

```bash
python generate_latex.py
```

This creates `outputs/MLBA_Group_11_Complete.tex` - an IEEE-formatted research paper with all results integrated.

## Project Structure

```
MLBA/
├── run_ml_analysis.py              # Main analysis script
├── heart_attack_ml_analysis.ipynb  # Interactive notebook
├── generate_latex.py               # LaTeX generator
├── requirements.txt                # Pinned dependencies
├── README.md                       # This file
├── attached_assets/
│   └── heart_cleaned_*.csv         # Dataset (270 records, 14 features)
└── outputs/                        # Generated results
    ├── *.csv                       # Metrics and tables (10 files)
    ├── MLBA_Group_11_Complete.tex  # Research paper
    └── plots/                      # Visualizations (8 PNG files)
```

## Output Files

### CSV Files (10 files)

| File | Description |
|------|-------------|
| `model_comparison.csv` | Performance metrics for all 5 models |
| `data_card.csv` | Dataset metadata and feature descriptions |
| `split_table.csv` | Train/test split statistics |
| `threshold_analysis.csv` | Threshold sweep from 0.1 to 0.85 |
| `error_analysis.csv` | False positive/negative counts |
| `business_impact.csv` | Cost savings and lives saved estimates |
| `ablation_rf.csv` | Random Forest ablation (n_estimators) |
| `ablation_xgb.csv` | XGBoost ablation (learning_rate) |
| `rf_feature_importance.csv` | Feature rankings (Random Forest) |
| `xgb_feature_importance.csv` | Feature rankings (XGBoost) |

### Plots (8 PNG files, 300 DPI)

1. `target_distribution.png` - Class distribution
2. `model_comparison.png` - 6 metrics across all models
3. `roc_curves.png` - ROC curves for all 5 models
4. `pr_curves.png` - Precision-Recall curves
5. `confusion_matrices.png` - Confusion matrices grid
6. `feature_importance.png` - RF & XGB feature importance
7. `threshold_cost_analysis.png` - Threshold sweep & cost curve
8. `ablation_studies.png` - Hyperparameter sensitivity

## Reproducibility

### Fixed Random Seed
All random operations use `RANDOM_SEED = 42`:
- NumPy random operations
- TensorFlow/Keras model initialization
- scikit-learn train/test splitting
- SMOTE sampling
- Hyperparameter search

### Environment
```bash
python --version  # Python 3.11+
pip list | grep -E "scikit-learn|xgboost|tensorflow"
```

### Expected Results

With the fixed seed, you should get these exact results:

**Best Model: Logistic Regression**
- ROC-AUC: 0.9216
- PR-AUC: 0.8970
- Accuracy: 85.19%
- F1-Score: 0.850

**All Models:**
| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Logistic Regression | 0.922 | 0.897 |
| Neural Network | 0.920 | 0.899 |
| SVM (RBF) | 0.912 | 0.840 |
| XGBoost | 0.912 | 0.869 |
| Random Forest | 0.903 | 0.860 |

Minor variations (±0.001) may occur due to floating-point arithmetic differences across systems.

## Data Description

**Dataset:** Heart disease clinical data  
**Source:** `attached_assets/heart_cleaned_1762844952756.csv`  
**Records:** 270 patients  
**Features:** 14 clinical variables  
**Target:** Heart disease presence (binary classification)

**Features:**
- Age, Sex, Chest pain type
- BP, Cholesterol, FBS over 120
- EKG results, Max HR, Exercise angina
- ST depression, Slope of ST
- Number of vessels (fluoroscopy), Thallium

**Class Distribution:**
- Absence (0): ~45%
- Presence (1): ~55%
- Imbalance handled via SMOTE

## Troubleshooting

### Issue: Import errors
```bash
# Solution: Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: TensorFlow warnings about GPU
```
# This is normal - the code runs fine on CPU
# To suppress warnings:
export TF_CPP_MIN_LOG_LEVEL=2
python run_ml_analysis.py
```

### Issue: Matplotlib backend errors
```bash
# Solution: Use non-interactive backend (already configured in code)
# Or install tkinter:
sudo apt-get install python3-tk  # Linux
brew install python-tk           # macOS
```

### Issue: Memory errors
```bash
# Reduce batch size in run_ml_analysis.py, line ~170:
# Change batch_size=32 to batch_size=16
```

## Customization

### Change Random Seed
Edit `RANDOM_SEED = 42` at the top of `run_ml_analysis.py`

### Modify Train/Test Split
```python
# Line ~45 in run_ml_analysis.py
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30,  # Change to 0.20 for 80/20 split
    random_state=RANDOM_SEED, 
    stratify=y
)
```

### Add New Models
```python
# Add after line ~150 in run_ml_analysis.py
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_balanced, y_train_balanced)
# ... add evaluation code
```

### Change Cost Ratio
```python
# Line ~200 in run_ml_analysis.py
cost_fn_ratio = 5  # Change to adjust FN vs FP cost
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{heart_attack_ml_2025,
  title={Heart Attack Risk Prediction using Machine Learning: A Comprehensive Analysis},
  author={MLBA Group 11},
  year={2025},
  journal={Machine Learning in Business Analytics}
}
```

## License

This project is available for academic and research purposes.

## Contact

For questions or issues, please open an issue on GitHub:
https://github.com/Defaultt512/MLBA/issues

## Acknowledgments

- Dataset: Cleveland Heart Disease Database
- Libraries: scikit-learn, XGBoost, TensorFlow, imbalanced-learn
- Environment: Replit Python 3.11

---

**Last Updated:** November 11, 2025  
**Version:** 1.0  
**Reproducibility Status:** ✅ Fully Reproducible
