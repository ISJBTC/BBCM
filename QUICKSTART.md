# Quick Start Guide

This guide will help you get started with the Blood-Based Cancer Mathematical Model in 10 minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB RAM (16GB recommended)

## Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/yourusername/blood-cancer-model.git
cd blood-cancer-model

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Examples

### Example 1: Clinical Prediction (30 seconds)

Make predictions from a patient's blood biomarkers:

```python
from src.production_system import ClinicalCancerModel, BloodPanel

# Initialize model
model = ClinicalCancerModel()

# Create patient blood panel (example: early-stage patient)
panel = BloodPanel(
    ca153=45.0,      # Tumor marker CA 15-3
    cea=5.2,         # Tumor marker CEA
    cd8=850.0,       # CD8+ T cells
    cd4=1200.0,      # CD4+ T cells
    albumin=4.0,     # Serum albumin
    glucose=95.0,    # Blood glucose
    ldh=200.0        # Lactate dehydrogenase
)

# Get predictions
result = model.predict_from_blood_panel(panel)

# View results
print(f"Treatment Effectiveness: {result['parameters']['eta_E']:.2%}")
print(f"Confidence: {result['confidence']['level']}")
print(f"Recommendations: {result['confidence']['recommendations']}")
```

**Output:**
```
Treatment Effectiveness: 72%
Confidence: MODERATE
Recommendations: ['Monitor CD8 levels closely', 'Consider treatment optimization']
```

### Example 2: Generate Synthetic Data (2 minutes)

Generate realistic synthetic patient data for research:

```python
from src.complete_ml_cancer_analysis import MLEnhancedCancerModel
import pandas as pd

# Initialize ML model system
ml_model = MLEnhancedCancerModel()

# Generate 1000 synthetic patients
print("Generating synthetic patient data...")
synthetic_data = ml_model.generate_enhanced_synthetic_data(n_samples=1000)

# Save to CSV
synthetic_data.to_csv('my_synthetic_data.csv', index=False)
print(f"Generated {len(synthetic_data)} patients with {len(synthetic_data.columns)} features")

# View summary statistics
print(synthetic_data[['CA153', 'CEA', 'CD8', 'CD4', 'stage']].describe())
```

### Example 3: Stability Analysis (5 minutes)

Analyze mathematical stability of the system:

```python
from src.stability_simulation import ComprehensiveDocumentationGenerator

# Initialize analyzer
analyzer = ComprehensiveDocumentationGenerator(output_dir="my_analysis")

# Run stability analysis on 5 parameter sets
print("Running stability analysis...")
results = analyzer.run_stability_analysis_with_documentation(n_parameter_sets=5)

# View results
print(f"\nStability Summary:")
print(f"Total parameter sets: {len(results)}")
stable_count = sum(1 for r in results if r.get('stability_status') == 'STABLE')
print(f"Stable configurations: {stable_count}/{len(results)}")
print(f"\nResults saved in: my_analysis/")
```

**Output:**
```
Running stability analysis...
Parameter Set 1/5: STABLE ✓
Parameter Set 2/5: STABLE ✓
Parameter Set 3/5: STABLE ✓
Parameter Set 4/5: STABLE ✓
Parameter Set 5/5: STABLE ✓

Stability Summary:
Total parameter sets: 5
Stable configurations: 5/5

Results saved in: my_analysis/
```

### Example 4: Train Machine Learning Models (15 minutes)

Train all 8 ML models on synthetic data:

```python
from src.complete_ml_cancer_analysis import run_complete_ml_cancer_analysis

# This will:
# 1. Generate 5000 synthetic patients
# 2. Train 8 different ML models
# 3. Create ensemble models
# 4. Generate comprehensive results

print("Training ML models (this may take 10-15 minutes)...")
ml_model, results = run_complete_ml_cancer_analysis()

print("\nTraining Complete!")
print(f"Results saved in: mlstudy/")
print("\nModel Performance Summary:")
print(results['model_rankings'])
```

## Data Files Overview

### Available Datasets

1. **`data/synthetic_cancer_dataset_enhanced.csv`** (5000 patients)
   - 47 biomarkers per patient
   - Ground truth parameters
   - Multiple cancer stages

2. **`data/comprehensive_ml_results.csv`**
   - ML model performance metrics
   - 144 model-parameter combinations

3. **`data/parameter_ranges.csv`**
   - Parameter specifications
   - Biological ranges and units

## Common Use Cases

### Use Case 1: Research - Parameter Sensitivity Analysis

```python
from src.mathematical_validation import MathematicalAnalysis

analyzer = MathematicalAnalysis()
sensitivities = analyzer.sensitivity_analysis()

# View most sensitive parameters
for param, sensitivity in sorted(sensitivities.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:5]:
    print(f"{param}: {sensitivity:.1f}%")
```

### Use Case 2: Clinical - Patient Risk Stratification

```python
# Load patient data
import pandas as pd
patients = pd.read_csv('patient_biomarkers.csv')

# Classify each patient
for idx, patient in patients.iterrows():
    panel = BloodPanel(**patient.to_dict())
    prediction = model.predict_from_blood_panel(panel)
    
    risk = "HIGH" if prediction['parameters']['lambda1'] > 0.08 else "LOW"
    print(f"Patient {idx}: Risk={risk}, Confidence={prediction['confidence']['level']}")
```

### Use Case 3: Treatment Planning - Response Prediction

```python
# Compare treatment options
treatments = ['hormone', 'chemotherapy', 'immunotherapy']

for treatment in treatments:
    # Simulate treatment
    biomarkers_with_treatment = panel.__dict__.copy()
    # Adjust biomarkers based on treatment
    
    prediction = model.predict_from_blood_panel(BloodPanel(**biomarkers_with_treatment))
    print(f"{treatment}: Effectiveness = {prediction['parameters']['eta_E']:.2%}")
```

## Interactive Web Interface

Open the web calculator for interactive exploration:

```bash
# Open in browser
open src/model_calculator.html
# or
python -m http.server 8000
# Then navigate to: http://localhost:8000/src/model_calculator.html
```

## Troubleshooting

### Issue: ImportError

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# If specific package fails, install individually
pip install numpy pandas scipy scikit-learn
```

### Issue: Memory Error with Large Datasets

```python
# Use smaller batch sizes
ml_model = MLEnhancedCancerModel()
data = ml_model.generate_enhanced_synthetic_data(n_samples=1000)  # Instead of 5000
```

### Issue: Slow ML Training

```python
# Train only key models
models_to_train = ['XGBoost', 'LightGBM', 'CatBoost']
# Modify complete_ml_cancer_analysis.py to train only these models
```

## Next Steps

1. **Read Full Documentation**: See `README.md` for comprehensive guide
2. **Explore Notebooks**: Check `notebooks/` for detailed tutorials
3. **Run Tests**: Execute `pytest tests/` to verify installation
4. **Customize Models**: Modify parameters in source files
5. **Contribute**: See `CONTRIBUTING.md` for guidelines

## Getting Help

- **Issues**: Open an issue on GitHub
- **Email**: [your.email@domain.com]
- **Documentation**: Full docs in `docs/` folder

## Quick Reference - Key Files

| File | Purpose |
|------|---------|
| `src/production_system.py` | Clinical prediction system |
| `src/complete_ml_cancer_analysis.py` | ML training pipeline |
| `src/stability_simulation.py` | Mathematical stability analysis |
| `src/mathematical_validation.py` | Proof verification |
| `data/synthetic_cancer_dataset_enhanced.csv` | Main dataset |

## Performance Benchmarks

On a standard laptop (8GB RAM, Intel i5):
- **Clinical prediction**: < 1 second
- **Stability analysis (5 sets)**: ~2 minutes
- **ML training (5000 patients)**: ~15 minutes
- **Synthetic data generation**: ~30 seconds

---

**Ready to start? Try Example 1 above!** 🚀
