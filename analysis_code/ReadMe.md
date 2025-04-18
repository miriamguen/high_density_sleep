# High Density Sleep Analysis Pipeline

This repository contains a comprehensive pipeline for analyzing high-density EEG sleep data from the ANPHY-Sleep database.

## Setup

1. **Configuration**
   - Configure `parameters.yaml` with your settings:
     - Data directories
     - Channel configurations (EEG, EMG, EOG, ECG)
     - Processing parameters (window length, step size, batch size)
     - Band definitions for spectral analysis
     - Output directories for features, models, and figures

2. **Environment Setup**
   ```bash
   # Create and activate a new conda environment
   conda create -n sleep_analysis python=3.11.10
   conda activate sleep_analysis

   # Install dependencies from requirements.txt
   pip install -r requirements.txt
   ```

3. **Data Requirements**
   - Download data from the ANPHY-Sleep database
   - Required files:
     - EEG recordings (.edf files)
     - Sleep stage annotations (.txt files)
     - Subject metadata (Details information for healthy subjects.csv)
     - Channel positions file (Co-registered average positions.pos)

## Pipeline Overview and Outputs

### 1. Demographic and Sleep Label Analysis
**Script**: `demographic_and_sleep_label_analysis.py`
**Functionality**:
- Processes demographic information from metadata file
- Analyzes sleep stage distributions
- Calculates sleep quality metrics
- Generates statistical summaries

**Outputs**:
- Demographic summary statistics (.csv)
- Sleep stage distribution plots (.png)
- Sleep quality metrics by subject (.csv)
- Statistical test results (.csv)

### 2. Data Preprocessing
**Scripts**: 
- `preprocessing_main.py`: Main processing script
- `preprocessing_utils.py`: Utility functions for preprocessing

**Functionality**:
- Batch processing of spectral features
- Channel type mapping and montage setup
- Filtering and referencing
- Epoch creation with configurable length
- Feature extraction:
  - Spectral features (delta, theta, alpha, beta bands)
  - EMG features (muscle activity)
  - EOG features (eye movements)
  - ECG features (heart rate variability)

**Outputs**:
- Feature matrices (.csv) containing:
  - Spectral band powers
  - Normalized spectral features
  - Heart rate variability metrics
  - EMG activity measures
  - EOG movement indicators
  - Synchronized sleep stage labels
- Processing log files (.log)

### 3. Signal Decomposition
**Script**: `decomposition_main.py`
**Functionality**:
- Cross-subject analysis, joins and preprocess data from all subjects
- Applies advanced signal decomposition techniques
- Dimensionality reduction
- Component extraction
- Pattern identification
- for  "six_channels" subset add this to the output path:
*->* output_path = Path(PARAMETERS["OUTPUT_DIR"]) / "six_channels"


**Outputs**:
- Decomposition matrices (.npy)
- Component weights and map figures (.csv, .svg)
- Explained variance ratios (.csv)
- Component visualizations - state space and timeline (.svg)
- for the "six_channels" the same output wil be saved in the subfolder


### 4. Component Analysis
**Script**: `component_statistics.py`
**Functionality**:
- Statistical analysis of extracted components
- Significance testing
- Component clustering
- Sleep stage association analysis
- Inter-subject variability assessment

**Outputs**:
- Statistical test results (.csv)
- Component cluster assignments (.csv)
- Stage-component associations (.csv)
- Variability metrics (.csv)
- Statistical visualization plots (.png)

### 5. Hidden Markov Modeling
**Script**: `hmm_main.py`
**Functionality**:
- HMM model training and validation
- State sequence prediction
- Transition probability estimation
- Model selection and optimization
- State map generation

**Outputs**:
- Trained HMM models (.pkl)
- State sequences (.csv)
- Transition matrices (.csv)
- Model performance metrics (.csv)
- State probability maps (.png)
- Cross-validation results (.csv)

### 6. Sleep Evaluation
**Script**: `sleep_evaluation_utils.py`
**Functionality**:
- Sleep quality metric calculations
- State sequence analysis
- Transition pattern detection
- Sleep architecture analysis
- Validation against manual scoring

**Outputs**:
- Sleep quality metrics (.csv)
- State sequence statistics (.csv)
- Transition pattern reports (.csv)
- Validation metrics (.csv)
- Sleep architecture plots (.png)

### 7. Visualization
**Scripts**:
#### `plot_state_map.py`
**Functionality**:
- Generates topographic maps of state features
- Visualizes spatial patterns
- Creates state-specific brain activity maps
- Temporal evolution visualization

**Outputs**:
- Topographic maps (.png)
- State feature visualizations (.png)
- Temporal evolution plots (.png)
- Interactive visualizations (.html)

#### `visualize_most_typical.py`
**Functionality**:
- Identifies representative EEG examples
- Generates state-specific visualizations
- Creates comparative displays
- Highlights key features

**Outputs**:
- Representative EEG plots (.png)
- State comparison figures (.png)
- Feature highlight plots (.png)
- Summary visualizations (.pdf)

### 8. State Inference
**Script**: `transform_and_infer_states.py`
**Functionality**:
- Applies trained models to new data
- Real-time state inference
- Confidence estimation
- Model adaptation

**Outputs**:
- Inferred state sequences (.csv)
- Confidence scores (.csv)
- Adaptation parameters (.json)
- Performance reports (.pdf)

## Statistical Methods and Analysis Approaches

### 1. Preprocessing and Feature Extraction (`preprocessing_main.py`, `preprocessing_utils.py`)
**Statistical Methods**:
- **Spectral Analysis**:
  - Welch's method for power spectral density estimation
  - Band-specific power calculation (delta, theta, alpha, beta)
  - Frequency-domain feature extraction
- **Signal Filtering**:
  - High-pass filtering (0.5 Hz cutoff for EEG, 0.3 Hz for EOG, 10 Hz for EMG)
  - Low-pass filtering (40 Hz for EEG, 15 Hz for EOG, 50 Hz for EMG, 20 Hz for ECG)
- **Normalization**:
  - Z-score normalization (mean subtraction and division by standard deviation)
  - Channel-wise standardization

**Packages**:
- `mne`: EEG processing pipeline, filter application, PSD calculation
- `numpy`: Numerical computations and array processing
- `pandas`: Data manipulation and CSV file handling
- `tqdm`: Progress tracking for batch processing

### 2. Decomposition Analysis (`decomposition_main.py`, `decomposition_utils.py`)
**Statistical Methods**:
- **Dimensionality Reduction**:
  - Principal Component Analysis (PCA)
  - Independent Component Analysis (ICA)
- **Component Extraction**:
  - FastICA algorithm
  - Variance explained threshold

**Packages**:
- `sklearn.decomposition`: PCA and ICA implementations
- `pandas`: Data handling
- `numpy`: Matrix operations


### 3. Component Analysis (`component_statistics.py`)
**Statistical Methods**:
- **Mixed Effects Models**:
  - Linear mixed models with subject as random effect
  - Fixed effects for sleep stages
  - Confidence interval calculation using standard errors
- **Significance Testing**:
  - P-value calculation for coefficient significance
  - Significance stars (*p<0.001, **p<0.0001, ***p<0.00001)
- **Correlation Analysis**:
  - Pairwise correlation between components

**Packages**:
- `statsmodels.formula.api`: Mixed linear models (`smf.mixedlm`)
- `pandas`: Data handling
- `numpy`: Numerical operations
- `scipy.stats`: Statistical testing
- `seaborn` & `matplotlib`: Statistical visualization
- `arviz`: Bayesian visualization components

### 4. Hidden Markov Modeling (`hmm_main.py`, `hmm_utils.py`)
**Statistical Methods**:
- **HMM Implementation**:
  - Gaussian emission HMM with full covariance matrices
  - Viterbi algorithm for state sequence decoding
  - Forward algorithm for likelihood calculation
  - Baum-Welch algorithm for parameter estimation
- **Model Selection Criteria**:
  - Bayesian Information Criterion (BIC)
  - Akaike Information Criterion (AIC)
- **Performance Metrics**:
  - Accuracy score
  - Adjusted Rand Index (ARI)
  - Cohen's Kappa for inter-rater agreement
- **Cross-Validation**:
  - Leave-one-subject-out cross-validation
  - Patient-specific normalization

**Packages**:
- `hmmlearn.hmm`: GaussianHMM implementation
- `sklearn.metrics`: Performance evaluation (accuracy, ARI, kappa)
- `numpy`: Array operations
- `pandas`: Data manipulation
- `networkx`: Transition graph visualization



### 5. Sleep Evaluation (`sleep_evaluation_utils.py`)
**Statistical Methods**:
- **Transition Analysis**:
  - Markov transition probability matrices
  - Sequence length estimation
- **Sleep Architecture Analysis**:
  - Stage proportion calculation
  - Sleep efficiency metrics
  - Sleep fragmentation indices

**Packages**:
- `numpy`: Array operations
- `pandas`: Data analysis
- `scipy`: Scientific computing
- `collections.Counter`: Frequency counting

### 6. Visualization (`plot_state_map.py`, `visualize_most_typical.py`)
**Statistical Methods**:
- **Topographic Mapping**:
  - Spatial interpolation
  - Channel-wise statistics
- **Distribution Analysis**:
  - Kernel Density Estimation (KDE)
  - Histogram visualization

**Packages**:
- `matplotlib`: Basic plotting
- `seaborn`: Statistical visualization
- `mne`: Topographic plotting

## Model Evaluation and Validation

### Cross-Validation
- **Leave-One-Subject-Out**:
  ```python
  # From hmm_utils.py
  def fit_and_score_cv(data, feature_names, label_col, n_states, cv_col="patient", scale=False, return_model=False):
      # Cross-validation across subjects
      patients = data[cv_col].unique()
      for test_patient in patients:
          train = data[data[cv_col] != test_patient]
          test = data[data[cv_col] == test_patient]
          # Model training and evaluation
  ```

### Performance Metrics
- **Cohen's Kappa**: Measures inter-rater agreement beyond chance
  ```python
  # From hmm_utils.py
  kappa = cohen_kappa_score(labels, mapped_states)
  ```
- **Adjusted Rand Index**: Measures similarity between clusterings
  ```python
  # From hmm_utils.py
  ari = adjusted_rand_score(labels, hidden_states)
  ```
- **Accuracy**: Proportion of correctly classified samples
  ```python
  # From hmm_utils.py
  acc = accuracy_score(labels, mapped_states)
  ```

### Model Selection Criteria
- **BIC**: Penalizes model complexity
  ```python
  # From hmm_utils.py
  bic = model._compute_score(X) + np.log(len(X)) * model.n_parameters
  ```
- **AIC**: Balances goodness of fit and model complexity
  ```python
  # From hmm_utils.py (inferred from evaluation function)
  aic = -2 * log_likelihood + 2 * model.n_parameters
  ```

## File Structure

```
high_density_sleep/
├── analysis_code/
│   ├── preprocessing_*.py
│   ├── decomposition_*.py
│   ├── visualization_*.py
│   └── parameters.yaml
├── data/
│   ├── raw/
│   └── processed/
├── output/
│   ├── features/
│   ├── models/
│   └── figures/
└── requirements.txt
```

## Notes

- Process files in the order specified above
- Adjust batch_size in parameters.yaml based on available RAM
- Monitor memory usage during processing of large datasets
- All outputs are saved in their respective directories as specified in parameters.yaml
- Log files are generated for each processing step
- Intermediate results are cached when possible to speed up reprocessing