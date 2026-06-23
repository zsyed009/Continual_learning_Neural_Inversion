# Continual Learning for Real-Time Stormwater Control Surrogate Modeling

This repository contains the codebase implementing the methodology and experiments for sequential Continual Learning (CL) surrogate modeling and real-time stormwater control (RTC) optimization.

The framework integrates deep neural network surrogates within a physical Storm Water Management Model (SWMM) environment to predict Combined Sewer Overflows (CSOs), maintain performance stability across future extreme weather scenarios, and optimize control actions.

---

## Key Features

1. **Surrogate Modeling**:
   - **SDNN**: Standard Deep Neural Network surrogate.
   - **BDNN**: Bayesian Deep Neural Network implementing uncertainty estimation.
   - **XGBoost + DNN**: Two-step surrogate classifier-regressor pipeline.
2. **Continual Learning (CL) Implementations**:
   - Elastic Weight Consolidation (**EWC**)
   - Variational Continual Learning (**VCL**)
   - Elastic Variational Continual Learning (**EVCL**)
   - Episodic Replay Buffer (integrated across all three methods)
   - Naive Fine-Tuning & Joint Training baselines
3. **Control Optimization**:
   - Gradient-based **Neural Inversion** backpropagating CSO cost directly to the input layer to optimize 10 control actions (7 gates/orifices + 3 pumps).
   - Multi-start optimization (100 independent trials per storm event).
4. **Physical Validation**:
   - Automatic integration with **PySWMM** to run optimized actions in the physical model and compute actual CSO volumes and storage utilization over time.

---

## Directory Structure

```text
├── data/                       # Directory for precipitation datasets and SWMM input models
│   ├── precipitation/          # Place rainfall CSV files here
│   └── swmm_model/             # Place the SWMM input (.inp) file here
├── src/                        # Python package modules
│   ├── continual_learning/     # CL algorithms (EWC, VCL, Replay Buffer)
│   ├── models/                 # Surrogate architectures (SDNN, BDNN, Two-Step)
│   └── optimization/           # Neural Inversion Optimizer
├── scripts/                    # Execution entrypoints
│   ├── 01_generate_data.py     # Generates dataset via SWMM simulations
│   ├── 02_train_continual.py   # Trains CL surrogates sequentially
│   ├── 03_run_optimization.py  # Runs neural inversion & SWMM validation
│   └── 04_generate_plots.py    # Generates publication-ready figures
├── .gitignore                  # Git tracking rules
├── requirements.txt            # Project python dependencies
└── README.md                   # This file
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/continual-learning-stormwater.git
   cd continual-learning-stormwater
   ```

2. **Set up a Python environment** (Conda recommended):
   ```bash
   conda create -n swmm_cl_env python=3.9 -y
   conda activate swmm_cl_env
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Input Data**:
   - Place your precipitation time-series CSV in `data/precipitation/atlas14_15min_pcp.csv`
   - Place your PFSMC SWMM network file in `data/swmm_model/7MPF_CSO_08.inp`

---

## Running the Pipeline

The codebase is structured to run sequentially:

### Step 1: Generate Training Data
Run simulations under baseline, future tasks (1-4), and test storm events to build surrogate training data:
```bash
python scripts/01_generate_data.py
```
*Outputs: training datasets saved to `data/baseline/` and `data/tasks/`.*

### Step 2: Train Continual Learning Surrogates
Run sequential training of all 7 CL techniques across Tasks 1-4, including 8-fold baseline cross-validation:
```bash
python scripts/02_train_continual.py
```
*Outputs: trained model checkpoints and metrics stored in `results/`.*

### Step 3: Run Real-Time Control Optimization
Perform Neural Inversion-based RTC optimization across the 100 random multi-start trials, and run physical SWMM validations:
```bash
python scripts/03_run_optimization.py
```
*Outputs: CSO volumes, MAEs, and storage timeseries saved to `results/`.*

### Step 4: Generate Publication Figures
Generate all publication-quality heatmaps, bar charts, and timeseries plots (Figures 5, 6, and 7):
```bash
python scripts/04_generate_plots.py
```
*Outputs: figures saved to `results/figures/`.*
