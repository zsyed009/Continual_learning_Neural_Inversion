"""
02_train_continual.py
=====================
Sequential Continual Learning training pipeline.

Implements the paper's full experimental design:
  1. 8-fold cross-validation on baseline for model stability (Table 2).
  2. Baseline comparison methods: Naive Fine-Tuning, Joint Training, Expanding Window (Table 3).
  3. 7 CL methods sequentially across Tasks 1-4.
  4. Evaluation on ALL tasks after each step to track forgetting (Figure 5).
  5. Final retraining on Baseline to test recovery.
  6. 5 independent seeds for statistical significance.
"""

import os, sys, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.surrogate_model import SDNN
from src.models.bayesian_dnn import BDNN
from src.continual_learning.ewc import EWC, train_with_ewc
from src.continual_learning.vcl_replay import (
    train_vcl_with_replay, update_prior, compute_fisher_bdnn
)
from src.continual_learning.buffer import ReplayBuffer

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR    = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR  = os.path.join(RESULTS_DIR, 'trained_models')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Paper Table 1: Task-specific hyperparameters
HYPERPARAMS = {
    'EWC':          {'lambda': [0.40, 2.80, 3.40, 3.10]},
    'EWC+Replay':   {'lambda': [1.30, 1.30, 1.60, 1.70], 'buffer_capacity': 2500},
    'VCL':          {'beta': [0.60, 0.90, 0.75, 0.95]},
    'VCL+Replay':   {'beta': [0.70, 0.80, 0.90, 1.00], 'buffer_capacity': 2500},
    'EVCL':         {'lambda': [0.80, 1.00, 1.40, 1.30], 'beta': [0.60, 0.80, 0.90, 0.85]},
    'EVCL+Replay':  {'lambda': [0.70, 0.90, 1.70, 1.20], 'beta': [0.70, 0.80, 0.80, 0.95], 'buffer_capacity': 2500},
    'Replay':       {'buffer_capacity': 2500},
}

BATCH_SIZE      = 128
EPOCHS_PER_TASK = 50
LEARNING_RATE   = 1e-3
NUM_SEEDS       = 5
K_FOLDS         = 8

# ==========================================
# 2. DATA LOADING
# ==========================================
def load_task_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)
    return X, y

def make_dataloader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                      batch_size=batch_size, shuffle=shuffle)

# ==========================================
# 3. EVALUATION
# ==========================================
def evaluate_model(model, X, y, is_bayesian=False):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_t, sample=False).squeeze().numpy() if is_bayesian \
                else model(X_t).squeeze().numpy()
    return r2_score(y, preds), np.sqrt(mean_squared_error(y, preds))

# ==========================================
# 4. 8-FOLD CROSS-VALIDATION (Paper Table 2)
# ==========================================
def run_kfold_baseline(input_dim, X_bl, y_bl, X_test, y_test, task_datasets=None):
    """
    Paper: "8-fold cross-validation on the training set to assess stability.
    Standard deviations ranging from ±0.010 to ±0.018 for training R²."
    Trains all 3 models:
      Model 1: SDNN (Joint Training — baseline + all tasks pooled)
      Model 2: SDNN (baseline only)
      Model 3: BDNN (baseline only)
    """
    print("\n" + "=" * 60)
    print("  8-FOLD CROSS-VALIDATION (Table 2)")
    print("=" * 60)

    # Prepare pooled data for Model 1 (baseline + all tasks)
    if task_datasets is not None:
        task_keys = ['task1', 'task2', 'task3', 'task4']
        all_X = [X_bl] + [task_datasets[tk][0] for tk in task_keys if tk in task_datasets]
        all_y = [y_bl] + [task_datasets[tk][1] for tk in task_keys if tk in task_datasets]
        X_pooled = np.concatenate(all_X)
        y_pooled = np.concatenate(all_y)
    else:
        X_pooled = X_bl
        y_pooled = y_bl

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    cv_results = {
        'Model1_SDNN_withTasks': {'train_r2': [], 'train_rmse': [], 'test_r2': [], 'test_rmse': []},
        'Model2_SDNN_baseline':  {'train_r2': [], 'train_rmse': [], 'test_r2': [], 'test_rmse': []},
        'Model3_BDNN_baseline':  {'train_r2': [], 'train_rmse': [], 'test_r2': [], 'test_rmse': []},
    }

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_bl)):
        print(f"\n  Fold {fold_idx + 1}/{K_FOLDS}")
        X_tr, y_tr = X_bl[train_idx], y_bl[train_idx]
        X_val, y_val = X_bl[val_idx], y_bl[val_idx]
        loader_bl = make_dataloader(X_tr, y_tr)

        # Model 1: SDNN (Joint Training — pooled baseline + tasks)
        sdnn1 = SDNN(input_dim)
        opt1 = torch.optim.Adam(sdnn1.parameters(), lr=LEARNING_RATE)
        loader_pooled = make_dataloader(X_pooled, y_pooled)
        train_with_ewc(sdnn1, opt1, loader_pooled, ewc_module=None, epochs=EPOCHS_PER_TASK)
        tr_r2, tr_rmse = evaluate_model(sdnn1, X_pooled, y_pooled)
        te_r2, te_rmse = evaluate_model(sdnn1, X_test, y_test)
        cv_results['Model1_SDNN_withTasks']['train_r2'].append(tr_r2)
        cv_results['Model1_SDNN_withTasks']['train_rmse'].append(tr_rmse)
        cv_results['Model1_SDNN_withTasks']['test_r2'].append(te_r2)
        cv_results['Model1_SDNN_withTasks']['test_rmse'].append(te_rmse)

        # Model 2: SDNN (baseline only)
        sdnn2 = SDNN(input_dim)
        opt2 = torch.optim.Adam(sdnn2.parameters(), lr=LEARNING_RATE)
        train_with_ewc(sdnn2, opt2, loader_bl, ewc_module=None, epochs=EPOCHS_PER_TASK)
        tr_r2, tr_rmse = evaluate_model(sdnn2, X_tr, y_tr)
        te_r2, te_rmse = evaluate_model(sdnn2, X_test, y_test)
        cv_results['Model2_SDNN_baseline']['train_r2'].append(tr_r2)
        cv_results['Model2_SDNN_baseline']['train_rmse'].append(tr_rmse)
        cv_results['Model2_SDNN_baseline']['test_r2'].append(te_r2)
        cv_results['Model2_SDNN_baseline']['test_rmse'].append(te_rmse)

        # Model 3: BDNN (baseline only)
        bdnn = BDNN(input_dim)
        buf_dummy = ReplayBuffer(0)
        train_vcl_with_replay(bdnn, loader_bl, buf_dummy, prior_params=None,
                              beta=1.0, epochs=EPOCHS_PER_TASK, lr=LEARNING_RATE)
        tr_r2, tr_rmse = evaluate_model(bdnn, X_tr, y_tr, is_bayesian=True)
        te_r2, te_rmse = evaluate_model(bdnn, X_test, y_test, is_bayesian=True)
        cv_results['Model3_BDNN_baseline']['train_r2'].append(tr_r2)
        cv_results['Model3_BDNN_baseline']['train_rmse'].append(tr_rmse)
        cv_results['Model3_BDNN_baseline']['test_r2'].append(te_r2)
        cv_results['Model3_BDNN_baseline']['test_rmse'].append(te_rmse)

    # Print Table 2 format
    labels = {
        'Model1_SDNN_withTasks': 'Model 1: SDNN (with Tasks)',
        'Model2_SDNN_baseline':  'Model 2: SDNN (w/o Tasks)',
        'Model3_BDNN_baseline':  'Model 3: BDNN (w/o Tasks)',
    }
    print("\n  == Table 2: Model Predictive Performance (Mean +/- SD) ==")
    for key, metrics in cv_results.items():
        print(f"\n  {labels[key]}:")
        print(f"    Train  RMSE: {np.mean(metrics['train_rmse']):.0f} +/- {np.std(metrics['train_rmse']):.0f} ft3"
              f"   R2: {np.mean(metrics['train_r2']):.3f} +/- {np.std(metrics['train_r2']):.3f}")
        print(f"    Test   RMSE: {np.mean(metrics['test_rmse']):.0f} +/- {np.std(metrics['test_rmse']):.0f} ft3"
              f"   R2: {np.mean(metrics['test_r2']):.3f} +/- {np.std(metrics['test_r2']):.3f}")

    return cv_results

# ==========================================
# 5. BASELINE COMPARISON METHODS (Paper Table 3)
# ==========================================
def run_baseline_comparisons(input_dim, task_datasets, seed=42):
    """
    Naive Fine-Tuning, Joint Training (Model 1), Expanding Window.
    Evaluated on baseline test event and Task 4.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    results = {}
    bl_X, bl_y = task_datasets['baseline']
    task_keys = ['task1', 'task2', 'task3', 'task4']

    # --- A. Naive Fine-Tuning: sequential w/o any CL ---
    print("\n  [Naive Fine-Tuning] ...")
    model = SDNN(input_dim)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_with_ewc(model, opt, make_dataloader(bl_X, bl_y), epochs=EPOCHS_PER_TASK)
    for tk in task_keys:
        if tk in task_datasets:
            t_X, t_y = task_datasets[tk]
            train_with_ewc(model, opt, make_dataloader(t_X, t_y), epochs=EPOCHS_PER_TASK)
            # Save No CL checkpoint for each task!
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'No CL_{tk}.pt'))
    results['Naive Fine-Tuning'] = {
        'baseline': evaluate_model(model, bl_X, bl_y),
        'task4': evaluate_model(model, *task_datasets.get('task4', (bl_X, bl_y)))
    }
    # Save No CL model for neural inversion (Table 4)
    torch.save(model.state_dict(),
               os.path.join(MODELS_DIR, f'No CL_final_seed{seed}.pt'))

    # --- B. Joint Training (Model 1): pool all data ---
    print("  [Joint Training] ...")
    all_X = [bl_X]
    all_y = [bl_y]
    for tk in task_keys:
        if tk in task_datasets:
            all_X.append(task_datasets[tk][0])
            all_y.append(task_datasets[tk][1])
    pooled_X = np.concatenate(all_X)
    pooled_y = np.concatenate(all_y)
    model = SDNN(input_dim)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_with_ewc(model, opt, make_dataloader(pooled_X, pooled_y), epochs=EPOCHS_PER_TASK)
    results['Joint Training'] = {
        'baseline': evaluate_model(model, bl_X, bl_y),
        'task4': evaluate_model(model, *task_datasets.get('task4', (bl_X, bl_y)))
    }

    # --- C. Expanding Window: retrain on cumulative data each step ---
    print("  [Expanding Window] ...")
    cum_X, cum_y = bl_X.copy(), bl_y.copy()
    for tk in task_keys:
        if tk in task_datasets:
            cum_X = np.concatenate([cum_X, task_datasets[tk][0]])
            cum_y = np.concatenate([cum_y, task_datasets[tk][1]])
            model = SDNN(input_dim)
            opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            train_with_ewc(model, opt, make_dataloader(cum_X, cum_y), epochs=EPOCHS_PER_TASK)
    results['Expanding Window'] = {
        'baseline': evaluate_model(model, bl_X, bl_y),
        'task4': evaluate_model(model, *task_datasets.get('task4', (bl_X, bl_y)))
    }

    return results

# ==========================================
# 6. SEQUENTIAL CL PIPELINE (with retraining)
# ==========================================
def run_continual_learning(method_name, input_dim, task_datasets, seed=42):
    """
    Full CL pipeline: Baseline → Task1-4 → Retrain Baseline.
    Evaluates on ALL tasks after every training step.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    hp = HYPERPARAMS[method_name]
    is_bayesian    = method_name in ['VCL', 'VCL+Replay', 'EVCL', 'EVCL+Replay']
    uses_replay    = 'Replay' in method_name
    uses_ewc_only  = 'EWC' in method_name and 'VCL' not in method_name
    uses_evcl      = 'EVCL' in method_name

    model = BDNN(input_dim) if is_bayesian else SDNN(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(hp.get('buffer_capacity', 2500)) if uses_replay else None
    ewc_module    = EWC(model) if uses_ewc_only else None
    prior_params  = None
    fisher_dict   = None

    task_names = list(task_datasets.keys())
    # results[task_name] = list of (r2, rmse) after each training phase
    results = {tn: [] for tn in task_names}

    def _eval_all():
        for tn in task_names:
            r2, rmse = evaluate_model(model, *task_datasets[tn], is_bayesian=is_bayesian)
            results[tn].append((r2, rmse))

    def _train_step(X, y, task_idx, is_baseline=False):
        nonlocal prior_params, fisher_dict
        loader = make_dataloader(X, y)

        if is_bayesian:
            beta = 1.0 if is_baseline else hp.get('beta', [1.0]*4)[task_idx]
            lam  = 0.0 if (is_baseline or not uses_evcl) else hp.get('lambda', [0.0]*4)[task_idx]
            train_vcl_with_replay(
                model, loader,
                replay_buffer=replay_buffer or ReplayBuffer(0),
                prior_params=prior_params, beta=beta,
                epochs=EPOCHS_PER_TASK, lr=LEARNING_RATE,
                fisher_dict=fisher_dict, lambda_evcl=lam,
            )
        else:
            if uses_ewc_only and not is_baseline:
                ewc_module.lambda_ewc = hp['lambda'][task_idx]
            train_with_ewc(model, optimizer, loader,
                           ewc_module=ewc_module if not is_baseline else None,
                           replay_buffer=replay_buffer,
                           epochs=EPOCHS_PER_TASK)

        # Consolidate knowledge
        if uses_ewc_only:
            ewc_module.compute_fisher(loader)
        if is_bayesian:
            prior_params = update_prior(model)
            if uses_evcl:
                fisher_dict = compute_fisher_bdnn(model, loader)
        if uses_replay:
            tid = 0 if is_baseline else task_idx + 1
            replay_buffer.update(X, y, task_id=tid)

    print(f"\n{'='*50}\n  Method: {method_name} | Seed: {seed}\n{'='*50}")

    # --- Step 0: Baseline ---
    bl_X, bl_y = task_datasets['baseline']
    print(f"\n  [BL] Training on {len(bl_X)} samples...")
    _train_step(bl_X, bl_y, task_idx=0, is_baseline=True)
    _eval_all()
    print(f"  [BL] R2 baseline: {results['baseline'][-1][0]:.4f}")

    # --- Steps 1-4: Sequential Tasks ---
    for idx, tk in enumerate(['task1', 'task2', 'task3', 'task4']):
        if tk not in task_datasets:
            continue
        t_X, t_y = task_datasets[tk]
        print(f"\n  [{tk.upper()}] Training on {len(t_X)} samples...")
        _train_step(t_X, t_y, task_idx=idx)
        _eval_all()
        # Save checkpoints for each task key so that 03_run_optimization.py can find them.
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f'{method_name}_{tk}.pt'))
        print(f"  [{tk.upper()}] R2 {tk}: {results[tk][-1][0]:.4f}  |  "
              f"R2 baseline: {results['baseline'][-1][0]:.4f}")

    # --- Step 5: Retrain on Baseline (recovery test) ---
    print(f"\n  [RE-TRAIN BL] Retraining on baseline after Task 4...")
    _train_step(bl_X, bl_y, task_idx=0, is_baseline=True)
    _eval_all()
    print(f"  [RE-TRAIN] R2 baseline: {results['baseline'][-1][0]:.4f}")

    # Save final model checkpoint for neural inversion
    save_path = os.path.join(MODELS_DIR, f'{method_name}_final_seed{seed}.pt')
    torch.save(model.state_dict(), save_path)
    print(f"  Model saved -> {save_path}")

    return results

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Loading datasets...")
    task_datasets = {}

    bl_path = os.path.join(DATA_DIR, 'baseline', 'baseline_42k.csv')
    if os.path.exists(bl_path):
        task_datasets['baseline'] = load_task_data(bl_path)
    else:
        print(f"  ERROR: {bl_path} not found. Run 01_generate_data.py first.")
        sys.exit(1)

    for i in range(1, 5):
        tp = os.path.join(DATA_DIR, 'tasks', f'task{i}.csv')
        if os.path.exists(tp):
            task_datasets[f'task{i}'] = load_task_data(tp)

    test_path = os.path.join(DATA_DIR, 'test_3k.csv')
    if os.path.exists(test_path):
        test_X, test_y = load_task_data(test_path)
    else:
        raise FileNotFoundError(
            f"Test dataset not found at: {test_path}\n"
            f"Please run 01_generate_data.py first to generate all task and test datasets."
        )

    input_dim = task_datasets['baseline'][0].shape[1]
    print(f"Input dimension: {input_dim}")
    bl_X, bl_y = task_datasets['baseline']

    # ── A. 8-Fold Cross-Validation (Table 2) ──
    cv_results = run_kfold_baseline(input_dim, bl_X, bl_y, test_X, test_y, task_datasets=task_datasets)

    # ── B. Baseline Comparisons (Table 3) ──
    print("\n" + "=" * 60)
    print("  BASELINE COMPARISONS (Table 3)")
    print("=" * 60)
    comparison_results = {}
    for seed in range(NUM_SEEDS):
        res = run_baseline_comparisons(input_dim, task_datasets, seed=seed)
        for method, metrics in res.items():
            comparison_results.setdefault(method, {'bl_r2': [], 'bl_rmse': [], 't4_r2': [], 't4_rmse': []})
            comparison_results[method]['bl_r2'].append(metrics['baseline'][0])
            comparison_results[method]['bl_rmse'].append(metrics['baseline'][1])
            comparison_results[method]['t4_r2'].append(metrics['task4'][0])
            comparison_results[method]['t4_rmse'].append(metrics['task4'][1])

    print("\n  == Table 3: Stability-Plasticity Balance ==")
    for m, v in comparison_results.items():
        print(f"\n  {m}:")
        print(f"    Baseline  RMSE: {np.mean(v['bl_rmse']):.0f}+/-{np.std(v['bl_rmse']):.0f}  "
              f"R2: {np.mean(v['bl_r2']):.3f}+/-{np.std(v['bl_r2']):.3f}")
        print(f"    Task 4    RMSE: {np.mean(v['t4_rmse']):.0f}+/-{np.std(v['t4_rmse']):.0f}  "
              f"R2: {np.mean(v['t4_r2']):.3f}+/-{np.std(v['t4_r2']):.3f}")

    # ── C. All 7 CL Methods × 5 Seeds ──
    print("\n" + "=" * 60)
    print("  CONTINUAL LEARNING METHODS (7 methods × 5 seeds)")
    print("=" * 60)

    all_results = {}
    for method in HYPERPARAMS.keys():
        seed_runs = []
        for seed in range(NUM_SEEDS):
            res = run_continual_learning(method, input_dim, task_datasets, seed=seed)
            seed_runs.append(res)
        all_results[method] = seed_runs

    # ── D. Export results in plotting-template format ──
    # The plotting template expects per-method dicts:
    #   {"RMSE": {"Base Task": [6 values], "Task 1": [6 values], ...},
    #    "R2":   {"Base Task": [6 values], "Task 1": [6 values], ...}}
    # where 6 values = [after BL, after T1, after T2, after T3, after T4, after Retrain BL]

    print("\n" + "=" * 60)
    print("  EXPORTING RESULTS FOR PLOTTING")
    print("=" * 60)

    import json

    # Map internal task keys to plotting template labels
    task_label_map = {
        'baseline': 'Base Task',
        'task1': 'Task 1',
        'task2': 'Task 2',
        'task3': 'Task 3',
        'task4': 'Task 4',
    }

    # --- D1. Per-method heatmap data (Figure 5a-g) ---
    # Average across seeds for each method
    heatmap_data = {}
    for method, seed_runs in all_results.items():
        method_rmse = {}
        method_r2 = {}

        for tn in ['baseline', 'task1', 'task2', 'task3', 'task4']:
            label = task_label_map[tn]
            # Each seed_run has results[tn] = [(r2, rmse), ...] with 6 entries
            # Average across seeds for each phase
            n_phases = min(len(sr[tn]) for sr in seed_runs) if seed_runs else 0
            rmse_phases = []
            r2_phases = []
            for phase_idx in range(n_phases):
                seed_rmses = [sr[tn][phase_idx][1] for sr in seed_runs]
                seed_r2s   = [sr[tn][phase_idx][0] for sr in seed_runs]
                rmse_phases.append(float(np.mean(seed_rmses)))
                r2_phases.append(float(np.mean(seed_r2s)))
            method_rmse[label] = rmse_phases
            method_r2[label] = r2_phases

        heatmap_data[method] = {"RMSE": method_rmse, "R2": method_r2}

    # Save heatmap data as JSON (loadable by plotting script)
    with open(os.path.join(RESULTS_DIR, 'heatmap_data.json'), 'w') as f:
        json.dump(heatmap_data, f, indent=4)
    print(f"  [SUCCESS] Heatmap data -> results/heatmap_data.json")

    # --- D2. Bar chart data (Figure 5h-i) ---
    bar_data = {
        'methods': [],
        'r2_after_task4': [],
        'r2_after_retraining': [],
    }
    for method, seed_runs in all_results.items():
        bar_data['methods'].append(method)
        # Mean R² across ALL tasks after Task 4 training (index 4)
        r2_t4_all = []
        for sr in seed_runs:
            task_r2s = [sr[tn][4][0] for tn in sr.keys() if len(sr[tn]) > 4]
            r2_t4_all.append(np.mean(task_r2s) if task_r2s else 0.0)
        bar_data['r2_after_task4'].append(float(np.mean(r2_t4_all)))

        # Mean R² across ALL tasks after BL retrain (index 5 / last)
        r2_ret_all = []
        for sr in seed_runs:
            task_r2s = [sr[tn][-1][0] for tn in sr.keys()]
            r2_ret_all.append(np.mean(task_r2s))
        bar_data['r2_after_retraining'].append(float(np.mean(r2_ret_all)))

    with open(os.path.join(RESULTS_DIR, 'bar_chart_data.json'), 'w') as f:
        json.dump(bar_data, f, indent=4)
    print(f"  [SUCCESS] Bar chart data -> results/bar_chart_data.json")

    # --- D3. Full per-phase CSV (for custom analysis) ---
    phase_rows = []
    for method, seed_runs in all_results.items():
        for seed_idx, sr in enumerate(seed_runs):
            for tn in sr.keys():
                label = task_label_map.get(tn, tn)
                for phase_idx, (r2, rmse) in enumerate(sr[tn]):
                    phase_rows.append({
                        'Method': method,
                        'Seed': seed_idx,
                        'Evaluated_On': label,
                        'Phase': phase_idx,
                        'R2': r2,
                        'RMSE': rmse,
                    })
    pd.DataFrame(phase_rows).to_csv(
        os.path.join(RESULTS_DIR, 'cl_per_phase_results.csv'), index=False
    )
    print(f"  [SUCCESS] Per-phase results -> results/cl_per_phase_results.csv")

    # --- D4. Summary table (cl_summary.csv) ---
    summary_rows = []
    for method, seed_runs in all_results.items():
        bl_after_t4  = [sr['baseline'][4][0] for sr in seed_runs if len(sr['baseline']) > 4]
        t4_after_t4  = [sr.get('task4', [(0,0)]*5)[4][0] for sr in seed_runs if 'task4' in sr and len(sr['task4']) > 4]
        bl_after_ret = [sr['baseline'][-1][0] for sr in seed_runs]
        mean_r2_all  = []
        for sr in seed_runs:
            task_r2s = [sr[tn][-1][0] for tn in sr.keys()]
            mean_r2_all.append(np.mean(task_r2s))

        row = {
            'Method': method,
            'BL R2 (after T4)': f"{np.mean(bl_after_t4):.3f}+/-{np.std(bl_after_t4):.3f}" if bl_after_t4 else "N/A",
            'T4 R2 (after T4)': f"{np.mean(t4_after_t4):.3f}+/-{np.std(t4_after_t4):.3f}" if t4_after_t4 else "N/A",
            'BL R2 (retrained)': f"{np.mean(bl_after_ret):.3f}+/-{np.std(bl_after_ret):.3f}",
            'Mean R2 (final)': f"{np.mean(mean_r2_all):.3f}+/-{np.std(mean_r2_all):.3f}",
        }
        summary_rows.append(row)
        print(f"\n  {method}:")
        for k, v in row.items():
            if k != 'Method':
                print(f"    {k}: {v}")

    pd.DataFrame(summary_rows).to_csv(os.path.join(RESULTS_DIR, 'cl_summary.csv'), index=False)

    print("\n[SUCCESS] All training complete. Results saved to results/")
