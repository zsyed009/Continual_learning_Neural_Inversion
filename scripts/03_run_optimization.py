"""
03_run_optimization.py
======================
Neural Inversion execution, SWMM validation, and storage usage analysis.

Implements the paper's Sections 3.3 and 3.4:
  - Runs neural inversion on each CL-trained model across Tasks 1-4
  - Validates optimized actions inside the SWMM numerical model
  - Computes MAE between predicted and actual CSOs (Figure 6)
  - Tracks storage utilization percentage over time (Figure 7, Table 4)
"""

import os, sys
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.optimization.neural_inversion import NeuralInversionOptimizer
from src.models.surrogate_model import SDNN
from src.models.bayesian_dnn import BDNN

# Only import PySWMM components when available
try:
    from pyswmm import Simulation, Nodes, Links, Subcatchments, RainGages
    SWMM_AVAILABLE = True
except ImportError:
    SWMM_AVAILABLE = False
    print("WARNING: pyswmm not available. SWMM validation will be skipped.")


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_DIR       = os.path.join(PROJECT_ROOT, 'data')
RESULTS_DIR    = os.path.join(PROJECT_ROOT, 'results')
MODELS_DIR     = os.path.join(PROJECT_ROOT, 'results', 'trained_models')
INPUT_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'swmm_model', '7MPF_CSO_08.inp')

os.makedirs(RESULTS_DIR, exist_ok=True)

N_OPTIMIZATION_TRIALS = 100   # Paper: 100 independent runs per event
TASK_KEYS = ['task1', 'task2', 'task3', 'task4']

# CL methods to evaluate (all 7 + No CL baseline, matching Table 4)
CL_METHODS = [
    'No CL',            # Naive Fine-Tuning baseline (Table 4 column 1)
    'EWC', 'EWC+Replay',
    'VCL', 'VCL+Replay',
    'EVCL', 'EVCL+Replay',
    'Replay'
]

# Storage node IDs in the PFSMC SWMM model (4 storage units)
STORAGE_NODE_IDS = ["SU1", "SU2", "SU3", "SU4"]  # Adjust to match actual SWMM model


# ==========================================
# 2. LOAD PRECIPITATION FOR TASKS
# ==========================================

def load_task_precipitation(task_key):
    """
    Loads the precipitation array for a given task.
    Extracts rain columns from the task CSV (Rain_1 through Rain_N).
    """
    csv_path = os.path.join(DATA_DIR, 'tasks', f'{task_key}.csv')
    df = pd.read_csv(csv_path)
    rain_cols = [c for c in df.columns if c.startswith('Rain_')]
    # All rows share the same precipitation; take the first row
    precipitation = df[rain_cols].iloc[0].values.astype(np.float32)
    return precipitation


# ==========================================
# 3. SWMM VALIDATION (Section 3.3)
# ==========================================

def validate_actions_in_swmm(optimized_actions, precipitation):
    """
    Applies the optimized control actions from neural inversion into the
    actual SWMM numerical model and measures the resulting actual CSO volume
    and storage utilization over time.

    Per paper: "testing these optimized actions within the physical SWMM
    environment is required to validate prescriptive feasibility"

    Args:
        optimized_actions (np.ndarray): Static gate/pump settings [10].
        precipitation (np.ndarray): Rain array for the event.

    Returns:
        actual_cso (float): Total CSO volume from SWMM simulation.
        storage_timeseries (list of float): Storage utilization % at each step.
    """
    if not SWMM_AVAILABLE:
        raise ImportError("pyswmm package is required to run SWMM validation.")

    if not os.path.exists(INPUT_FILE_PATH):
        raise FileNotFoundError(
            f"SWMM input file (.inp) not found at: {INPUT_FILE_PATH}\n"
            f"Please place your SWMM model file there before running validation."
        )

    sim = Simulation(INPUT_FILE_PATH)
    sim.step_advance(900)  # 15-min steps

    node_object = Nodes(sim)
    link_object = Links(sim)
    raingages   = RainGages(sim)

    # Managed links and pumps
    link_keys = ["1360", "1360_B", "35263", "35281_B", "35291_B", "3530", "3530_B"]
    pump_keys = ["PUMP27@35395-3535", "PUMP28@35390-3535", "PUMP29@35445-3565"]
    links = {k: link_object[k] for k in link_keys}
    pumps = {k: link_object[k] for k in pump_keys}

    # CSO outfall monitors
    W35391 = link_object["35391"]
    W35444 = link_object["35444"]

    # Storage nodes for utilization tracking
    storage_nodes = {}
    for sid in STORAGE_NODE_IDS:
        try:
            storage_nodes[sid] = node_object[sid]
        except Exception:
            pass  # Node may not exist in this model configuration

    rg1 = raingages['Design_RG']

    sim.start()

    # Apply static optimized actions
    for idx, lk in enumerate(link_keys):
        links[lk].target_setting = float(optimized_actions[idx])
    for idx, pk in enumerate(pump_keys):
        pumps[pk].target_setting = float(optimized_actions[7 + idx])

    total_cso = 0.0
    storage_timeseries = []
    t = 0

    for step in sim:
        # Inject precipitation
        precip_val = precipitation[t] if t < len(precipitation) else 0.0
        rg1.total_precip = float(max(precip_val, 0.0))

        # Accumulate CSO flow at outfalls
        total_cso += abs(W35391.flow) + abs(W35444.flow)

        # Compute storage utilization percentage
        if storage_nodes:
            usage_pcts = []
            for sid, snode in storage_nodes.items():
                if snode.full_depth > 0:
                    usage_pcts.append((snode.depth / snode.full_depth) * 100.0)
            avg_storage_pct = np.mean(usage_pcts) if usage_pcts else 0.0
            storage_timeseries.append(avg_storage_pct)

        t += 1

    sim.close()
    return total_cso, storage_timeseries


# ==========================================
# 4. OPTIMIZATION + VALIDATION PIPELINE
# ==========================================

def run_optimization_for_method(model, method_name, task_key, precipitation, is_bayesian):
    """
    Runs neural inversion (100 trials) and validates in SWMM.

    Per paper (Section 3.3):
      "We measured the Mean Absolute Error (MAE) between the simulated
      (actual) CSOs and the model's predicted CSOs; the distribution of
      these errors across the 100 trials is visualized in Figure 6."

    Returns:
        results_dict with predicted CSOs, actual CSOs, MAEs, storage data.
    """
    print(f"\n  [{method_name}] Optimizing for {task_key}...")

    optimizer = NeuralInversionOptimizer(
        model=model,
        precipitation=precipitation,
        lr=0.002,           # Paper: η = 0.002
        max_iters=200,      # Paper: max 200 iterations
        is_bayesian=is_bayesian,
    )

    # Run 100 independent optimizations from random starts
    all_actions, all_predicted_cso, all_histories = optimizer.optimize_multi_start(
        n_starts=N_OPTIMIZATION_TRIALS
    )

    # Validate each set of optimized actions in SWMM
    all_actual_cso = []
    all_mae = []
    all_storage = []

    for trial_idx, (actions, pred_cso) in enumerate(zip(all_actions, all_predicted_cso)):
        actual_cso, storage_ts = validate_actions_in_swmm(actions, precipitation)

        if actual_cso is not None:
            mae = abs(actual_cso - pred_cso)
            all_actual_cso.append(actual_cso)
            all_mae.append(mae)
            all_storage.append(storage_ts)
        else:
            # SWMM not available — record predicted only
            all_actual_cso.append(None)
            all_mae.append(None)
            all_storage.append(None)

    return {
        'method': method_name,
        'task': task_key,
        'predicted_cso': all_predicted_cso,
        'actual_cso': all_actual_cso,
        'mae': all_mae,
        'actions': all_actions,
        'storage_timeseries': all_storage,
        'convergence_histories': all_histories,
    }


# ==========================================
# 5. STORAGE USAGE ANALYSIS (Section 3.4)
# ==========================================

def compute_storage_metrics(results_list):
    """
    Computes median storage usage (%) across CL methods and tasks.

    Per paper (Table 4): "Median Storage Usage (%) for CL Methods Across Tasks"
    "VCL with Replay achieved the highest median storage (e.g., 50.44% for Task 1)"

    Args:
        results_list: List of result dicts from run_optimization_for_method.

    Returns:
        storage_table (pd.DataFrame): Table 4 formatted output.
    """
    rows = []
    for res in results_list:
        method = res['method']
        task = res['task']

        # Gather median storage from all trials
        median_storages = []
        for storage_ts in res['storage_timeseries']:
            if storage_ts is not None and len(storage_ts) > 0:
                median_storages.append(np.median(storage_ts))

        median_val = np.median(median_storages) if median_storages else None
        rows.append({'Method': method, 'Task': task, 'Median_Storage_Pct': median_val})

    df = pd.DataFrame(rows)

    # Pivot to Table 4 format: rows = tasks, columns = methods
    if not df.empty and 'Median_Storage_Pct' in df.columns:
        storage_table = df.pivot(index='Task', columns='Method', values='Median_Storage_Pct')
        return storage_table
    return df


# ==========================================
# 6. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("  NEURAL INVERSION & STORAGE ANALYSIS")
    print("=" * 60)

    # Load trained models (assumed saved after 02_train_continual.py)
    # In practice, models would be loaded from saved checkpoints.
    # Here we demonstrate the pipeline structure.

    all_results = []

    for task_key in TASK_KEYS:
        print(f"\n{'='*50}")
        print(f"  Processing: {task_key.upper()}")
        print(f"{'='*50}")

        # Load precipitation for this task
        try:
            precipitation = load_task_precipitation(task_key)
        except FileNotFoundError:
            print(f"  WARNING: Data for {task_key} not found. Skipping.")
            continue

        input_dim = len(precipitation) + 10  # rain features + 10 actions

        for method_name in CL_METHODS:
            # Determine model type (No CL and EWC variants use SDNN; VCL/EVCL use BDNN)
            is_bayesian = method_name in ['VCL', 'VCL+Replay', 'EVCL', 'EVCL+Replay']

            # Load or create model (in production, load from checkpoint)
            model_path = os.path.join(MODELS_DIR, f'{method_name}_{task_key}.pt')
            if os.path.exists(model_path):
                model = BDNN(input_dim) if is_bayesian else SDNN(input_dim)
                model.load_state_dict(torch.load(model_path))
                print(f"  Loaded model: {model_path}")
            else:
                raise FileNotFoundError(
                    f"Model checkpoint not found at: {model_path}\n"
                    f"Please run 02_train_continual.py first to train the models and save checkpoints."
                )

            # Run optimization + SWMM validation
            res = run_optimization_for_method(
                model, method_name, task_key, precipitation, is_bayesian
            )
            all_results.append(res)

    # ── A. MAE Summary (Figure 6) ──
    print("\n" + "=" * 60)
    print("  MAE SUMMARY (Figure 6)")
    print("=" * 60)

    mae_rows = []
    for res in all_results:
        valid_mae = [m for m in res['mae'] if m is not None]
        if valid_mae:
            mae_rows.append({
                'Method': res['method'],
                'Task': res['task'],
                'Median_MAE': np.median(valid_mae),
                'Mean_MAE': np.mean(valid_mae),
                'Std_MAE': np.std(valid_mae),
                'IQR_MAE': np.percentile(valid_mae, 75) - np.percentile(valid_mae, 25),
            })
            print(f"  {res['method']:20s} | {res['task']} | "
                  f"Median MAE: {np.median(valid_mae):8.1f} m3 | "
                  f"IQR: {mae_rows[-1]['IQR_MAE']:8.1f} m3")

    mae_df = pd.DataFrame(mae_rows)
    mae_df.to_csv(os.path.join(RESULTS_DIR, 'neural_inversion_mae.csv'), index=False)

    # ── B. Storage Usage (Table 4) ──
    print("\n" + "=" * 60)
    print("  STORAGE UTILIZATION (Table 4)")
    print("=" * 60)

    storage_table = compute_storage_metrics(all_results)
    if storage_table is not None and not storage_table.empty:
        print(storage_table.to_string())
        storage_table.to_csv(os.path.join(RESULTS_DIR, 'storage_usage_table4.csv'))
    else:
        print("  No SWMM storage data available (pyswmm not installed or no data).")

    # ── C. Save all predicted CSOs for further analysis ──
    pred_rows = []
    for res in all_results:
        for trial_idx, (pred, actual) in enumerate(zip(res['predicted_cso'], res['actual_cso'])):
            pred_rows.append({
                'Method': res['method'],
                'Task': res['task'],
                'Trial': trial_idx,
                'Predicted_CSO': pred,
                'Actual_CSO': actual,
            })
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(RESULTS_DIR, 'optimization_results.csv'), index=False
    )

    # ── D. Save storage timeseries for Figure 7 plotting ──
    # Uses the best trial (lowest actual CSO) per method/task
    import json
    storage_ts_data = {}
    for res in all_results:
        method = res['method']
        task = res['task']

        # Find the best trial (lowest actual CSO or predicted if no SWMM)
        valid_trials = [(i, a if a is not None else p)
                        for i, (a, p) in enumerate(zip(res['actual_cso'], res['predicted_cso']))]
        if not valid_trials:
            continue
        best_idx = min(valid_trials, key=lambda x: x[1])[0]

        ts = res['storage_timeseries'][best_idx]
        if ts is not None and len(ts) > 0:
            storage_ts_data.setdefault(task, {})[method] = ts

    # Also save the precipitation arrays for the dual-axis plot
    precip_data = {}
    for task_key in TASK_KEYS:
        try:
            precip_data[task_key] = load_task_precipitation(task_key).tolist()
        except Exception:
            pass

    with open(os.path.join(RESULTS_DIR, 'storage_timeseries.json'), 'w') as f:
        json.dump({'storage': storage_ts_data, 'precipitation': precip_data}, f)
    print(f"  [SUCCESS] Storage timeseries -> results/storage_timeseries.json")

    print(f"\n[SUCCESS] Neural inversion complete. Results saved to: {RESULTS_DIR}")
