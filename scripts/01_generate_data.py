"""
01_generate_data.py
====================
Generates surrogate model training data via PySWMM simulations.

Per the paper:
  - Baseline: 14 precipitation events × 3,000 samples = 42,000 samples
  - Tasks 1-4: 4 future projection events × 3,000 samples = 12,000 samples
  - Test: 1 event (Aug 27, 2020) × 3,000 samples = 3,000 samples

Each "sample" = one full simulation with STATIC random actions → scalar Total CSO.
"""

import os
import numpy as np
import pandas as pd
import torch

from pyswmm import Simulation, Nodes, Links, Subcatchments, RainGages
from gymnasium import Env, spaces

# Environment configurations
os.environ['CONDA_DLL_SEARCH_MODIFICATION_ENABLE'] = "1"

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Relative paths for data and SWMM input file
RAINFALL_PATH     = os.path.join(PROJECT_ROOT, 'data', 'precipitation', 'atlas14_15min_pcp.csv')
INPUT_FILE_PATH   = os.path.join(PROJECT_ROOT, 'data', 'swmm_model', '7MPF_CSO_08.inp')

# Output directories
BASE_OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'data')
BASELINE_DIR      = os.path.join(BASE_OUTPUT_DIR, 'baseline')
TASK_DIR          = os.path.join(BASE_OUTPUT_DIR, 'tasks')

NUM_SAMPLES_PER_EVENT = 3000   # Paper: 3,000 samples per precipitation event

# ==========================================
# 2. PRECIPITATION EVENTS
# ==========================================

if not os.path.exists(RAINFALL_PATH):
    raise FileNotFoundError(
        f"Precipitation data file not found at: {RAINFALL_PATH}\n"
        f"Please place your rainfall CSV file there before running data generation."
    )

rainfall_data = pd.read_csv(RAINFALL_PATH)

# --- Baseline events (14 total) ---
# 5 synthetic return periods from NOAA Atlas 14 (paper: "5, 10, 15, 20 and 25-year")
atlas14_columns = ["RT5", "RT10", "RT15", "RT20", "RT25"]

# 6 historical storm records (1980-2010)
historical_columns = ["Hist_1", "Hist_2", "Hist_3", "Hist_4", "Hist_5", "Hist_6"]

# 3 bias-corrected GCM events
gcm_baseline_columns = ["GCM_BL_1", "GCM_BL_2", "GCM_BL_3"]

BASELINE_EVENTS = atlas14_columns + historical_columns + gcm_baseline_columns  # 14 events

# --- Task events (4 total: 2 CANESM + 2 HadGEM under rcp8.5) ---
TASK_EVENTS = {
    "Task1": "CANESM_rcp85_1",    # Future RCM projection event 1
    "Task2": "CANESM_rcp85_2",    # Future RCM projection event 2
    "Task3": "HadGEM_rcp85_1",    # Future RCM projection event 3
    "Task4": "HadGEM_rcp85_2",    # Future RCM projection event 4
}

# --- Test event ---
TEST_EVENT = "Aug2020"  # August 27, 2020 extreme event


def load_precipitation(event_name):
    """
    Loads precipitation array for a given event name from the rainfall CSV.
    Appends 3 trailing zero rows (post-event drainage period).
    
    NOTE: Adjust column name mapping to match your actual CSV headers.
    """
    if event_name in rainfall_data.columns:
        rain = rainfall_data[event_name].values
    else:
        raise ValueError(f"Event '{event_name}' not found in {RAINFALL_PATH}. "
                         f"Available columns: {list(rainfall_data.columns)}")
    
    # Append 3 trailing zeros for post-storm drainage
    rain = np.append(rain, [0.0, 0.0, 0.0])
    return rain


# ==========================================
# 3. PYSWMM GYMNASIUM ENVIRONMENT
# ==========================================

class SWMMSurrogateEnv(Env):
    """
    PySWMM-based Gymnasium environment for surrogate data generation.
    Runs a full stormwater simulation with STATIC control actions and
    accumulates total CSO volume (scalar output).
    """
    def __init__(self, input_file=INPUT_FILE_PATH):
        super().__init__()
        self.input_file = input_file
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(
                f"SWMM input file (.inp) not found at: {self.input_file}\n"
                f"Please place your SWMM model file there before running data generation."
            )
        self.control_time_step = 900  # 15-minute step

        self.sim = Simulation(self.input_file)
        self.sim.step_advance(self.control_time_step)

        node_object    = Nodes(self.sim)
        link_object    = Links(self.sim)
        subcatch_object = Subcatchments(self.sim)
        raingages      = RainGages(self.sim)

        # Node tracking
        self.list_nodes = [node.nodeid for node in node_object]
        self.all_nodes  = [node_object[nid] for nid in self.list_nodes]

        # Managed hydraulic links (7 orifices/weirs)
        self.link_keys = ["1360", "1360_B", "35263", "35281_B", "35291_B", "3530", "3530_B"]
        self.links = {k: link_object[k] for k in self.link_keys}

        # Managed pumps (3)
        self.pump_keys = ["PUMP27@35395-3535", "PUMP28@35390-3535", "PUMP29@35445-3565"]
        self.pumps = {k: link_object[k] for k in self.pump_keys}

        # CSO outfall monitors
        self.W35391 = link_object["35391"]
        self.W35444 = link_object["35444"]

        # Rain gage
        self.sub1 = subcatch_object["900"]
        self.rg1  = raingages['Design_RG']

        self.sim.start()
        self._set_initial_targets()

        sim_len = self.sim.end_time - self.sim.start_time
        self.T = int(sim_len.total_seconds() / self.control_time_step) - 1
        self.t = 1

        # Action = 7 gate openings + 3 pump settings = 10 continuous [0, 1]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

    def _set_initial_targets(self):
        for link in self.links.values():
            link.target_setting = 0.5
        for pump in self.pumps.values():
            pump.target_setting = 0.0

    def step(self, action, precipitation):
        """Advance one timestep with given static action and precipitation."""
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        # Apply gate settings
        for idx, lk in enumerate(self.link_keys):
            self.links[lk].target_setting = float(action[idx])
        for idx, pk in enumerate(self.pump_keys):
            self.pumps[pk].target_setting = float(action[7 + idx])

        # Inject precipitation
        precip_val = precipitation[self.t - 1] if self.t - 1 < len(precipitation) else 0.0
        self.rg1.total_precip = float(max(precip_val, 0.0))

        # Advance SWMM engine
        self.sim.__next__()

        # CSO flow at outfalls (absolute value)
        cso_flow = abs(self.W35391.flow) + abs(self.W35444.flow)

        terminated = self.t >= self.T - 1
        self.t += 1

        return cso_flow, terminated

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.close()

        self.sim = Simulation(self.input_file)
        self.sim.step_advance(self.control_time_step)

        node_object     = Nodes(self.sim)
        link_object     = Links(self.sim)
        subcatch_object = Subcatchments(self.sim)
        raingages       = RainGages(self.sim)

        self.all_nodes = [node_object[nid] for nid in self.list_nodes]
        self.links  = {k: link_object[k] for k in self.link_keys}
        self.pumps  = {k: link_object[k] for k in self.pump_keys}
        self.W35391 = link_object["35391"]
        self.W35444 = link_object["35444"]
        self.sub1   = subcatch_object["900"]
        self.rg1    = raingages['Design_RG']

        self.sim.start()
        self.t = 1
        self._set_initial_targets()
        return None, {}

    def close(self):
        if hasattr(self, 'sim'):
            self.sim.close()


# ==========================================
# 4. DATA GENERATION ROUTINE
# ==========================================

def generate_dataset_for_event(env, precipitation, num_samples):
    """
    Generates `num_samples` surrogate training rows for one precipitation event.
    
    Per paper methodology:
      - Each sample uses STATIC random actions (fixed for the entire event).
      - The target is the SCALAR total CSO volume accumulated over the event.
    
    Returns:
        rows: list of arrays, each = [rain_t1, ..., rain_tN, act_1, ..., act_10, Total_CSO]
    """
    rows = []
    steps_per_episode = None

    for i in range(num_samples):
        env.reset()
        done = False

        # Paper: "static control actions directly to a scalar total CSO volume"
        static_action = np.random.choice(np.arange(0.0, 1.0, 0.05), size=10)
        total_cso = 0.0

        while not done:
            cso_flow, terminated = env.step(static_action, precipitation)
            total_cso += cso_flow
            done = terminated

        steps_per_episode = env.t - 1
        rain_slice = precipitation[:steps_per_episode]

        row = np.concatenate([rain_slice, static_action, [total_cso]])
        rows.append(row)

        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{num_samples} samples done.")

    return rows, steps_per_episode


def save_dataset(rows, steps_per_episode, output_path):
    """Saves generated dataset to CSV with proper column headers."""
    headers = (
        [f"Rain_{t+1}" for t in range(steps_per_episode)]
        + [f"Action_{a+1}" for a in range(10)]
        + ["Total_CSO"]
    )
    df = pd.DataFrame(np.array(rows), columns=headers)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved → {output_path}  (shape: {df.shape})")
    return df


# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    env = SWMMSurrogateEnv()

    # --- A. Generate Baseline Dataset (14 events × 3,000 = 42,000 samples) ---
    print("=" * 60)
    print("BASELINE DATA GENERATION (14 events × 3,000 samples)")
    print("=" * 60)

    all_baseline_rows = []
    for event_name in BASELINE_EVENTS:
        print(f"\n  Event: {event_name}")
        rain = load_precipitation(event_name)
        rows, steps = generate_dataset_for_event(env, rain, NUM_SAMPLES_PER_EVENT)
        all_baseline_rows.extend(rows)

    save_dataset(all_baseline_rows, steps,
                 os.path.join(BASELINE_DIR, "baseline_42k.csv"))

    # --- B. Generate Task Datasets (4 events × 3,000 = 12,000 samples) ---
    print("\n" + "=" * 60)
    print("TASK DATA GENERATION (4 events × 3,000 samples each)")
    print("=" * 60)

    for task_label, event_name in TASK_EVENTS.items():
        print(f"\n  {task_label}: {event_name}")
        rain = load_precipitation(event_name)
        rows, steps = generate_dataset_for_event(env, rain, NUM_SAMPLES_PER_EVENT)
        save_dataset(rows, steps,
                     os.path.join(TASK_DIR, f"{task_label.lower()}.csv"))

    # --- C. Generate Test Dataset (1 event × 3,000 samples) ---
    print("\n" + "=" * 60)
    print("TEST DATA GENERATION (1 event × 3,000 samples)")
    print("=" * 60)

    rain = load_precipitation(TEST_EVENT)
    rows, steps = generate_dataset_for_event(env, rain, NUM_SAMPLES_PER_EVENT)
    save_dataset(rows, steps,
                 os.path.join(BASE_OUTPUT_DIR, "test_3k.csv"))

    env.close()
    print("\n✓ All data generation complete.")
