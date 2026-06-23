"""
04_generate_plots.py
====================
Publication-quality figure generation for the CL paper.

Loads saved results from 02_train_continual.py and 03_run_optimization.py,
then generates all paper figures using the formatting conventions from the
plotting template.

Figures produced:
  - Figure 5 (a-g): RMSE + R² heatmaps per CL method across training phases
  - Figure 5 (h-i): Bar charts of mean R² after T4 and after BL retrain
  - Figure 6:       2×2 MAE boxplot from neural inversion (100 trials × 4 tasks)
  - Figure 7:       Storage utilization time series (optional, needs SWMM data)
"""

import os, sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# ==========================================
# COMMON FORMATTING (from plotting template)
# ==========================================

def set_publication_style():
    """Sets matplotlib rcParams to match the paper's formatting conventions."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif',
                        'Bitstream Charter', 'Palatino', 'Charter', 'serif'],
        'font.size': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 16,
        'figure.autolayout': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.linestyle': ':',
        'grid.color': 'lightgray',
        'grid.alpha': 0.7,
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
    })


# Method display names and colors (consistent across all figures)
METHOD_DISPLAY = {
    'EWC':          'EWC',
    'EWC+Replay':   'EWC with Replay',
    'VCL':          'VCL',
    'VCL+Replay':   'VCL with Replay',
    'EVCL':         'EVCL',
    'EVCL+Replay':  'EVCL with Replay',
    'Replay':       'Replay',
    'No CL':        'No CL',
}

METHOD_COLORS = {
    'No CL':            '#1f77b4',
    'EWC':              '#ff7f0e',
    'EWC with Replay':  '#2ca02c',
    'EVCL':             '#d62728',
    'EVCL with Replay': '#9467bd',
    'VCL':              '#8c564b',
    'VCL with Replay':  '#e377c2',
    'Replay':           '#7f7f7f',
}

METHOD_ORDER_BOXPLOT = [
    'No CL', 'EWC', 'EWC with Replay', 'EVCL',
    'EVCL with Replay', 'VCL', 'VCL with Replay', 'Replay'
]

HEATMAP_METHODS = [
    ('EWC',          'Without Replay'),
    ('EWC+Replay',   'With Replay'),
    ('EVCL',         'Without Replay'),
    ('EVCL+Replay',  'With Replay'),
    ('VCL',          'Without Replay'),
    ('VCL+Replay',   'With Replay'),
    ('Replay',       'With Replay'),
]


# ==========================================
# FIGURE 5 (a-g): HEATMAPS + (h-i): BAR CHARTS
# ==========================================

def plot_figure5():
    """
    Generates the combined Figure 5:
      (a)-(g): 7×2 heatmaps showing RMSE (left) and R² (right) per method
      (h)-(i): Bar charts of mean R² after Task 4 and after BL retrain
    """
    # Load data
    heatmap_path = os.path.join(RESULTS_DIR, 'heatmap_data.json')
    bar_path     = os.path.join(RESULTS_DIR, 'bar_chart_data.json')

    if not os.path.exists(heatmap_path):
        print("  WARNING: heatmap_data.json not found. Run 02_train_continual.py first.")
        return
    if not os.path.exists(bar_path):
        print("  WARNING: bar_chart_data.json not found. Run 02_train_continual.py first.")
        return

    with open(heatmap_path) as f:
        heatmap_data = json.load(f)
    with open(bar_path) as f:
        bar_data = json.load(f)

    set_publication_style()
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 10,
                         'axes.titlesize': 12, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'legend.fontsize': 10})

    task_labels_no_replay  = ["Task 0 End", "Task 1 End", "Task 2 End",
                              "Task 3 End", "Task 4 End", "Task 0 (Re-train) End"]
    task_labels_with_replay = ["After Training Task 0", "After Training Task 1",
                               "After Training Task 2", "After Training Task 3",
                               "After Training Task 4", "After Training Task 0 (Re-train) End"]

    fig, axes = plt.subplots(8, 2, figsize=(10, 12),
                             gridspec_kw={'wspace': 0.005, 'hspace': 0.075})
    fig.suptitle('Performance Metrics of Continual Learning Techniques', y=1.0)
    combined_labels = [f'({chr(97 + i)})' for i in range(len(HEATMAP_METHODS) + 2)]

    # --- Heatmaps (a)-(g) ---
    for i, (method_key, replay_status) in enumerate(HEATMAP_METHODS):
        if method_key not in heatmap_data:
            print(f"  Skipping {method_key} — not in results.")
            continue

        current_data = heatmap_data[method_key]
        col_labels = task_labels_with_replay if replay_status == "With Replay" \
                     else task_labels_no_replay
        display_name = METHOD_DISPLAY.get(method_key, method_key)

        rmse_df = pd.DataFrame(current_data["RMSE"], index=col_labels).T
        r2_df   = pd.DataFrame(current_data["R2"], index=col_labels).T

        # RMSE heatmap (left column)
        ax_rmse = axes[i, 0]
        rmse_annot = rmse_df.copy()
        for col in rmse_annot.columns:
            rmse_annot[col] = rmse_annot[col].apply(lambda x: f'{x:.1e}')
        sns.heatmap(rmse_df, annot=rmse_annot, fmt="", cmap="YlGnBu_r",
                    ax=ax_rmse, cbar=True, linewidths=.5, linecolor='white')
        ax_rmse.set_ylabel(display_name)
        ax_rmse.set_xlabel('')
        ax_rmse.set_title('')
        ax_rmse.tick_params(top=False, right=False)
        if i < len(HEATMAP_METHODS) - 1:
            ax_rmse.set_xticklabels([])
        else:
            ax_rmse.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
        ax_rmse.tick_params(axis='y', rotation=0, labelsize=9)

        # R² heatmap (right column)
        ax_r2 = axes[i, 1]
        sns.heatmap(r2_df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
                    ax=ax_r2, cbar=True, linewidths=.5, linecolor='white')
        ax_r2.set_ylabel('')
        ax_r2.set_xlabel('')
        ax_r2.set_title('')
        ax_r2.tick_params(left=False, labelleft=False, top=False, right=False)
        if i < len(HEATMAP_METHODS) - 1:
            ax_r2.set_xticklabels([])
        else:
            ax_r2.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)

        ax_rmse.text(-0.28, 1.0, combined_labels[i], transform=ax_rmse.transAxes,
                     fontsize=8, fontweight='bold', va='top', ha='right')

    # Column titles
    fig.text(0.26, 0.97, 'RMSE', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.72, 0.97, 'R²', ha='center', fontsize=12, fontweight='bold')

    # --- Bar charts (h)-(i) ---
    methods  = bar_data['methods']
    r2_t4    = bar_data['r2_after_task4']
    r2_ret   = bar_data['r2_after_retraining']
    display_names = [METHOD_DISPLAY.get(m, m) for m in methods]
    colors   = [METHOD_COLORS.get(METHOD_DISPLAY.get(m, m), '#333333') for m in methods]
    hatches  = ['/', '\\', '|', '-', '+', 'x', 'o']
    bar_pos  = np.arange(len(methods))

    ax_h = axes[7, 0]
    ax_h.bar(bar_pos, r2_t4, color=colors,
             hatch=[hatches[j % len(hatches)] for j in range(len(methods))])
    ax_h.set_title('h) Mean R² Score After Training All Tasks', fontsize=12)
    ax_h.set_xticks(bar_pos)
    ax_h.set_xticklabels(display_names, rotation=45, ha='right')
    ax_h.set_ylim(0, 1)
    ax_h.grid(axis='y', linestyle='--', alpha=0.7)

    ax_i = axes[7, 1]
    ax_i.bar(bar_pos, r2_ret, color=colors,
             hatch=[hatches[j % len(hatches)] for j in range(len(methods))])
    ax_i.set_title('i) Mean R² Score After Retraining on Base Task', fontsize=12)
    ax_i.set_xticks(bar_pos)
    ax_i.set_xticklabels(display_names, rotation=45, ha='right')
    ax_i.set_ylim(0, 1)
    ax_i.grid(axis='y', linestyle='--', alpha=0.7)
    ax_i.tick_params(left=False, labelleft=False)

    plt.tight_layout(rect=[0.08, 0.08, 1, 0.96])

    save_path = os.path.join(FIGURES_DIR, 'figure5_heatmaps_bars.jpg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [SUCCESS] Figure 5 -> {save_path}")
    plt.close()


# ==========================================
# FIGURE 6: MAE BOXPLOTS (2×2)
# ==========================================

def plot_figure6():
    """
    Generates Figure 6: 2×2 boxplot of MAE distribution per CL method.
    Uses results from neural_inversion_mae.csv or optimization_results.csv.
    """
    opt_path = os.path.join(RESULTS_DIR, 'optimization_results.csv')
    if not os.path.exists(opt_path):
        print("  WARNING: optimization_results.csv not found. Run 03_run_optimization.py first.")
        return

    df = pd.read_csv(opt_path)
    # Compute MAE per trial
    df['MAE'] = (df['Predicted_CSO'] - df['Actual_CSO']).abs()
    df = df.dropna(subset=['MAE'])

    if df.empty:
        print("  WARNING: No valid MAE data (SWMM validation may not have run).")
        return

    # Map method names to display names
    df['Method_Display'] = df['Method'].map(METHOD_DISPLAY).fillna(df['Method'])

    set_publication_style()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    task_display = ['task1', 'task2', 'task3', 'task4']
    task_titles  = ['Task 1', 'Task 2', 'Task 3', 'Task 4']

    boxplot_colors = [METHOD_COLORS.get(m, '#999999') for m in METHOD_ORDER_BOXPLOT]

    for i, (tk, title) in enumerate(zip(task_display, task_titles)):
        ax = axes[i]
        task_df = df[df['Task'] == tk]

        plot_data = []
        for method in METHOD_ORDER_BOXPLOT:
            method_mae = task_df[task_df['Method_Display'] == method]['MAE'].values
            plot_data.append(method_mae if len(method_mae) > 0 else np.array([0]))

        bp = ax.boxplot(plot_data, patch_artist=True, labels=METHOD_ORDER_BOXPLOT,
                        boxprops=dict(facecolor='lightgray', color='black'),
                        medianprops=dict(color='black', linewidth=1.5),
                        whiskerprops=dict(color='black'),
                        capprops=dict(color='black'),
                        flierprops=dict(marker='o', markerfacecolor='grey',
                                       markersize=4, linestyle='none'))

        for patch, color in zip(bp['boxes'], boxplot_colors):
            patch.set_facecolor(color)

        ax.text(0.05, 0.95, subplot_labels[i], transform=ax.transAxes,
                fontsize=14, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
        ax.set_title(title)

        if i >= 2:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        else:
            ax.set_xticklabels('')

        ax.grid(True, linestyle=':', color='lightgray', alpha=0.7)
        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_edgecolor('black')
            ax.spines[side].set_linewidth(1.0)

    fig.supylabel(r"Mean Absolute Error (MAE) (CSO Volume $\mathrm{Ft}^3$)",
                  fontsize=14, x=0.01)
    fig.supxlabel('Method', fontsize=14, y=0.10)
    plt.tight_layout(rect=[0.02, 0.12, 1, 0.98])

    handles = [plt.Rectangle((0,0), 1, 1, color=c) for c in boxplot_colors]
    fig.legend(handles, METHOD_ORDER_BOXPLOT, loc='lower center',
               ncol=4, bbox_to_anchor=(0.5, 0.0))

    save_path = os.path.join(FIGURES_DIR, 'figure6_mae_boxplots.jpg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [SUCCESS] Figure 6 -> {save_path}")
    plt.close()


# ==========================================
# FIGURE 7 (a-d): STORAGE UTILIZATION TIMESERIES
# ==========================================

# Line styles per method (matching template conventions)
METHOD_LINESTYLES = {
    'No CL':            '-',                  # solid
    'EWC':              ':',                  # dotted
    'EWC with Replay':  '--',                 # dashed
    'EVCL':             '-.',                 # dash-dot
    'EVCL with Replay': (0, (3, 1, 1, 1)),    # densely dash-dotted
    'VCL':              (0, (5, 2)),           # long dashed
    'VCL with Replay':  (0, (3, 1, 1, 1, 1, 1)),  # dash-dot-dot
    'Replay':           (0, (1, 1)),           # densely dotted
}


def plot_figure7():
    """
    Generates Figure 7 (a-d): Storage utilization (%) over time for each task.

    Per paper (Section 3.4):
      "Figure 7a-d illustrates how different CL strategies manage a sewer
       system's storage capacity across various tasks (Task 1 to Task 4)."

    Layout: 2×2 subplots. Each subplot has:
      - Primary y-axis (left): Storage utilization %
      - Secondary y-axis (right, inverted): Precipitation intensity (mm)
      - 8 CL method lines overlaid
    """
    ts_path = os.path.join(RESULTS_DIR, 'storage_timeseries.json')
    if not os.path.exists(ts_path):
        print("  WARNING: storage_timeseries.json not found. Run 03_run_optimization.py first.")
        return

    with open(ts_path) as f:
        ts_data = json.load(f)

    storage_data = ts_data.get('storage', {})
    precip_data  = ts_data.get('precipitation', {})

    if not storage_data:
        print("  WARNING: No storage timeseries data available (SWMM may not have run).")
        return

    set_publication_style()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    axes = axes.flatten()
    subplot_labels = ['(a)', '(b)', '(c)', '(d)']
    task_keys  = ['task1', 'task2', 'task3', 'task4']
    task_titles = ['Task 1', 'Task 2', 'Task 3', 'Task 4']

    for i, (tk, title) in enumerate(zip(task_keys, task_titles)):
        ax = axes[i]

        if tk not in storage_data:
            ax.set_title(f'{title} (No data)')
            ax.text(0.05, 0.95, subplot_labels[i], transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))
            continue

        task_storage = storage_data[tk]

        # Plot storage utilization for each CL method
        for method_key, ts_values in task_storage.items():
            display_name = METHOD_DISPLAY.get(method_key, method_key)
            color = METHOD_COLORS.get(display_name, '#333333')
            ls = METHOD_LINESTYLES.get(display_name, '-')

            timesteps = np.arange(len(ts_values))
            # Convert to 15-min timestep x-axis
            ax.plot(timesteps, ts_values,
                    label=display_name, color=color, linestyle=ls,
                    linewidth=1.5, alpha=0.85, zorder=3)

        # --- Secondary y-axis: Precipitation (inverted, top-down) ---
        if tk in precip_data:
            ax2 = ax.twinx()
            precip = precip_data[tk]
            precip_timesteps = np.arange(len(precip))
            ax2.bar(precip_timesteps, precip,
                    color='steelblue', alpha=0.3, width=0.8, zorder=1,
                    label='Precipitation')
            ax2.set_ylim(max(precip) * 3, 0)  # Inverted: bars hang from top
            ax2.set_ylabel('Precipitation (mm)', fontsize=12, color='steelblue')
            ax2.tick_params(axis='y', labelcolor='steelblue')

        # Formatting
        ax.set_title(title, fontsize=14)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle=':', alpha=0.4, zorder=0)

        # Subplot label
        ax.text(0.02, 0.95, subplot_labels[i], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7, ec='none'))

        # Spine styling
        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_edgecolor('black')
            ax.spines[side].set_linewidth(1.0)

        # Axis labels: only on edges
        if i >= 2:  # Bottom row
            ax.set_xlabel('15-Minute Timestep', fontsize=12)
        if i % 2 == 0:  # Left column
            ax.set_ylabel('Storage Utilization (%)', fontsize=12)

    # Shared legend below all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    # Deduplicate
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)

    fig.legend(unique_handles, unique_labels,
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02),
               frameon=True, facecolor='white', edgecolor='gray', fontsize=11)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    save_path = os.path.join(FIGURES_DIR, 'figure7_storage_utilization.jpg')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [SUCCESS] Figure 7 -> {save_path}")
    plt.close()


# ==========================================
# TABLE 4: STORAGE USAGE SUMMARY
# ==========================================

def plot_storage_table():
    """Prints Table 4 from saved storage CSV."""
    storage_path = os.path.join(RESULTS_DIR, 'storage_usage_table4.csv')
    if not os.path.exists(storage_path):
        print("  WARNING: storage_usage_table4.csv not found. Run 03_run_optimization.py first.")
        return

    df = pd.read_csv(storage_path, index_col=0)
    print("\n  == Table 4: Median Storage Usage (%) ==")
    print(df.to_string())
    print(f"  [SUCCESS] Storage table loaded from {storage_path}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("  GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    plot_figure5()
    plot_figure6()
    plot_figure7()
    plot_storage_table()

    print(f"\n[SUCCESS] All figures saved to: {FIGURES_DIR}")
