import wandb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def get_wandb_data(run_paths):
    api = wandb.Api()
    data = []

    for path in run_paths:
        try:
            run = api.run(path)
            
            # Fetch config args
            config = run.config
            row = {
                "Run ID": run.name,
                "Model": config.get("model_name", "N/A"),
                "Seq Len": config.get("seq_len", "N/A"),
                "DP": config.get("data_parallel", "N/A"),
                "World Size": config.get("world_size", "N/A"),
                "TP": config.get("tp_size", "N/A"),
            }
            #breakpoint()

            # Calculate averages from history
            history = run.history(keys=["perf/tps_per_gcd", "perf/tflops_per_gpu", "perf/mfu"])
            row["Avg Tokens/s"] = history["perf/tps_per_gcd"].mean()
            row["TFLOPS/s"] = history["perf/tflops_per_gpu"].mean()
            row["MFU"] = history["perf/mfu"].mean()
            #breakpoint()
            data.append(row)
        except Exception as e:
            print(f"Error fetching {path}: {e}")

    return pd.DataFrame(data)

# PUT THE RUNS IDS HERE
run_paths = [
"shanshan-wang-csc-csc/qwen3-vl_8b/jvldxa7c",
"shanshan-wang-csc-csc/qwen3-vl_8b/7pfx6v2q"
]

# Fetch data and sort by GPU count to ensure correct X-axis ordering
df = get_wandb_data(run_paths)
#breakpoint()
df = df.sort_values(by="World Size").reset_index(drop=True)

# Enable LaTeX text rendering
"""
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 11
})
"""

# --- DATA ---
# Extract directly from WandB DataFrame
gpus = df["World Size"].to_numpy()
seq_len = df["Seq Len"].to_numpy()
tps_per_gpu = df["Avg Tokens/s"].to_numpy()
tflops_per_gpu = df["TFLOPS/s"].to_numpy()

# Automatic calculations
measured_total = gpus * tps_per_gpu
optimal_total = gpus * tps_per_gpu[0]  
efficiency = (measured_total / optimal_total) * 100

# Categorical X-axis positions
x_pos = np.arange(len(gpus))

# --- PLOT SETUP ---
fig, (ax1, ax3) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6), sharex=True)
fig.subplots_adjust(hspace=0.08)

# --- TOP PANEL: Total Throughput ---
ax1.set_title(r"Weak Scaling Qwen3VL-8B on LUMI")
ax1.set_ylabel(r"Tokens/sec")
ax1.grid(True, linestyle='--', alpha=0.5, axis='y')

# Format left Y-axis to millions
ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x*1e-6:.1f}M' if x != 0 else '0M'))

# Adjusted bar width for categorical spacing (0 to 1 scale)
bar_width = 0.4 
ax1.bar(x_pos, measured_total, width=bar_width, color='#A9DBF4', zorder=2)
l1, = ax1.plot(x_pos, measured_total, marker='o', color='blue', label=r"Measured", zorder=3)
l2, = ax1.plot(x_pos, optimal_total, marker='o', color='orange', linestyle='--', label=r"Optimal Scaling", zorder=3)

# --- TOP PANEL: Efficiency (Right Y-Axis) ---
ax2 = ax1.twinx()
ax2.set_ylabel(r"Efficiency (\%)")
ax2.set_ylim(0, 105)
l3, = ax2.plot(x_pos, efficiency, marker='s', color='red', label=r"Efficiency", zorder=4)

# Combine legends
lines = [l1, l2, l3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', framealpha=1.0, edgecolor='black', fancybox=False)

# --- BOTTOM PANEL: Per GPU Metrics ---
ax3.set_xlabel(r"Number of GPUs")
ax3.set_ylabel(r"TFLOPS/s/GPU")
ax3.set_ylim(0, 60)
ax3.grid(True, linestyle='--', alpha=0.5, axis='y')

bars = ax3.bar(x_pos, tflops_per_gpu, width=bar_width, color='#43A047', zorder=2, label=r"TFLOPS/s/GPU")

# Add text labels for TFLOPS
for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval - 100, f'{yval:,.2f}', ha='center', va='bottom', fontsize=9)

# --- NEW: Twin Axis for Tokens/s/GPU in Bottom Panel ---
ax4 = ax3.twinx()
ax4.set_ylabel(r"Tokens/s/GPU")
# Set a dynamic y-lim for the secondary axis so the line floats nicely above/across the bars
ax4.set_ylim(min(tps_per_gpu)*0.8, max(tps_per_gpu)*1.2)

# Plot Tokens/s/GPU as a line
l4, = ax4.plot(x_pos, tps_per_gpu, marker='D', color='purple', label=r"Tokens/s/GPU", zorder=4)

# Add text labels for Tokens/s/GPU
for x, y in zip(x_pos, tps_per_gpu):
    ax4.text(x, y + (max(tps_per_gpu)*0.02), f'{y:,.2f}', ha='center', va='bottom', fontsize=9, color='purple', fontweight='bold')

# Combine legends for the bottom panel
bottom_lines = [bars, l4]
bottom_labels = [l.get_label() for l in bottom_lines]
ax3.legend(bottom_lines, bottom_labels, loc='upper right', framealpha=1.0, edgecolor='black', fancybox=False, fontsize=9)

# Set categorical ticks and labels
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f"{g}\n(Seq: {s})" for g, s in zip(gpus, seq_len)])

# Save and show
plt.savefig("throughput_scaling.pdf", format="pdf", bbox_inches='tight')
plt.show()
