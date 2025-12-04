import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define Data ---

# X-axis labels: Nine percentage groups for initial seed level
x_labels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
x = np.arange(len(x_labels)) # The label locations for the 9 groups

# --- Mock Data (Simulated Results for 3 Metrics across 9 Percentage Groups) ---
# All metrics are now counts (Number of pairs count), so a single Y-axis is used.

# Metric 1: Initial Seed Pair Count (replaces percent_seed_data)
# Percentage=10.0, TP=2296, Total final mapping=2296,  Precision=1.0000, Recall=0.4380, Accuracy=0.4380====
evaluation_data = [
[10.0,524,2296,2296,1.0000,0.4380,0.4380],
[20.0,1048,3047,3047,1.0000,0.5813,0.5813],
[30.0,1572,3525,3525,1.0000,0.6725,0.6725],
[40.0,2096,3982,3982,1.0000,0.7596,0.7596],
[50.0,2621,4287,4287,1.0000,0.8178,0.8178],
[60.0,3145,4543,4543,1.0000,0.8667,0.8667],
[70.0,3669,4814,4814,1.0000,0.9184,0.9184],
[80.0,4193,5010,5010,1.0000,0.9557,0.9557],
[90.0,4717,5145,5145,1.0000,0.9815,0.9815],
]

# initial_seed_count = np.array([524, 1048, 1572, 2096, 2621, 3000, 3500, 4000, 4500])
# # Metric 2: Final Mapped Pair Count (replaces final_count_data)
# final_count = np.array([2296, 800, 2300, 3000, 3500, 4000, 4400, 4600, 4800])
# # Metric 3: True Positive Pair Count (replaces true_positive_data)
# true_positive_count = np.array([2296, 1400, 2100, 2800, 3300, 3700, 4000, 4200, 4300])

initial_seed_count = []
final_count = []
true_positive_count = []

for data in evaluation_data:
  initial_seed_count.append(data[1])
  final_count.append(data[3])
  true_positive_count.append(data[2])

initial_seed_count = np.array(initial_seed_count)
final_count = np.array(final_count)
true_positive_count = np.array(true_positive_count)

# --- 2. Setup Plotting Environment ---
# Using a wider figure size to accommodate 27 bars clearly
fig, ax1 = plt.subplots(figsize=(16, 7))

# Parameters for bar positioning (3 bars per group)
bar_width = 0.25
# Calculate positions for the three bars centered around the tick mark 'x'
r1 = x - bar_width     # Position for the first bar (Initial Seed)
r2 = x                 # Position for the second bar (Final Count)
r3 = x + bar_width     # Position for the third bar (True Positive)

# --- 3. Plotting: Single Y-axis for all Counts ---
# color1 = 'tab:blue'  # Initial Seed
# color2 = 'tab:red'   # Final Count
# color3 = 'tab:green' # True Positive

color1 = 'teal'      # Initial Seed
color2 = 'darkorange'# Final Count
color3 = 'olivedrab' # True Positive

# Plot the three sets of bars
ax1.bar(r1, initial_seed_count, color=color1, width=bar_width, edgecolor='grey', label='Initial Seed Pair Count')
ax1.bar(r2, final_count, color=color2, width=bar_width, edgecolor='grey', label='Final Mapped Pair Count')
ax1.bar(r3, true_positive_count, color=color3, width=bar_width, edgecolor='grey', label='True Positive Pair Count')


# --- 4. Customize Axes and Labels ---
plt.title('Performance Metrics by Initial Seed Percentage', fontsize=24)

# Y-Axis Label: Updated to reflect the new Count data
ax1.set_ylabel('Number of pairs count', fontsize=24)

# X-Axis Label: Updated
ax1.set_xlabel('Initial Seed Mapping Percentage', fontsize=24)

# Set the X-axis tick labels to the percentage groups, centered on 'x'
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, rotation=0, ha="center")

# Determine the maximum value for count data to set the Y axis limit
max_count = max(np.max(initial_seed_count), np.max(final_count), np.max(true_positive_count))
# Set the count axis limit slightly above the max data value for clarity
ax1.set_ylim(0, max_count * 1.1)

ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)

# --- 5. Final Touches ---
ax1.legend(loc='upper left', fontsize=24)

# Add grid lines for readability
ax1.grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()