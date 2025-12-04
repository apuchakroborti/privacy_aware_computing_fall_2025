import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define Data ---

# X-axis data: Initial heuristic based mapping (Number of nodes)
# x_values = np.array([100, 500, 1100, 1700, 2500, 3500, 4500, 4900])
# x_values = np.array([100, 500, 1100, 1700, 2500, 3500])
x_values = np.array([500, 1100, 2500, 3500])

# Theta values to be analyzed
theta_values = [0.5, 1.0, 1.5, 2.0, 2.5]

# Mock Y-axis data (Final Mapped Data Count) for each theta value.
# The data simulates a scenario where a higher theta generally results in a higher final count.
# y_data = {
#     # Theta: Mock Final Mapped Data Counts
#     0.5: np.array([20, 100, 300, 800, 1200, 1400]),
#     1.0: np.array([30, 150, 450, 1200, 1800, 2100]),
#     1.5: np.array([36, 180, 540, 1440, 2160, 2520]),
#     2.0: np.array([40, 200, 600, 1600, 2400, 2800]),
#     2.5: np.array([42, 210, 630, 1680, 2520, 2940]),
# }

y_data = {
#     # Theta: Mock Final Mapped Data Counts
    0.5: [],
    1.0: [],
    1.5: [],
    2.0: [],
    2.5: [],
}
evaluation_data = [   
#         [100,216,216,1.0000,0.0412,0.0412],
# [100,214,214,1.0000,0.0408,0.0408],
# [100,193,193,1.0000,0.0368,0.0368],
# [100,194,194,1.0000,0.0370,0.0370],
# [100,145,145,1.0000,0.0277,0.0277],
[500,1618,1618,1.0000,0.3087,0.3087],
[500,1616,1616,1.0000,0.3083,0.3083],
[500,1531,1531,1.0000,0.2921,0.2921],
[500,1405,1405,1.0000,0.2680,0.2680],
[500,1373,1373,1.0000,0.2619,0.2619],
[1100,2338,2338,1.0000,0.4460,0.4460],
[1100,2334,2334,1.0000,0.4452,0.4452],
[1100,2321,2321,1.0000,0.4428,0.4428],
[1100,2295,2295,1.0000,0.4378,0.4378],
[1100,2288,2288,1.0000,0.4365,0.4365],
# [1700,2849,2849,1.0000,0.5435,0.5435],
# [1700,2849,2849,1.0000,0.5435,0.5435],
# [1700,2847,2847,1.0000,0.5431,0.5431],
# [1700,2823,2823,1.0000,0.5385,0.5385],
# [1700,2821,2821,1.0000,0.5382,0.5382],
[2500,3376,3376,1.0000,0.6440,0.6440],
[2500,3376,3376,1.0000,0.6440,0.6440],
[2500,3376,3376,1.0000,0.6440,0.6440],
[2500,3376,3376,1.0000,0.6440,0.6440],
[2500,3376,3376,1.0000,0.6440,0.6440],
[3500,3980,3982,0.9995,0.7593,0.7593],
[3500,3980,3982,0.9995,0.7593,0.7593],
[3500,3980,3982,0.9995,0.7593,0.7593],
[3500,3980,3982,0.9995,0.7593,0.7593],
[3500,3980,3982,0.9995,0.7593,0.7593],
# [4500,4700,4702,0.9996,0.8966,0.8966],
# [4500,4700,4702,0.9996,0.8966,0.8966],
# [4500,4700,4702,0.9996,0.8966,0.8966],
# [4500,4700,4702,0.9996,0.8966,0.8966],
# [4500,4700,4702,0.9996,0.8966,0.8966],
# [4900,4929,4931,0.9996,0.9403,0.9403],
# [4900,4929,4931,0.9996,0.9403,0.9403],
# [4900,4929,4931,0.9996,0.9403,0.9403],
# [4900,4929,4931,0.9996,0.9403,0.9403],
# [4900,4929,4931,0.9996,0.9403,0.9403],

]
theta_based_final_count = [[],[], [], [], []]
index = 0
for index in range(0, len(evaluation_data)):
  theta_based_final_count[index%5].append(evaluation_data[index][2])

y_data = {
#     # Theta: Mock Final Mapped Data Counts
    0.5: np.array(theta_based_final_count[0]),
    1.0: np.array(theta_based_final_count[1]),
    1.5: np.array(theta_based_final_count[2]),
    2.0: np.array(theta_based_final_count[3]),
    2.5: np.array(theta_based_final_count[4]),
}
# rows = []
# rows.append((100,216,216,1.0000,0.0412,0.0412))
# rows.append((500,1618,1618,1.0000,0.3087,0.3087))
# rows.append((1100,2338,2338,1.0000,0.4460,0.4460))
# rows.append((1700,2849,2849,1.0000,0.5435,0.5435))
# rows.append((2500,3376,3376,1.0000,0.6440,0.6440))
# rows.append((3500,3980,3982,0.9995,0.7593,0.7593))
# rows.append((4500,4700,4702,0.9996,0.8966,0.8966))
# rows.append((4900,4929,4931,0.9996,0.9403,0.9403))
# for row in rows:
#   precision.append(row[3]*100)
#   recall.append(row[4]*100)
#   accuracy.append(row[5]*100)

# precision = np.array(precision)
# recall = np.array(recall)
# accuracy = np.array(accuracy)

# Define colors and markers for better visual differentiation
line_styles = [
    {'color': 'tab:blue', 'marker': 'o', 'linestyle': '-'},
    {'color': 'tab:orange', 'marker': 's', 'linestyle': '--'},
    {'color': 'tab:green', 'marker': '^', 'linestyle': '-.'},
    {'color': 'tab:red', 'marker': 'D', 'linestyle': ':'},
    {'color': 'tab:purple', 'marker': 'p', 'linestyle': '-'}
]


# --- 2. Setup Plotting Environment ---
plt.figure(figsize=(10, 6))

# --- 3. Plot the Lines ---
for i, theta in enumerate(theta_values):
    style = line_styles[i]
    plt.plot(
        x_values,
        y_data[theta],
        marker=style['marker'],
        linestyle=style['linestyle'],
        color=style['color'],
        label=f'Theta ($\u03B8$) = {theta}' # Use LaTeX syntax for theta
    )

# --- 4. Customize Axes and Labels ---
plt.title('Impact of Theta ($\u03B8$) on Final Mapped Data', fontsize=24)
plt.xlabel('Initial Heuristic-Based Mapping Count', fontsize=24)
plt.ylabel('Final Mapped Data Count', fontsize=24)

# X-Axis Customization: Use the defined x_values as the tick positions and labels
plt.xticks(x_values, [str(val) for val in x_values], rotation=45, ha='right')
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)

# Add a grid and legend
plt.legend(title='Theta Values', loc='upper left', fontsize=16)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# Show the plot
plt.show()