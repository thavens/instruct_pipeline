import matplotlib.pyplot as plt
import re
from collections import defaultdict

# File path
log_file = "test_all.log"

# Data storage
data = defaultdict(lambda: {'checkpoints': [], 'ratios': []})

# Regular expression to match checkpoint, probability, and ratio
pattern = re.compile(r"checkpoint: (\d+) and prob: ([0-9\.]+).*?Ratio: ([0-9\.]+)", re.DOTALL)

# Read and parse the file
with open(log_file, "r") as file:
    content = file.read()
    matches = pattern.findall(content)
    
    for checkpoint, prob, ratio in matches:
        checkpoint = int(checkpoint)
        prob = float(prob)
        ratio = float(ratio)
        data[prob]['checkpoints'].append(checkpoint)
        data[prob]['ratios'].append(ratio)

# Plot the data
plt.figure(figsize=(10, 6))
for prob, values in sorted(data.items()):
    checkpoints, ratios = zip(*sorted(zip(values['checkpoints'], values['ratios'])))
    plt.plot(checkpoints, ratios, marker='o', linestyle='-', label=f'Prob: {prob}')

# Labels and legend
plt.xlabel("Checkpoint")
plt.ylabel("Ratio")
plt.title("Checkpoint vs. Ratio of Exclamations to Periods where prob is prob of exclamation given punctuation")
plt.legend()
plt.grid()

# Save the plot as a PNG file
plt.savefig("checkpoint_ratio_plot.png")

# Show the plot
plt.show()