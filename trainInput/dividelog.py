import pandas as pd
import random

# Step 1: Load labeled data
label_df = pd.read_csv('anomaly_label_cleaned.csv')

# Step 2: Create a mapping of ComputeInstance to label
instance_label_map = label_df.set_index('ComputeInstance')['Label'].to_dict()

# Step 3: Index log lines with instance and label info
indexed_lines = {'normal': [], 'Anomaly': []}

# Step 4: Scan log file and categorize lines
with open('nova-sample.log', 'r') as log_file:
    for line_number, line in enumerate(log_file, start=1):
        for instance, label in instance_label_map.items():
            if instance in line:
                indexed_lines[label].append((line_number, line))
                break  # stop at first match

# Step 5: Shuffle and split each label set
training_lines, demo_lines = [], []

for label, lines in indexed_lines.items():
    random.shuffle(lines)
    split_idx = int(len(lines) * 0.6)
    training_lines.extend(lines[:split_idx])
    demo_lines.extend(lines[split_idx:])

# Step 6: Write training and demo files
with open('nova-sample-training.log', 'w') as train_file:
    for _, line in sorted(training_lines):
        train_file.write(line)

with open('nova-sample-demo.log', 'w') as demo_file:
    for _, line in sorted(demo_lines):
        demo_file.write(line)

print("✅ Log files split and saved:")
print(" - 'nova-sample-training.log' (60%)")
print(" - 'nova-sample-demo.log' (40%)")