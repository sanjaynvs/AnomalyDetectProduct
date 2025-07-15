import pandas as pd

# Step 1: Read the report file
report_df = pd.read_csv('anomaly_line_report.csv')

# Step 2: Filter instances with EntryCount == 0
zero_entry_instances = report_df[report_df['EntryCount'] == 0]['ComputeInstance'].unique()

# Step 3: Read original anomaly label file
label_df = pd.read_csv('anomaly_label.csv')

# Step 4: Remove rows with ComputeInstance in zero_entry_instances
cleaned_df = label_df[~label_df['ComputeInstance'].isin(zero_entry_instances)]

# Save the cleaned dataset
cleaned_df.to_csv('anomaly_label_cleaned.csv', index=False)

print("✅ Cleaned file saved as 'anomaly_label_cleaned.csv'")