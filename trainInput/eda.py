import pandas as pd

# Load anomaly labels
df = pd.read_csv('anomaly_line_report.csv')
zero_entry_df = df[df['EntryCount'] == 0]
print(zero_entry_df['Label'].value_counts())
