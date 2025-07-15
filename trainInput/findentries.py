import pandas as pd

# Load anomaly labels
df = pd.read_csv('anomaly_label.csv')
instance_labels = df[['ComputeInstance', 'Label']].dropna()

# Dictionary to collect line numbers and counts for each instance
results = {
    row['ComputeInstance']: {
        'Label': row['Label'],
        'LineNumbers': [],
        'EntryCount': 0
    }
    for _, row in instance_labels.iterrows()
}

# Scan nova-sample.log
with open('nova-sample.log', 'r') as log_file:
    for line_number, line in enumerate(log_file, start=1):
        print(f"Scanning line {line_number}")
        for instance in results:
            if instance in line:
                results[instance]['LineNumbers'].append(line_number)
                results[instance]['EntryCount'] += 1

# Prepare rows for export
output_rows = []
for instance, data in results.items():
    output_rows.append({
        'ComputeInstance': instance,
        'Label': data['Label'],
        'EntryCount': data['EntryCount'],
        'LineNumbers': ', '.join(map(str, data['LineNumbers'])) if data['LineNumbers'] else ''
    })

# Save to CSV
report_df = pd.DataFrame(output_rows)
report_df.to_csv('anomaly_line_report.csv', index=False)

print("✅ Full report with entry counts saved as 'anomaly_line_report.csv'")