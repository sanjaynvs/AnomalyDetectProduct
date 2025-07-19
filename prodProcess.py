import sys
sys.path.append('../')

import os
import re
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from logparser import DrainPred
from deeplog import Predicter, Model, options

pred_input_dir  = './predInput/'
pred_input_dataset_dir  = pred_input_dir+'datasets/'
input_log_file   = pred_input_dataset_dir+"nova-sample.log"

pred_input_event_templates_dir = pred_input_dir+'templates/'
event_templates_file = pred_input_event_templates_dir + "event_templates.csv"

pred_output_dir = './predOutput/'  # The output directory of parsing results
pred_sequence_file = pred_output_dir + "event_sequence.csv"
pred_sequences = pred_output_dir+'sequences'

model_file = pred_output_dir + '/deeplog/log_model.pkl'
vocab_file = './models/vocab.pkl'

class PredTask():
    def __init__(self, input_log_file, log_format, log_name):
        self.log_name = log_name
        self.log_format = log_format
        self.pred_input_dir  = './predInput/'
        self.pred_input_dataset_dir  = self.pred_input_dir+'datasets/'
        self.input_log_file = input_log_file

        self.pred_input_event_templates_dir = self.pred_input_dir+'templates/'
        self.event_templates_file = self.pred_input_event_templates_dir + "trained_log_templates.csv"

        self.pred_output_dir = './predOutput/'  # The output directory of parsing results
        self.pred_sequence_file = self.pred_output_dir + "event_sequence.csv"
        self.pred_sequences = pred_output_dir+'sequences'

    def parse_log(self):

        regex = [
        r'^[a-z0-9.-]+,\d+,[a-fA-F0-9]+,(?:- ){4}-\]$', # represents - cp-1.slowvm1.tcloud-pg0.utah.cloudlab.us,34,462cb051,- - - - -] 
        r'[a-z0-9.-]+(?:\.[a-z0-9-]+)+',
        r"[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}",  #compute instance id e.g. a208479c-c0e3-4730-a5a0-d75e8afd0252
        r"(/[-\w]+)+",  # file path
        r'\d+\.\d+\.\d+\.\d+',  # IP
        r"(/[-\w]+)+ HTTP/1.1",  # file path with HTTP protocol
        r'\d+(\.\d+)?',  # Numbers with optional decimal point
        ]

        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.5  # Similarity threshold
        depth = 10  # Depth of all leaf nodes

        parser = DrainPred.LogParser(self.log_format, depth=depth, st=st, rex=regex, keep_para=False, log_templates_file=self.event_templates_file, outdir=self.pred_output_dir, sequence_file=self.pred_sequence_file, log_name=None)
        retsequences = parser.parse(self.input_log_file,self.log_name)
        return retsequences

    def predict(self):
        pass 


if __name__ == "__main__":
    print("in main...")

    log_format = '<Level> <Component> <ADDR> <Content>' 
    pred_input_dir  = './predInput/'
    pred_input_dataset_dir  = pred_input_dir+'datasets/'
    log_file = "nova-sample.log"

    pd = PredTask(pred_input_dataset_dir + log_file, log_format,log_file)
    parsed_sequences = pd.parse_log()
    predicter = Predicter(Model, options)
    print(len(parsed_sequences))
    anomalyCS = predicter.predict_unsupervised_3(parsed_sequences)
    print(anomalyCS)

    import json
    import pandas as pd

    # Load input files
    df_sequence = pd.read_csv('./predOutput/event_sequence.csv')
    df_templates = pd.read_csv('./predInput/templates/trained_log_templates.csv')

    # Your anomaly indexes
    # anomalyCS = [4, 6, 7, ..., 1920]  # Replace with full list

    # Load and clean EventSequence column
    df_sequence['EventSequence'] = df_sequence['EventSequence'].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )

    # Map templateIndex to actual template string
    template_map = df_templates.set_index('templateIndex')['EventTemplate'].to_dict()

    # Filter rows based on anomaly indices
    df_anomalies = df_sequence.iloc[anomalyCS].copy()

    # Convert EventSequence numeric values to template strings
    def decode_sequence(seq):
        return [template_map.get(int(item), f"<unknown:{item}>") if pd.notna(item) else None for item in seq]

    df_anomalies['DecodedSequence'] = df_anomalies['EventSequence'].apply(decode_sequence)

    # Build final JSON output
    result = {
        row['ComputeInstance']: row['DecodedSequence']
        for _, row in df_anomalies.iterrows()
    }

    # Print or write to file
    # print(json.dumps(result, indent=2))

    with open('decoded_anomalies.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # Optional: save to file
    # with open('decoded_anomaly_sequences.json', 'w') as f:
    #     json.dump(result, f, indent=2)

