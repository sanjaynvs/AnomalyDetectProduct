import sys
sys.path.append('../')

import os
import re
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from logparser import DrainTraining
import deeplog as Deeplog

class TrainTask:
    def __init__(self, input_dir, output_dir, log_file, anomaly_label_file, log_format, log_structured_file=None, log_templates_file=None, log_sequence_file="event_sequence.csv"):
        
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.log_file = log_file
        self.anomaly_label_file = anomaly_label_file
       
        self.log_format = log_format

        self.log_structured_file = log_structured_file
        self.log_templates_file = log_templates_file
        self.log_sequence_file = log_sequence_file
        
    def parser(self):
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

        parser = DrainTraining.LogParser(log_format=self.log_format, indir=self.input_dir, outdir=self.output_dir, logName=self.log_file,depth=depth, st=st, rex=regex, keep_para=False)
        return parser.parse()

    def generate_train_test(self, n=None, ratio=0.7):
        computeInst_label_dict = {}
        computeInst_label_file = os.path.join(self.input_dir, self.anomaly_label_file)
        computeInst_df = pd.read_csv(computeInst_label_file)
        for idx, row in tqdm(computeInst_df.iterrows()):
            computeInst_label_dict[row["ComputeInstance"]] = 1 if row["Label"] == "Anomaly" else 0

        compute_instance_file = os.path.join(self.output_dir, "event_sequence.csv")
        seq = pd.read_csv(compute_instance_file)
        seq["Label"] = seq["ComputeInstance"].apply(lambda x: computeInst_label_dict.get(x)) #add label to the sequence of each ComputeInstance
        normal_seq = seq[seq["Label"] == 0]["EventSequence"]
        normal_seq = normal_seq.sample(frac=1, random_state=20) # shuffle normal data

        abnormal_seq = seq[seq["Label"] == 1]["EventSequence"]
        normal_len, abnormal_len = len(normal_seq), len(abnormal_seq)
        train_len = n if n else int(normal_len * ratio)

        train = normal_seq.iloc[:train_len]
        test_normal = normal_seq.iloc[train_len:]
        test_abnormal = abnormal_seq

        self.df_to_file(train, output_dir + "train")
        self.df_to_file(test_normal, output_dir + "test_normal")
        self.df_to_file(test_abnormal, output_dir + "test_abnormal")
        print("generate train test data done")
        return str("normal size {0}, abnormal size {1}, training size {2}".format(normal_len, abnormal_len, train_len))

    def df_to_file(self, df, file_name):
        with open(file_name, 'w') as f:
            for _, row in df.items():
                f.write(' '.join([str(ele) for ele in eval(row)]))
                f.write('\n')

if __name__ == "__main__":
    log_format = '<Level> <Component> <ADDR> <Content>' 
    train_input_dir  = './trainInput/'
    train_log_file = "nova-sample.log"
    anomaly_label_file = "anomaly_label.csv"
    # anomaly_label_file = "anomaly_label_cleaned.csv"
    output_dir = './trainOutput/'
    
    trainTask = TrainTask(train_input_dir, output_dir, train_log_file, anomaly_label_file, log_format)
    parsereturn = trainTask.parser()
    print(parsereturn)
    trainTask.generate_train_test( )
    vocab_size = Deeplog.vocab()
    print("vocab_size: ", vocab_size)
    Deeplog.set_vocab_size(vocab_size)
    Deeplog.train()
    