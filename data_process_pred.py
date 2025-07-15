import sys
sys.path.append('../')

import os
import re
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from logparser import Drain

# get [log key, delta time] as input for deeplog
train_input_dir  = './predInput/datasets'
train_output_dir = './trainOutput/'  # The output directory of parsing results
log_file   = "nova-sample.log"
log_structured_file = train_output_dir + log_file + "_structured.csv"
# log_structured_file = output_dir + "common2_structured.csv"
train_log_templates_file = train_output_dir + log_file + "_templates.csv"
log_sequence_file = train_output_dir + "deeplog_sequence.csv"
pred_input_dir = "./predInput/"
pred_output_dir = "./predOutput/"
pred_log_templates_file= pred_input_dir + log_file + "_templates.csv"

def prePredict_process():
    log_temp = pd.read_csv(train_log_templates_file)
    log_temp.sort_values(by = ["Occurrences"], ascending=False, inplace=True)
    log_temp["index"] = range(1, len(log_temp)+1)
    if not os.path.exists(pred_input_dir):
        os.makedirs(pred_input_dir)
    log_temp.to_csv(pred_log_templates_file, index=None)



def parser(input_dir, output_dir, log_file, log_format):

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

    parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=False,pred_log_templates_file=pred_log_templates_file,sequence_file=output_dir+'event_sequence.csv')
    parser.parse_pred(log_file)




if __name__ == "__main__":
    #preprocess and copy the created template file
    prePredict_process()
    # 1. OS log
    # Below two lines are for parsing the log file. Will have to be refactored later 
    # log_format = '<Logrecord> <Date> <Time> <Pid> <Level> <Component> <ADDR> <Content>'  # OS log format
    log_format = '<Level> <Component> <ADDR> <Content>'  # OS log format
    parser(pred_input_dir, pred_output_dir, log_file, log_format)
