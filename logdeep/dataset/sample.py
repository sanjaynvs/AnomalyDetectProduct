import json
from collections import Counter
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.stdout.reconfigure(encoding='utf-8')


def read_json(filename):
    with open(filename, 'r') as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n):
    """ Truncate or pad a list """
    r = l[:n]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def down_sample(logs, labels, sample_ratio):
    print('sampling...')
    total_num = len(labels)
    all_index = list(range(total_num))
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels


# https://stackoverflow.com/questions/15357422/python-determine-if-a-string-should-be-converted-into-int-or-float
def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a)
    except ValueError:
        return False
    else:
        return a == b


def split_features(data_path, train_ratio=1, scale=None, scale_path=None, min_len=0):
    with open(data_path, 'r') as f:
        data = f.readlines()

    sample_size = int(len(data) * train_ratio)
    data = data[:sample_size]
    logkeys = []
    times = []
    for line in data:
        line = [ln.split(",") for ln in line.split()]

        if len(line) < min_len:
            continue

        line = np.array(line)
        # if time duration exists in data
        if line.shape[1] == 2:
            tim = line[:, 1].astype(float)
            tim[0] = 0
            logkey = line[:, 0]
        else:
            logkey = line.squeeze()
            # if time duration doesn't exist, then create a zero array for time
            tim = np.zeros(logkey.shape)

        logkeys.append(logkey.tolist())
        times.append(tim.tolist())

    if scale is not None:
        total_times = np.concatenate(times, axis=0).reshape(-1,1)
        scale.fit(total_times)

        for i, tn in enumerate(times):
            tn = np.array(tn).reshape(-1,1)
            times[i] = scale.transform(tn).reshape(-1).tolist()

        with open(scale_path, 'wb') as f:
            pickle.dump(scale, f)
        print("Save scale {} at {}\n".format(scale, scale_path))

    return logkeys, times


def sliding_window(data_iter, vocab, window_size, is_train=True):
    print("In sliding window: print vocab...")
    print(vocab.stoi.get("10", vocab.unk_index))

    for token, index in vocab.stoi.items():
        print(f"{token} - {index}")

    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    #event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    result_logs['Parameters'] = []
    labels = []

    num_sessions = 0
    num_classes = len(vocab)

    # templine = zip(*data_iter)
    # print("start sliding window...templine",data_iter[0])

    tempParam = zip(*data_iter)
    for line, params in tempParam:
        print("In sliding window: for loop ... line...",line)
        print("In sliding window: for loop ... params...",params)
    # data_iter = zip(tempLine, tempParam)
    # print("In sliding window: tempParam...",tempParam)
    for line, params in zip(*data_iter):
        print(f"Original line: {line}")
        print(f"Line elements: {list(line)}")
        print(f"Line elements str: {list(str(line))}")
        print(f"Element types: {[type(ln) for ln in line]}")
        
        # Process each element and show the mapping
        result = []
        for i, ln in enumerate(line):
            mapped_value = vocab.stoi.get(ln, vocab.unk_index)
            print(f"Element {i}: {ln} -> {mapped_value}")
            result.append(mapped_value)
        
        print(f"Final result: {result}")
        break  # Just check first iteration
    
    for line, params in zip(*data_iter):
        if num_sessions % 1000 == 0:
            print("processed %s lines"%num_sessions, end='\r')
        num_sessions += 1
        print("In sliding window: line...",line)
  
        # line = str(line)
        # line = [line]
        line = [vocab.stoi.get(ln, vocab.unk_index) for ln in line]
        # line = vocab.stoi.get(str(line), vocab.unk_index)
        print("in slider window: line before padding...",line)
        print("in slider window: len(line)...",len(line))
        print("in slider window: window_size...",window_size)

        session_len = max(len(line), window_size) + 1# predict the next one
        padding_size = session_len - len(line)
        params = params + [0] * padding_size
        line = line + [vocab.pad_index] * padding_size
        print("in slider window: line after padding...",line)

        for i in range(session_len - window_size):
            Parameter_pattern = params[i:i + window_size]
            Sequential_pattern = line[i:i + window_size]
            Semantic_pattern = []

            Quantitative_pattern = [0] * num_classes
            log_counter = Counter(Sequential_pattern)

            for key in log_counter:
                Quantitative_pattern[key] = log_counter[key]

            # Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
            # Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
            Sequential_pattern = np.array(Sequential_pattern)
            Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]

            result_logs['Sequentials'].append(Sequential_pattern)
            result_logs['Quantitatives'].append(Quantitative_pattern)
            result_logs['Semantics'].append(Semantic_pattern)
            result_logs["Parameters"].append(Parameter_pattern)
            
            print("in slider window: len(line)...",len(line))
            print("in slider window: i....",i)
            print("in slider window: i + window_size",i + window_size)
            print("in slider window: line[i + window_size]",line[i + window_size])
            labels.append(line[i + window_size])

    if is_train:
        print('number of sessions {}'.format(num_sessions))
        print('number of seqs {}'.format(len(result_logs['Sequentials'])))

    return result_logs, labels



def session_window(data_dir, datatype, sample_ratio=1):
    event2semantic_vec = read_json(data_dir + 'deeplog/event2semantic_vec.json')
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []

    if datatype == 'train':
        data_dir += 'deeplog/robust_log_train.csv'
    elif datatype == 'val':
        data_dir += 'deeplog/robust_log_valid.csv'
    elif datatype == 'test':
        data_dir += 'deeplog/robust_log_test.csv'

    train_df = pd.read_csv(data_dir)
    for i in tqdm(range(len(train_df))):
        ori_seq = [
            int(eventid) for eventid in train_df["Sequence"][i].split(' ')
        ]
        Sequential_pattern = trp(ori_seq, 50)
        Semantic_pattern = []
        for event in Sequential_pattern:
            if event == 0:
                Semantic_pattern.append([-1] * 300)
            else:
                Semantic_pattern.append(event2semantic_vec[str(event - 1)])
        Quantitative_pattern = [0] * 29
        log_counter = Counter(Sequential_pattern)

        for key in log_counter:
            Quantitative_pattern[key] = log_counter[key]

        Sequential_pattern = np.array(Sequential_pattern) # [:, np.newaxis]
        Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
        result_logs['Sequentials'].append(Sequential_pattern)
        result_logs['Quantitatives'].append(Quantitative_pattern)
        result_logs['Semantics'].append(Semantic_pattern)
        labels.append(int(train_df["label"][i]))

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    # result_logs, labels = up_sample(result_logs, labels)

    print('Number of sessions({}): {}'.format(data_dir,
                                              len(result_logs['Semantics'])))
    return result_logs, labels