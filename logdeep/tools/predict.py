#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
from collections import Counter, defaultdict
sys.path.append('../../')

import pickle
import numpy as np
import torch
import pandas as pd
# import dataframe as df
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from logdeep.dataset.log import log_dataset, log_dataset_2
from logdeep.dataset.sample import session_window, sliding_window



def generate(output_dir, name):
    print("Loading", output_dir + name)
    with open(output_dir + name, 'r') as f:
        data_iter = f.readlines()
        num_lines = len(data_iter)
        print("Number of lines:", num_lines)

    return data_iter, len(data_iter)

def df_to_file(df, file_name):
    with open(file_name, 'w') as f:
        for _, row in df.items():
            f.write(' '.join([str(ele) for ele in eval(row)]))
            f.write('\n')


def generate_2(output_dir, name):
    print("Loading", output_dir + name)
    seq = pd.read_csv(output_dir + name)
    seqline = seq["EventSequence"]
    df_to_file(seqline, output_dir + "predfile")
    with open(output_dir + name, 'r') as f:
        data_iter = f.readlines()
        num_lines = len(data_iter)
        print("Number of lines:", num_lines)

    return data_iter, len(data_iter)


class Predicter():
    def __init__(self, model, options):
        print("in __init__for class Predicter")
      
        self.output_dir = options['output_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.parameters = options['parameters']
        self.batch_size = options['batch_size']
        self.num_classes = options['num_classes']
        self.threshold = options["threshold"]
        self.gaussian_mean = options["gaussian_mean"]
        self.gaussian_std = options["gaussian_std"]
        self.save_dir = options['save_dir']
        self.is_logkey = options["is_logkey"]
        self.is_time = options["is_time"]
        self.vocab_path = options["vocab_path"]
        self.min_len = options["min_len"]
        self.test_ratio = options["test_ratio"]

        print("self.sequentials", self.sequentials)


    def detect_logkey_anomaly(self, output, label):
        num_anomaly = 0
        for i in range(len(label)):
            predicted = torch.argsort(output[i])[-self.num_candidates:].clone().detach().cpu()
            if label[i] not in predicted:
                num_anomaly += 1
        return num_anomaly
    
    def detect_logkey_anomaly_2(self, output, label):
        num_anomaly = 0
        anomalies = []
        for i in range(len(label)):
            predicted = torch.argsort(output[i])[-self.num_candidates:].clone().detach().cpu()
            print("in detect_logkey_anomaly_2, predicted: ", predicted)
            print("in detect_logkey_anomaly_2, label: ", label[i])
            if label[i] not in predicted:
                anomalies.append(i)
                num_anomaly += 1
        return anomalies, num_anomaly

    def compute_anomaly(self, results, threshold=0):
        total_errors = 0
        for seq_res in results:
            if isinstance(threshold, float):
                threshold = seq_res["predicted_logkey"] * threshold

            error = (self.is_logkey and seq_res["logkey_anomaly"] > threshold) or \
                    (self.is_time and seq_res["params_anomaly"] > threshold)
            total_errors += int(error)

        return total_errors

    def find_best_threshold(self, test_normal_results, test_abnormal_results, threshold_range):
        test_abnormal_length = len(test_abnormal_results)
        test_normal_length = len(test_normal_results)
        res = [0, 0, 0, 0, 0, 0, 0, 0]  # th,tp, tn, fp, fn,  p, r, f1
        for th in threshold_range:
            FP = self.compute_anomaly(test_normal_results, th)
            TP = self.compute_anomaly(test_abnormal_results, th)
            if TP == 0:
                continue

            # Compute precision, recall and F1-measure
            TN = test_normal_length - FP
            FN = test_abnormal_length - TP
            P = 100 * TP / (TP + FP)
            R = 100 * TP / (TP + FN)
            F1 = 2 * P * R / (P + R)
            if F1 > res[-1]:
                res = [th, TP, TN, FP, FN, P, R, F1]
        return res

    def unsupervised_helper(self, model, data_iter, vocab, data_type, scale=None, min_len=0):

        test_results = []
        normal_errors = []

        num_test = len(data_iter)
        rand_index = torch.randperm(num_test)
        rand_index = rand_index[:int(num_test * self.test_ratio)]

        with torch.no_grad():
            for idx, line in tqdm(enumerate(data_iter)):
                if idx not in rand_index:
                    continue

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

                if scale is not None:
                    tim = np.array(tim).reshape(-1,1)
                    tim = scale.transform(tim).reshape(-1).tolist()

                logkeys, times = [logkey.tolist()], [tim.tolist()] # add next axis

                logs, labels = sliding_window((logkeys, times), vocab, window_size=self.window_size, is_train=False)
                dataset = log_dataset(logs=logs,
                                        labels=labels,
                                        seq=self.sequentials,
                                        quan=self.quantitatives,
                                        sem=self.semantics,
                                        param=self.parameters)
                data_loader = DataLoader(dataset,
                                               batch_size=min(len(dataset), 128),
                                               shuffle=True,
                                               pin_memory=True)
                # batch_size = len(dataset)
                num_logkey_anomaly = 0
                num_predicted_logkey = 0
                for _, (log, label) in enumerate(data_loader):
                    features = []
                    for value in log.values():
                        features.append(value.clone().detach().to(self.device))

                    output = model(features=features, device=self.device)

                    num_predicted_logkey += len(label)

                    num_logkey_anomaly += self.detect_logkey_anomaly(output, label)

                # result for line at idx
                result = {"logkey_anomaly":num_logkey_anomaly,
                          "predicted_logkey": num_predicted_logkey
                          }
                test_results.append(result)
                if idx < 10 or idx % 1000 == 0:
                    print(data_type, result)

            return test_results, normal_errors

    def unsupervised_helper_2(self, model, data_iter, vocab, data_type, scale=None, min_len=0):

        test_results = []
        normal_errors = []
        retAnomalies = []

        # num_test = len(data_iter)
        # rand_index = torch.randperm(num_test)
        # rand_index = rand_index[:int(num_test * self.test_ratio)]

        with torch.no_grad():
            for idx, line in tqdm(enumerate(data_iter)):
                # if idx not in rand_index:
                #     continue
                # print("in unsupervised_helper_2....",line)
                line = line.strip().split(",",1)
                # line = [ln.split(",") for ln in line.split()]
                print("line: ",line)
                # print(type(line))
                # print(len(line))
                
                # if len(line[1]) < min_len:
                #     continue
                # print("After first if....",line[1])
                # if line[1] == "EventSequence":
                #     print("in continue if..")
                #     continue
                
                #add the compute instances required later
                # print("post if......")
                # print(type(line[1]))
                # tempVar = (line[1].strip('"[]'))
                # print(tempVar)
                # int_list = list(map(int, line[1].strip('"[]').split(",")))
                line = np.array(line)
                # line = np.array(line[1])
                print("line: post np.array...",line)

                print("line.shape...",line.shape)

                # print("line.shape[0]...",line.shape[0])
                # # if time duration exists in data
                # if line.shape[1] == 2:
                #     tim = line[:, 1].astype(float)
                #     tim[0] = 0
                #     logkey = line[:, 0]
                # else:
                logkey = line.squeeze()
                if logkey.ndim == 0:  # If it's a scalar
                    logkey = np.array([logkey])
                # if time duration doesn't exist, then create a zero array for time
                tim = np.zeros(logkey.shape)


                if scale is not None:
                    tim = np.array(tim).reshape(-1,1)
                    tim = scale.transform(tim).reshape(-1).tolist()
                
                print("[logkey.tolist()]: ",[logkey.tolist()])
                print("[tim.tolist()]: ",[tim.tolist()])

                logkeys, times = [str(x) for x in logkey.tolist()], [tim.tolist()] # add next axis
                print("in unsupervised_helper_2...logkeys type, after conversion: ",type(logkeys))
                print("in unsupervised_helper_2...logkeys type: ",type(logkeys))
                logs, labels = sliding_window((logkeys, times), vocab, window_size=self.window_size, is_train=False)
                
                dataset = log_dataset(logs=logs,
                                        labels=labels,
                                        seq=self.sequentials,
                                        quan=self.quantitatives,
                                        sem=self.semantics,
                                        param=self.parameters
                                        # metadata=None
                                        )
                
                data_loader = DataLoader(dataset,
                                               batch_size=min(len(dataset), 128),
                                               shuffle=False,
                                               pin_memory=True)
                # batch_size = len(dataset)
                num_logkey_anomaly = 0
                num_predicted_logkey = 0
                
                for _, (log, label) in enumerate(data_loader):
                    features = []
                    for value in log.values():
                        features.append(value.clone().detach().to(self.device))

                    print("in unsupervised_helper_2....features....",features)
                    output = model(features=features, device=self.device)

                    print("in unsupervised_helper_2....output...",output)

                    num_predicted_logkey += len(label)

                    # anomaly, num_logkey_anomaly_temp = self.detect_logkey_anomaly(output, label)
                    anomaly, numAnomalies= self.detect_logkey_anomaly_2(output, label)
                    print("anomaly is....",anomaly,"type:",type(anomaly),"num anomaly",numAnomalies)
                    print("anomaly[0] is....",anomaly[0],"type:",type(anomaly[0]))
                    # num_logkey_anomaly += num_logkey_anomaly_temp
                    if anomaly[0]!=0:
                        retAnomalies.append(idx)

                # # result for line at idx
                # result = {"logkey_anomaly":num_logkey_anomaly,
                #           "predicted_logkey": num_predicted_logkey
                #           }
                # test_results.append(result)
                # if idx < 10 or idx % 1000 == 0:
                #     print(data_type, result)

            return retAnomalies, normal_errors


    def predict_unsupervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))

        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        test_normal_loader, _ = generate(self.output_dir, 'test_normal')
        test_abnormal_loader, _ = generate(self.output_dir, 'test_abnormal')

        scale = None
        if self.is_time:
            with open(self.save_dir + "scale.pkl", "rb") as f:
                scale = pickle.load(f)

        # Test the model
        start_time = time.time()
        test_normal_results, normal_errors = self.unsupervised_helper(model, test_normal_loader, vocab, 'test_normal', scale=scale, min_len=self.min_len)
        test_abnormal_results, abnormal_errors = self.unsupervised_helper(model, test_abnormal_loader, vocab, 'test_abnormal', scale=scale, min_len=self.min_len)

        print("Saving test normal results", self.save_dir + "test_normal_results")
        with open(self.save_dir + "test_normal_results", "wb") as f:
            pickle.dump(test_normal_results, f)

        print("Saving test abnormal results", self.save_dir + "test_abnormal_results")
        with open(self.save_dir + "test_abnormal_results", "wb") as f:
            pickle.dump(test_abnormal_results, f)

        TH, TP, TN, FP, FN, P, R, F1 = self.find_best_threshold(test_normal_results,
                                                                test_abnormal_results,
                                                                threshold_range=np.arange(10))
        
        print('Best threshold', TH)
        outputResultString = f'<br>Best threshold: {TH}'+"<br>"

        print("Confusion matrix")
        print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        outputResultString = outputResultString + 'Best threshold ' + "TP: {0}, TN: {1}, FP: {2}, FN: {3}".format(TP, TN, FP, FN) +"<br>"
        print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(P, R, F1))
        outputResultString = outputResultString + 'Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(P, R, F1)+"<br>"
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))
        outputResultString = outputResultString + 'elapsed_time: {}'.format(elapsed_time)
        return outputResultString

    def predict_unsupervised_2(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))

        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        # pred_loader, _ = generate(self.output_dir, 'pred_ds')
        pred_loader, _ = generate_2('./predOutput/', 'event_sequence.csv')
        # pred_loader, _ = generate('./predOutput/', 'predfile')
        print("in predict_unsupervised_2...pred_loader...",pred_loader)
        # test_abnormal_loader, _ = generate(self.output_dir, 'test_abnormal')
        
        scale = None
        if self.is_time:
            with open(self.save_dir + "scale.pkl", "rb") as f:
                scale = pickle.load(f)

        # Test the model
        start_time = time.time()
        pred_results, pred_errors = self.unsupervised_helper_2(model, pred_loader, vocab, 'pred_ds', scale=scale, min_len=self.min_len)
        # test_abnormal_results, abnormal_errors = self.product_unsupervised_helper(model, test_abnormal_loader, vocab, 'test_abnormal', scale=scale, min_len=self.min_len)

        # print("Saving test normal results", self.save_dir + "test_normal_results")
        # with open(self.save_dir + "test_normal_results", "wb") as f:
        #     pickle.dump(test_normal_results, f)

        # print("Saving test abnormal results", self.save_dir + "test_abnormal_results")
        # with open(self.save_dir + "test_abnormal_results", "wb") as f:
        #     pickle.dump(test_abnormal_results, f)

        # TH, TP, TN, FP, FN, P, R, F1 = self.find_best_threshold(test_normal_results,
        #                                                         test_abnormal_results,
        #                                                         threshold_range=np.arange(10))

        print("pred_results\n", pred_results)
        print("pred_errors\n", pred_errors)


        # self.compute_anomaly(pred_results, pred_errors)
        # print('Best threshold', TH)
        # outputResultString = f'<br>Best threshold: {TH}'+"<br>"

        # print("Confusion matrix")
        # print("TP: {}, TN: {}, FP: {}, FN: {}".format(TP, TN, FP, FN))
        # outputResultString = outputResultString + 'Best threshold ' + "TP: {0}, TN: {1}, FP: {2}, FN: {3}".format(TP, TN, FP, FN) +"<br>"
        # print('Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(P, R, F1))
        # outputResultString = outputResultString + 'Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(P, R, F1)+"<br>"
        # elapsed_time = time.time() - start_time
        # print('elapsed_time: {}'.format(elapsed_time))
        # outputResultString = outputResultString + 'elapsed_time: {}'.format(elapsed_time)
        outputResultString = pred_results
        return outputResultString
    
    def predict_supervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_logs, test_labels = session_window(self.output_dir, datatype='test')
        test_dataset = log_dataset(logs=test_logs,
                                   labels=test_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True)
        tbar = tqdm(self.test_loader, desc="\r")
        TP, FP, FN, TN = 0, 0, 0, 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().to(self.device))
            output = self.model(features=features, device=self.device)
            output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
            # predicted = torch.argmax(output, dim=1).cpu().numpy()
            predicted = (output < 0.2).astype(int)
            label = np.array([y.cpu() for y in label])
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
            .format(FP, FN, P, R, F1))
        
    