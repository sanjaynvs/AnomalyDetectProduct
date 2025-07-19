"""
Description : This file implements the Drain algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""

import re
import os
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime


class Logcluster:
    def __init__(self, logTemplate='', logIDL=None):
        self.logTemplate = logTemplate
        if logIDL is None:
            logIDL = []
        self.logIDL = logIDL


class Node:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        if childD is None:
            childD = dict()
        self.childD = childD
        self.depth = depth
      
        self.digitOrtoken = digitOrtoken


class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', logName=None, depth=4, st=0.4,
                 maxChild=100, rex=[], keep_para=True):
        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            depth : depth of all leaf nodes
            st : similarity threshold
            maxChild : max number of children of an internal node
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.maxChild = maxChild
        self.logName = logName
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para
        # self.pred_log_templates_file = pred_log_templates_file
        # self.sequence_file = sequence_file

    def hasNumbers(self, s):
        # print(" in hasNumbers: ", s)
        # print([char for char in s if char.isdigit()])
        return any(char.isdigit() for char in s)

    def treeSearch(self, rn, seq):
        retLogClust = None

        seqLen = len(seq)
        if seqLen not in rn.childD:
            return retLogClust

        parentn = rn.childD[seqLen]

        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break

            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif '<*>' in parentn.childD:
                parentn = parentn.childD['<*>']
            else:
                return retLogClust
            currentDepth += 1

        logClustL = parentn.childD

        retLogClust = self.fastMatch(logClustL, seq)

        return retLogClust

    def addSeqToPrefixTree(self, rn, logClust):
        
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            firtLayerNode = Node(depth=1, digitOrtoken=seqLen)
            rn.childD[seqLen] = firtLayerNode
        else:
            firtLayerNode = rn.childD[seqLen]

        parentn = firtLayerNode

        currentDepth = 1
        for token in logClust.logTemplate:
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break

            # If token not matched in this layer of existing tree.
            if token not in parentn.childD:
                if not self.hasNumbers(token):
                    if '<*>' in parentn.childD:
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']
                    else:
                        if len(parentn.childD) + 1 < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD) + 1 == self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken='<*>')
                            parentn.childD['<*>'] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD['<*>']

                else:
                    if '<*>' not in parentn.childD:
                        # print("Creating new <*> node for number token")
                        newNode = Node(depth=currentDepth + 1, digitOrtoken='<*>')
                        parentn.childD['<*>'] = newNode
                        parentn = newNode
                    else:
                        # print("Using existing <*> node for number token")
                        parentn = parentn.childD['<*>']

            # If the token is matched
            else:
                # print("Token found in current node's children: ", token)
                parentn = parentn.childD[token]

            currentDepth += 1

    # seq1 is template
    def seqDist(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        simTokens = 0
        numOfPar = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == '<*>':
                numOfPar += 1
                continue #comment@haixuanguo: <*> == <*> are similar pairs
            if token1 == token2:
                simTokens += 1

        retVal = float(simTokens) / len(seq1)

        return retVal, numOfPar

    def fastMatch(self, logClustL, seq):
        retLogClust = None

        maxSim = -1
        maxNumOfPara = -1
        maxClust = None

        for logClust in logClustL:
            curSim, curNumOfPara = self.seqDist(logClust.logTemplate, seq)
            if curSim > maxSim or (curSim == maxSim and curNumOfPara > maxNumOfPara):
                maxSim = curSim
                maxNumOfPara = curNumOfPara
                maxClust = logClust

        if maxSim >= self.st:
            retLogClust = maxClust

        return retLogClust

    def getTemplate(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        retVal = []

        i = 0
        for word in seq1:
            if word == seq2[i]:
                retVal.append(word)
            else:
                retVal.append('<*>')

            i += 1

        return retVal

    def outputResult(self, logClustL):
        log_templates = [0] * self.df_log.shape[0]
        log_templateids = [0] * self.df_log.shape[0]
        df_events = []
        
        checkTemplate2 = []
        i = 0
        for logClust in logClustL:
            checkTemplate2.append(" ".join(logClust.logTemplate))
            template_str = ' '.join(logClust.logTemplate)
            occurrence = len(logClust.logIDL)
            template_id = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logID in logClust.logIDL:
                logID -= 1
                log_templates[logID] = template_str
                log_templateids[logID] = template_id
            df_events.append([template_id, template_str, occurrence])

        df_event = pd.DataFrame(df_events, columns=['EventId', 'EventTemplate', 'Occurrences'])
        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates
        
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        
        #SPJ: Find the compute instance as well and append
        uuid_regex = r'[a-f0-9]{8}-[a-f0-9]{4}-[1-5][a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}'
        self.df_log['ComputeInstance'] = self.df_log['Content'].str.extract(f'({uuid_regex})', flags=re.IGNORECASE)
        #SPJ:end

        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(str(x).encode('utf-8')).hexdigest()[0:8])
        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        
        #SPJ: add index and compute instance
        # Step 1: Sort the DataFrame by 'Occurrences' in descending order
        df_event = df_event.sort_values(by='Occurrences', ascending=False).reset_index(drop=True)

        # Step 2: Create 'templateIndex' with ascending values starting from 1
        df_event['templateIndex'] = range(1, len(df_event) + 1)

        #SPJ:end

        #SPJ: Assign the EventId, based on the 'templateIndex' above

        template_to_index = df_event.set_index('EventTemplate')['templateIndex'].to_dict()
        self.df_log['EventNumericId'] = self.df_log['EventTemplate'].map(template_to_index)

        #SPJ:end
        self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)

        df_event.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=True,
                        columns=["EventId", "EventTemplate", "Occurrences","templateIndex"])

        #SPJ: Create the sequence file as well

        sequence_df = (
            self.df_log
            .groupby('ComputeInstance')['EventNumericId']
            .apply(list)
            .reset_index()
            .rename(columns={'EventNumericId': 'EventSequence'})
        )
        sequence_df.to_csv('drain_sequence.csv', index=False)

        # Convert EventSequence lists to string format for CSV
        sequence_df['EventSequence'] = sequence_df['EventSequence'].apply(lambda seq: str(seq))

        # Save to CSV
        sequence_df.to_csv(os.path.join(self.savePath,'event_sequence.csv'), index=False)

        #SPJ: end

    def printTree(self, node, dep):
        pStr = ''
        for i in range(dep):
            pStr += '\t'

        if node.depth == 0:
            pStr += 'Root'
        elif node.depth == 1:
            pStr += '<' + str(node.digitOrtoken) + '>'
        else:
            pStr += node.digitOrtoken

        print(pStr)

        if node.depth == self.depth:
            return 1
        for child in node.childD:
            self.printTree(node.childD[child], dep + 1)

    def parse(self, mode='train'):
        print('Parsing file: ' + os.path.join(self.path, self.logName))
        start_time = datetime.now()
        # self.logName = logName
        rootNode = Node()
        logCluL = []

        self.load_data()

        count = 0
        checkTemplate= []
        for idx, line in self.df_log.iterrows():

            logID = line['LineId']
            logmessageL = self.preprocess(line['Content']).strip().split()
            matchCluster = self.treeSearch(rootNode, logmessageL)

            if mode == 'train':
            
                if matchCluster is None:
                    checkTemplate.append(" ".join(logmessageL))
                    newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
                    logCluL.append(newCluster)
                    self.addSeqToPrefixTree(rootNode, newCluster)

                # Add the new log message to the existing cluster
                else:
                    newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
                    matchCluster.logIDL.append(logID)
                    if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
                        matchCluster.logTemplate = newTemplate

                # tracer.stop()
                # tracer.save("trace.json")

                count += 1
                if count % 1000 == 0 or count == len(self.df_log):
                    print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)), end='\r')

        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        print("#"*10)
        
        self.outputResult(logCluL)

        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))
        return 'Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time)

    
    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        #print("load_data: ", headers, regex)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)


    def preprocess(self, line):
        additionalCleanUpStrings = ['^- - - - -] ',
                                     r'^[a-f0-9]{32}\s[a-fA-F0-9]{32}\s-\s-\s-\]'
                                     #r'^[a-f0-9]{32}\s[a-f0-9]{32}\s-\s-\s-\]\s\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
                                   ]
        for cleanUpStr in additionalCleanUpStrings:
            line = re.sub(cleanUpStr, '', line)

        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        cnt = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                cnt += 1
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    # print("in Exception in log_to_dataframe")
                    # print("\n", line,"count: ", cnt)
                    # print(e)
                    pass
        print("Total size after encoding is", linecount, cnt)
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf['LineId'] = [i + 1 for i in range(linecount)]

        return logdf

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages"""
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(r' +', '\\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile(r'^' + regex + '$')
        return headers, regex

    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", str(row["EventTemplate"]))
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r' +', r'\\s+', template_regex)
        template_regex = r"^" + template_regex.replace(r"\<\*\>", "(.*?)") + r"$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list