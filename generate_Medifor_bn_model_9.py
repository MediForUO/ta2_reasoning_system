import random
import math
import sys
import os
import sklearn
import re
import json
import numpy as np
import pandas as pd
import argparse
from PIL import Image
from sklearn import linear_model
from skimage import data
from skimage.util.shape import view_as_blocks
from skimage import io
import itertools
import matplotlib.pyplot as plt
from medifor.resources import Resource



def read_bn_training_data(training_file):
    df = pd.read_csv(training_file)
    return df

def initialize_bn_model(dataframe, regional_manipulation_list, global_manipulation_list, schema_dict, rows, cols):
    """ 
        A regional manipulation is a manipulation we have some data about at the pixel level (for example, we have pixel level heatmaps for removal. Global manipulations are manipulations we only have a global variable for. An example of a global variable in the NIST dataset is a lighting change, since we only know whether a change occured in the image, we don't know where it occured.
        We will create a node for each manipulation at the global level. This will be determined by the binary values provided for us in our datafile.
    """
    bn_lines = []
    n = -1
    bn_lines.append('BN {\n\n')
    node_dict = {}
    """
        Creating global variables for each regional manipulation in the model.
    """
    for i in range(len(regional_manipulation_list)):
        n += 1
        node_table_start = 'v' + str(n) + ' {\n    table {\n'
        bn_lines.append(node_table_start)
        node_dict[regional_manipulation_list[i] + '_global'] = n
        counts = dataframe[regional_manipulation_list[i] + '_global'].value_counts()
        """ We are assuming all variables are binary variables for the time being, this may need to be adjusted eventually. """
        sample_space = 0
        for k in counts.keys():
            sample_space += counts[k]
        for s in range(schema_dict[regional_manipulation_list[i]]):
            p = float(counts[s]) / float(sample_space)
            if(p == 0):
                p = 1e-10
            log_prob = math.log(p)
            cpd_line = '    %+f +v%d_%d\n' % (log_prob, n, s)
            bn_lines.append(cpd_line)
        bn_lines.append('  }\n}\n\n')
    for i in range(len(global_manipulation_list)):
        n += 1
        node_table_start = 'v' + str(n) + ' {\n    table {\n'
        bn_lines.append(node_table_start)
        node_dict[global_manipulation_list[i]] = n
        counts = dataframe[global_manipulation_list[i]].value_counts()
        sample_space = 0
        for k in counts.keys():
            sample_space += counts[k]
        for s in range(schema_dict[global_manipulation_list[i]]):
            p = float(counts[s]) / float(sample_space)
            if(p == 0):
                p = 1e-10
            log_prob = math.log(p)
            cpd_line = '    %+f +v%d_%d\n' % (log_prob, n, s)
            bn_lines.append(cpd_line)
        bn_lines.append('  }\n}\n\n')

    """
        Now the regional manipulations are added. To add the regional manipulations we need the number of rows and columns in the grid, this is passed into the function.
    """
    for i in range(rows * cols):
        for m in range(len(regional_manipulation_list)):
            n += 1
            node_table_start = 'v' + str(n) + ' {\n    table {\n'
            bn_lines.append(node_table_start)
            node_dict[regional_manipulation_list[m] + '_hm_sec' + str(i)] = n
            local_df = dataframe[regional_manipulation_list[m] + '_hm_sec' + str(i)].copy()
            local_df[local_df > 0] = 'temp'
            local_df[local_df == 0] = 1
            local_df[local_df == 'temp'] = 0
            global_df = dataframe[regional_manipulation_list[m] + '_global'].copy()
            temp_frame = pd.concat([global_df, local_df], axis=1)
            labelings_sampler = [range(schema_dict[regional_manipulation_list[m]]), range(schema_dict[regional_manipulation_list[m]])]
            possible_labelings = list(itertools.product(*labelings_sampler))
            for labeling in possible_labelings:
                pos_cases = 0
                tot_cases = 0
                for row in map(list, temp_frame.values):
                    if(list(labeling) == row):
                        pos_cases += 1
                    tot_cases += 1
                p = float(pos_cases) / float(tot_cases)
                if(p == 0):
                    p = 1e-10
                log_prob = math.log(p)
                cpd_line = '    ' + str(log_prob) + ' '
                for j in range(len(temp_frame.columns.values)):
                    ## FLAG
                    cpd_line += '+v' + str(node_dict[temp_frame.columns.values[j]]) + '_' + str(labeling[j]) + ' '
                cpd_line += '\n'
                bn_lines.append(cpd_line)
            bn_lines.append('  }\n}\n\n')
    return bn_lines, node_dict

def add_evidence_nodes(df, bn_lines, node_dict, regional_manipulation_list, global_manipulation_list, algorithms_list, schema_dict, rows, cols, query_image_dict, multi_label=False):
    query_evidence = [[], []]
    query_global_algorithms = []
    query_regional_algorithms = []
    """ Find algorithm scores for each algorithm applied to the query image."""
    for i in range(len(algorithms_list)):
        if(query_image_dict[algorithms_list[i]]['score'] != 'nan'):
            query_global_algorithms.append(algorithms_list[i])
            query_evidence[0].append(algorithms_list[i] + '_global')
            query_evidence[1].append(query_image_dict[algorithms_list[i]]['score'])
            #query_evidence[1].append(query_algor_scores[i])
    """ Find the heatmap values for each region of the query image. """
    for i in range(len(algorithms_list)):
        heatmap_file_path = query_image_dict[algorithms_list[i]]['heatmap']
        if(heatmap_file_path != ''):
            query_regional_algorithms.append(algorithms_list[i])
            heatmap = heatmap_file_path.data #io.imread(heatmap_file_path)
            rpb = heatmap.shape[0]/rows
            cpb = heatmap.shape[1]/cols
            blocks = view_as_blocks(heatmap, block_shape=(rpb,cpb))
            for j in range(rows):
                for k in range(cols):
                    query_evidence[0].append(algorithms_list[i] + '_sec' + str(j*rows + k))
                    query_evidence[1].append(np.amin(blocks[j][k]))
    headers = query_evidence.pop(0)
    query_ev_df = pd.DataFrame(query_evidence, columns=headers)
    """
        First we will add the global soft evidence node, which will be a child of all the global manipulations and the global nodes for each regional manipulation.
    """
    """ If multi_label is set to True then the evidence variables created by the network are each children of every manipulation in their section. """
    if(multi_label == True):
        logreg = linear_model.LogisticRegression(C=1e5, solver='sag', multi_class='ovr')
        for i in range(rows*cols):
            n = max(node_dict.values()) + 1
            node_dict['se_sec' + str(i)] = n
            regional_nodes = []
            for manip in regional_manipulation_list:
                regional_nodes.append(manip + '_hm_sec' + str(i))
            regional_algor_list = []
            for algor in algorithms_list:
                regional_algor_list.append(algor + '_sec' + str(i))
            X = df[regional_nodes]
            y = df[regional_algor_list]
            logreg.fit(X,y)
        """ If the multi_label option is set to false (default) then the evidence variables are each children of only 1 parent (a manipulation at their specified level.)"""
    else:
        logreg = linear_model.LogisticRegression(C=1e5)
        manip_list = []
        global_algor_list = []
        query_global_algorithms_list = []
        for algor in query_global_algorithms:
            query_global_algorithms_list.append(algor + '_global')
      #  for algor in algorithms_list:
      #      global_algor_list.append(algor + '_global')
        for manip in regional_manipulation_list:
            manip_list.append(manip + '_global')
        for manip in global_manipulation_list:
            manip_list.append(manip)
        for manip in manip_list:
            if(not query_global_algorithms_list):
                continue
            """ Here we create soft evidence nodes for the global variables. """
            n = max(node_dict.values()) + 1
            se_manip_node = 'se-' + manip
            node_dict[se_manip_node] = n
            node_table_start = 'v' + str(n) + '{\n table {\n'
            bn_lines.append(node_table_start)
            X = df[query_global_algorithms_list].copy()
            y = df[manip].copy()
            
            logreg.fit(X,y)
            log_probs = []
            evidence_list = query_ev_df[query_global_algorithms_list].iloc[[0]].copy()
            log_probs = logreg.predict_log_proba(evidence_list)[0]
            new_line = ""
            new_line += "    " + str(log_probs[0]) + " +v" + str(node_dict[manip]) + "_0 +v" + str(n) + "_0\n"
            new_line += "    " + str(log_probs[1]) + " +v" + str(node_dict[manip]) + "_0 +v" + str(n) + "_1\n"
            new_line += "    " + str(log_probs[1]) + " +v" + str(node_dict[manip]) + "_1 +v" + str(n) + "_0\n"
            new_line += "    " + str(log_probs[0]) + " +v" + str(node_dict[manip]) + "_1 +v" + str(n) + "_1\n"
            bn_lines.append(new_line)
            bn_lines.append('  }\n}\n\n')

        for i in range(rows* cols):
            if(not query_regional_algorithms):
                continue
            query_regional_algorithms_list = []
            for algor in query_regional_algorithms:
                query_regional_algorithms_list.append(algor + '_sec' + str(i))
            regional_algor_list = []
            for algor in algorithms_list:
                regional_algor_list.append(algor + '_sec' + str(i))
            regional_nodes = []
            for manip in regional_manipulation_list:
                n = max(node_dict.values()) + 1
                manip_node = manip + '_hm_sec' + str(i)
                se_manip_node = 'se-' + manip_node
                node_dict[se_manip_node] = n
                node_table_start = 'v' + str(n) + ' {\n table {\n'
                bn_lines.append(node_table_start)
                X = df[query_regional_algorithms_list].copy()
                y = df[manip_node].copy()
                y[y > 0] = 'temp'
                y.replace(0,1,inplace=True)
                y.replace('temp',0,inplace=True)
                logreg.fit(X,y)
                log_probs = []
                evidence_list = query_ev_df[query_regional_algorithms_list].iloc[[0]].copy()
                log_probs = logreg.predict_log_proba(evidence_list)[0]
                new_line = ""
                new_line += "    " + str(log_probs[0]) + " +v" + str(node_dict[manip + '_hm_sec' + str(i)]) + "_0 +v" + str(n) + "_0\n"
                new_line += "    " + str(log_probs[1]) + " +v" + str(node_dict[manip + '_hm_sec' + str(i)]) + "_0 +v" + str(n) + "_1\n"
                new_line += "    " + str(log_probs[1]) + " +v" + str(node_dict[manip + '_hm_sec' + str(i)]) + "_1 +v" + str(n) + "_0\n"
                new_line += "    " + str(log_probs[0]) + " +v" + str(node_dict[manip + '_hm_sec' + str(i)]) + "_1 +v" + str(n) + "_1\n"
                bn_lines.append(new_line)
                bn_lines.append('  }\n}\n\n')
        
        return bn_lines, node_dict

    """ Now we must add the evidence nodes for the local manipulations. """

def add_schema_line(bn_lines, schema_dict, node_dict):
    bn_schema_line = ''
    for i in range(max(node_dict.values()) + 1):
        for node_name, node_number in node_dict.items():
            if(node_number == i):
                node_name = re.sub(r'_.*$', r'', node_name)
                bn_schema_line += str(schema_dict[node_name]) + ','
    bn_schema_line = re.sub(r',$', r'\n', bn_schema_line)
    bn_lines.insert(0, bn_schema_line)
    return bn_lines

def write_base_bn_file(bn_file, bn_lines):
    bn = open(bn_file, 'w')
    for line in bn_lines:
        bn.write(line)
    bn.close()
    return bn_file
        
def write_bn_file(bn_file, bn_lines):
    bn = open(bn_file, 'w')
    for line in bn_lines:
        bn.write(line)
    bn.write('}')
    bn.close()
    return bn_file

def create_evidence_file(node_dict):
    node_list = []
    for key, value in node_dict.items():
        node = [value, key]
        node_list.append(node)
    node_list.sort()
    evidence_file_line = ''
    re_se = re.compile(r'^se-.*')
    for node in node_list:
        if(re_se.match(node[1])):
            evidence_file_line += '1,'
        else:
            evidence_file_line += '*,'
    evidence_file_line = re.sub(',$', r'', evidence_file_line)
    evidence_file = 'libra_evidence.ev'
    evidence = open(evidence_file, 'w')
    evidence.write(evidence_file_line)
    evidence.close()
    return evidence_file

def write_node_file(node_file, node_dict):
    with open(node_file, 'w') as f:
        json.dump(node_dict, f)

def load_node_dict(node_dict_file):
    with open(node_dict_file, 'r') as f:
        try:
            data = json.load(f)
        # If the file is empty a value error will be thrown.
        except ValueError:
            data = {}
    return data

def run_libra_bp_inference(bn_model_file, bn_evidence_file, node_dict, query_dict, local_manipulation_list, global_manipulation_list, rows, cols):
    os.system('libra bp -m ' + bn_model_file + ' -ev ' + bn_evidence_file + '> MediFor_inference_output.txt')
    libra_output = open('MediFor_inference_output.txt', 'r')
    libra_output_lines = libra_output.readlines()
    libra_output.close()
    shape = query_dict['image'].data.shape
    results_dict = {}
    heatmap_region_dicts = {}
    for key, value in node_dict.items():
        if('se-' in key):
            continue
        line = libra_output_lines[value]
        line = re.sub(r'^[^ ]* ', r'', line)
        line = re.sub(r'\n', r'', line)
        confidence = float(line)
        manip = re.sub(r'_.*', r'', key)
        if(manip not in results_dict.keys()):
            results_dict[manip] = {}
            heatmap_region_dicts[manip] = {}
        if(key in global_manipulation_list or '_global' in key):
            results_dict[manip]['confidence'] = confidence
        if('_sec' in key):
            section = int(re.sub(r'^[^_]*_hm_sec', r'', key))
            heatmap_region_dicts[manip][section] = confidence
    for manip in heatmap_region_dicts.keys():
        if(manip in global_manipulation_list):
            continue
        heatmap_array = np.concatenate([np.concatenate([np.full([shape[0]/rows, shape[1]/cols], heatmap_region_dicts[manip][x]) for x in range(i*rows, i*rows + cols)], axis=1) for i in range(rows)], axis=0)
        rescaled = (255 * (1- heatmap_array)).astype(np.uint8)
        im = Image.fromarray(rescaled)
        results_dict[manip]['heatmap'] = Resource('image', im, 'image/png')
    return results_dict

def train_CL_model():
    rows = 2
    cols = 2
    NIST_df = read_bn_training_data('test_NIST_v15.csv')
    NIST_regional_manips = ['removal', 'splice', 'copyclone']
    NIST_global_manips = ['lighting']
    NIST_manip_schema_dict = {'removal': 2, 'splice': 2, 'copyclone': 2, 'lighting': 2, 'se-removal': 2, 'se-splice': 2, 'se-copyclone': 2, 'se-lighting': 2}
    NIST_TA1_algorithms = ['block01', 'block02', 'copymove01', 'dct01', 'dct02', 'dct03_A', 'dct03_NA', 'ela01', 'noise01', 'noise02']
    query_dict = {}
    for algor in NIST_TA1_algorithms:
        query_dict[algor] = {'score': 'nan', 'heatmap': ''}
    NIST_base_bn_lines, node_dict = initialize_bn_model(NIST_df, NIST_regional_manips, NIST_global_manips, NIST_manip_schema_dict, rows, cols)
    bn_base_file = write_bn_file('libra_bn_base_file.bn', NIST_base_bn_lines)
    return rows, cols, NIST_df.copy(), NIST_regional_manips[:], NIST_global_manips[:], NIST_manip_schema_dict.copy(), NIST_TA1_algorithms[:], NIST_base_bn_lines[:], node_dict.copy(), query_dict.copy()

def train_model():
    rows = 2
    cols = 2
    NIST_df = read_bn_training_data('test_NIST_v15.csv')
    NIST_regional_manips = ['removal', 'splice', 'copyclone']
    NIST_global_manips = ['lighting']
    NIST_manip_schema_dict = {'removal': 2, 'splice': 2, 'copyclone': 2, 'lighting': 2, 'se-removal': 2, 'se-splice': 2, 'se-copyclone': 2, 'se-lighting': 2}
    NIST_TA1_algorithms = ['block01', 'block02', 'copymove01', 'dct01', 'dct02', 'dct03_A', 'dct03_NA', 'ela01', 'noise01', 'noise02']
    query_dict = {}
    for algor in NIST_TA1_algorithms:
        query_dict[algor] = {'score': 'nan', 'heatmap': ''}
    NIST_base_bn_lines, node_dict = initialize_bn_model(NIST_df, NIST_regional_manips, NIST_global_manips, NIST_manip_schema_dict, rows, cols)
    bn_base_file = write_base_bn_file('libra_bn_base_file.bn', NIST_base_bn_lines)
    node_dict_file = write_node_file('node_dict_file.json', node_dict)
    



def run_CL_inference(rows, cols, df, regional_manipulations, global_manipulations, manipulation_schema_dict, algorithms_list, bn_lines, node_dict, query_dict):
    node_bn_lines, node_dict = add_evidence_nodes(df, bn_lines, node_dict, regional_manipulations, global_manipulations, algorithms_list, manipulation_schema_dict, rows, cols, query_dict)
    bn_lines = add_schema_line(node_bn_lines, manipulation_schema_dict, node_dict)
    write_bn_file('libra_bn_model.bn', bn_lines)
    evidence_file = create_evidence_file(node_dict)
    run_libra_bp_inference('libra_bn_model.bn', evidence_file, node_dict, query_dict)

def run_inference(query_dict):
    for algor in query_dict['algorithms']:
        if(algor not in query_dict.keys()):
            query_dict[algor] = {'score': 'nan', 'heatmap': ''}
        else:
            if('score' not in query_dict[algor].keys()):
                query_dict[algor]['score'] = 'nan'
            if('heatmap' not in query_dict[algor].keys()):
                query_dict[algor]['heatmap'] = ''
    bn_base_file = 'libra_bn_base_file.bn'
    node_dict = load_node_dict('node_dict_file.json')
    df = read_bn_training_data('test_NIST_v15.csv')
    rows = 2
    cols = 2
    local_manipulations = ['copyclone', 'removal', 'splice']
    global_manipulations = ['lighting']
    manipulation_schema_dict = {'removal': 2, 'splice': 2, 'copyclone': 2, 'lighting': 2, 'se-removal': 2, 'se-splice': 2, 'se-copyclone': 2, 'se-lighting': 2}
    algorithms_list = query_dict['algorithms']
    bn_base = open(bn_base_file, 'r')
    bn_base_lines = bn_base.readlines()
    node_bn_lines, node_dict = add_evidence_nodes(df, bn_base_lines, node_dict, local_manipulations, global_manipulations, algorithms_list, manipulation_schema_dict, rows, cols, query_dict)
    bn_lines = add_schema_line(node_bn_lines, manipulation_schema_dict, node_dict)
    write_bn_file('libra_bn_model.bn', bn_lines)
    evidence_file = create_evidence_file(node_dict)
    results_dict = run_libra_bp_inference('libra_bn_model.bn', evidence_file, node_dict, query_dict, local_manipulations, global_manipulations, rows, cols)
    return results_dict
