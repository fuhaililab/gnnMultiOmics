import pandas as pd
import numpy as np

import networkx as nx
import pickle

import seaborn as sns
import matplotlib.pyplot as plt


def compare_disgenet(disgenet_vars, result_G):
    tp = 0
    fp = 0
    fn = 0
    overlapped = []

    for node in result_G.nodes:
        if node in disgenet_vars['Gene'].values:
            tp += 1
            overlapped.append(node)
        else:
            fp += 1

    for gene in disgenet_vars['Gene'].values:
        if gene not in result_G.nodes:
            fn += 1

    return tp, fp, fn, overlapped


def find_related_genes(msigdb, result_G):
    related_genes = []
    for node in result_G.nodes:
        description = msigdb.loc[msigdb['human_gene_symbol'] == node, 'gs_desc_short'].values
        if len(description) > 0 and 'inflammatory resp' in description[0]:
            related_genes.append(node)
    print(related_genes)


def main():
    case_type = "Gr3"  # Gr3 or Gr4
    gender_type = 'M'  # 'M', 'F', or 'allsamples'
    graph_path = 'GNN_GAT_GCN/graphs/0401new'
    disgenet_path = 'data/0327_new/C0025149_disease_vda_summary.csv'
    # graph_path = './graphs/0401new'
    # disgenet_path = '../data/0327_new/C0025149_disease_vda_summary.csv'
    disgenet_vars = pd.read_csv(disgenet_path)
    msigdb = pd.read_csv('data/mSigDB_hallmark_human.csv')

    with open(f'{graph_path}/{case_type}_{gender_type}_200e.gpickle', 'rb') as f:
        result_G = pickle.load(f)

    tp, fp, fn, overlapped = compare_disgenet(result_G, disgenet_vars)
    print(f"In {case_type}-{gender_type}, "
          f"the precision is: {(tp / (tp + fp)):.2f}; "
          f"the recall is: {(tp / (tp + fn)):.2f}; overlapped genes are: {overlapped}. \n")

    # for case_type in {'Gr3', 'Gr4'}:
    #     for gender_type in {'M', 'F', 'allsamples'}:
    #         with open(f'{graph_path}/{case_type}_{gender_type}_200e.gpickle', 'rb') as f:
    #             result_G = pickle.load(f)
    #         tp, fp, fn, overlapped = compare_disgenet(result_G, disgenet_vars)
    #         print(f"In {case_type}-{gender_type}, "
    #               f"the precision is: {(tp/(tp+fp)):.2f}; "
    #               f"the recall is: {(tp/(tp+fn)):.2f}; overlapped genes are: {overlapped}. \n")


if __name__ == "__main__":
    main()
