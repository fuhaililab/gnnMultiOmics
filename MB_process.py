"""
Zitian Tang
Generate model input, gene and edge lists, and edge index.
"""

import numpy as np
import pandas as pd
import ujson as json
from itertools import chain
import networkx as nx
import torch


# Define functions needed
def save_json(filename, obj, message=None, ascii=True):
    """Save data in JSON format.
    Args:
        filename (str): Name of save directory (including file name).
        obj (object): Data you want to save.
        message (str): Anything you want to print.
    """
    if message is not None:
        print(f"Saving {message}...")
    with open(filename, "w") as fh:
        json.dump(obj, fh, ensure_ascii=ascii)


def load_json(filename):
    """Load data from JSON format.
    Args:
        filename (str): Path of JSON file.
    """
    with open(filename) as f:
        file = json.load(f)
    return file


def reorder_nx_graph(G):
    """Sort nodes in nx graph.
    Args:
    G (networkx): Networkx graph.
    """
    H = nx.DiGraph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))
    return H


def reindex_nx_graph(G, ordered_node_list):
    """Reindex the nodes in nx graph according to given ordering.
    Args:
    G (networkx): Networkx graph.
    ordered_node_list (list): A list served as node ordering.
    """
    ordered_node_dict = dict(zip(ordered_node_list, range(len(ordered_node_list))))
    G = nx.relabel_nodes(G, ordered_node_dict)
    return G


def filter_genes(filtered_ex, KEGG, control, cases):
    """Filter the datasets according to KEGG full pathway genes, keep only those in KEGG
    Args:
        filtered_ex: example filtered data, its index is the gene list of MB dataset.
        KEGG: KEGG full human gene pathway, contains source and target genes.
        control: previously generated control dataset (include WNT & SHH).
        cases: previously generated case dataset (Gr3 or Gr4 depend on case_type input).
    Returns:
        gene_dict: gene dictionary generated from KEGG.
        gene_list: list of genes of KEGG database.
        genes_filtered: 4224 genes that has been filtered out (both in KEGG and in dataset.
        filtered_control: control dataset with genes filtered according to KEGG.
        filtered_gr3: group 3 dataset with genes filtered according to KEGG.
        filtered_gr4: group 4 dataset with genes filtered according to KEGG.
    """
    source_gene_list = KEGG["source"].to_list()
    target_gene_list = KEGG["target"].to_list()
    total_gene_list = list(chain(*[source_gene_list, target_gene_list]))
    unique_gene = np.unique(np.array(total_gene_list))
    gene_dict = {}
    for i, g in enumerate(unique_gene):
        gene_dict[g] = i

    gene_list = list(gene_dict.keys())
    sample_gene_list = list(filtered_ex.index)
    filter_idx = np.isin(sample_gene_list, np.array(gene_list))
    genes_filtered = filtered_ex.index[filter_idx]

    filtered_control = control[:, filter_idx, ]
    filtered_cases = cases[:, filter_idx, ]

    return gene_dict, gene_list, genes_filtered, filtered_control, filtered_cases


def generate_kegg_graph(KEGG, gene_dict, gene_list):
    """Generate directed graph based on filtered genes.
    Args:
        KEGG: KEGG full human gene pathway, contains source and target genes.
        gene_dict: gene dictionary generated from KEGG.
        gene_list: list of genes of KEGG database.
    Returns:
        G: graph with x nodes and x edges.
    """
    source_keep_index = [True if g in gene_list else False for g in KEGG["source"]]
    target_keep_index = [True if g in gene_list else False for g in KEGG["target"]]
    KEGG_filter = KEGG.loc[np.logical_and(source_keep_index, target_keep_index), ]
    print(len(source_keep_index), len(target_keep_index), len(KEGG_filter))

    # save edge type:
    unique_edge_type = np.unique(KEGG_filter["edge_type"])
    edge_dict = {}
    for i, e in enumerate(unique_edge_type):
        edge_dict[e] = i

    # save edge list
    G = nx.MultiDiGraph()
    for i in range(KEGG_filter.shape[0]):
        source_gene = KEGG_filter.iloc[i, 1]
        target_gene = KEGG_filter.iloc[i, 3]
        direct = KEGG_filter.iloc[i, 4]
        edge_type = KEGG_filter.iloc[i, 5]
        if direct == "directed":
            G.add_edge(gene_dict[source_gene], gene_dict[target_gene], edge_type=edge_dict[edge_type])
        else:
            G.add_edge(gene_dict[source_gene], gene_dict[target_gene], edge_type=edge_dict[edge_type])
            G.add_edge(gene_dict[target_gene], gene_dict[source_gene], edge_type=edge_dict[edge_type])

    return G


def construct_graph_data(genes_filtered, gene_dict, G):
    """Construct edge_list and edge_index, and subgraph based on genes filtered.
    Args:
        genes_filtered: filtered genes exist both in KEGG and MB dataset.
        gene_dict: gene dictionary for all KEGG genes.
        G: original graph contains all nodes & paths in KEGG.
    Returns:
        edge_list: list of edges in subgraph.
        edge_index: edge index for 1hop edges.
        gene_ids: ids of the filtered genes.
    """
    gene_ids = [gene_dict[g] for g in genes_filtered]

    # process graph data
    sub_G = G.subgraph(gene_ids).copy()
    sub_G = reindex_nx_graph(sub_G, gene_ids)
    edges = sub_G.edges
    edge_set = set()
    for u, v, k in edges:
        if (u, v) in edge_set:
            continue
        edge = (u, v)
        edge_set.add(edge)

    edge_list = list(edge_set)
    edge_index = torch.from_numpy(np.array(edge_list).T).long()

    return gene_ids, edge_list, edge_index


def construct_2hop_edge_index(gene_ids, edge_list):
    """Construct 2hop edge index and corresponding mask.
    Args:
        gene_ids: ids of the filtered genes.
        edge_list: list of edges in subgraph.
    Returns:
        x
    """
    temp_G = nx.DiGraph()
    temp_G.add_nodes_from(range(len(gene_ids)))
    temp_G.add_edges_from(edge_list)
    temp_G = reorder_nx_graph(temp_G)
    adj = nx.adjacency_matrix(temp_G)
    adj = adj.toarray()

    # construct 2 hop edge_index
    adj = np.mat(adj)
    adj2 = adj ** 2
    adj2[adj2 > 1] = 1
    np.fill_diagonal(adj2, 0)
    Gk = nx.from_numpy_matrix(adj2, create_using=nx.DiGraph)
    Gk.add_edges_from(temp_G.edges(data=True))
    print(Gk.number_of_edges(), Gk.number_of_nodes(), temp_G.number_of_edges(), temp_G.number_of_nodes())
    edge_index_2 = np.array(Gk.edges).T

    # construct mask
    mask = np.zeros((len(edge_index_2[0]), 2))
    for i in range(len(edge_index_2[0])):
        u, v = edge_index_2[:, i]
        if temp_G.has_edge(u, v):
            mask[i, 0] = 1
        else:
            mask[i, 1] = 1

    return temp_G, Gk, edge_index_2, mask


def generate_input(control, cases, case_type, zero_label=False):
    """Generate input data (x) and labels (y) for model input.
    Args:
        control: control dataset (including WNT and SHH).
        cases: cases dataset.
        case_type: can either be Gr3 or Gr4.
        zero_label: whether need to zero out control genes.
    Returns:
        input_data: input data x into the model.
        ids: gene ids unique for each gene.
        y: input ground truth labels y into the model.
    """
    # remove nan values in Gr3.
    if case_type == "Gr3":
        print(f'removing nans in {case_type}!')
        nan_idx = np.where(np.isnan(cases))[0]
        mean_val = np.mean(~np.isnan(cases[:, 1418, 0]))
        cases[nan_idx, 1418, 0] = mean_val

    if zero_label:
        print(f'zeroing corresponding genes in control!')
        zero_idx = np.where(cases[0][:, 0] == 0)
        control[:, zero_idx, 0] = 0

    input_data = np.concatenate((control, cases), axis=0)

    # add unique gene id to input data
    a, b, c = input_data.shape
    ids = np.arange(b)

    y = [0] * len(control) + [1] * len(cases)

    return input_data, ids, y


def main():
    # Define data paths #
    case_type = "gr4"
    gender_type = 'allsamples'
    data_path = "MB_project/data/preprocessed_data/0321new"
    save_path = "MB_project/data"

    filtered_ex = pd.read_csv(f"{data_path}/wnt_M_md_filtered.csv", index_col=0)
    KEGG = pd.read_csv(f"{data_path}/full_kegg_pathway_list_human_symbol.csv", index_col=0)

    control = np.load(f"{data_path}/control_{gender_type}_wnt_shh.npz")['data']
    cases = np.load(f"{data_path}/{case_type}_{gender_type}_preprocessed_data.npz")['data']

    # Data processing #
    gene_dict, gene_list, genes_filtered, filtered_control, filtered_cases = filter_genes(filtered_ex,
                                                                                          KEGG, control, cases)
    G = generate_kegg_graph(KEGG, gene_dict, gene_list)
    gene_ids, edge_list, edge_index = construct_graph_data(genes_filtered, gene_dict, G)
    temp_G, Gk, edge_index_2, mask = construct_2hop_edge_index(gene_ids, edge_list)

    input_data, ids, y = generate_input(filtered_control, filtered_cases, case_type, False)
    # save final input
    np.savez(f"{save_path}/processed_data_{case_type}_{gender_type}.npz", input=input_data, output=y,
             edge_index=edge_index_2, mask=mask, ids=ids)  # prev: ids=ids

    # save for later use #
    np.save(f'{data_path}/filtered_genes.npy', genes_filtered)
    save_json(f"{save_path}/gene_dict.json", gene_dict, f"gene mapping for all")
    np.savez_compressed(f"{save_path}/control_{gender_type}_filtered.npz", data=filtered_control)
    np.savez_compressed(f"{save_path}/case_{case_type}_{gender_type}_filtered.npz", data=filtered_cases)
    np.save(f'{save_path}/gene_ids.npy', gene_ids)

    nx.write_gpickle(G, f'{save_path}/KEGG_graph_5311_130666.gpickle')  # 5311 nodes, 130666 edges
    nx.write_gpickle(temp_G, f'{save_path}/graph_1hop_4224_43071.gpickle')  # 4224 nodes, 43071 edges
    nx.write_gpickle(Gk, f'{save_path}/graph_2hop_4224_294736.gpickle')  # 4224 nodes, 294736 edges

    ###
    test_1 = np.load(f'{save_path}/processed_data_gr4_allsamples.npz')['input']


if __name__ == "__main__":
    main()
