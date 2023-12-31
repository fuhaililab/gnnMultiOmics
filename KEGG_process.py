import ujson as json
import pandas as pd
import numpy as np
from itertools import chain
import networkx as nx
import torch

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



def reindex_nx_graph(G, ordered_node_list):
    """Reindex the nodes in nx graph according to given ordering.
    Args:
    G (networkx): Networkx graph.
    ordered_node_list (list): A list served as node ordering.
    """
    ordered_node_dict = dict(zip(ordered_node_list, range(len(ordered_node_list))))
    return nx.relabel_nodes(G, ordered_node_dict)



def generate_gene_mapping_from_KEGG(save_dir, species="human"):
    """Generate gene id mapping from KEGG signaling pathway dataset.
    Args:
        save_dir (str): path to the save dir for processed data.
        species (str): The specie of the data, choose from human/mouse
    """
    #change file path
    KEGG = pd.read_csv(f"data/KEGG/full_kegg_pathway_list_{species}.csv", index_col=0)
    KEGG_source_gene_list = KEGG["source"].to_list()
    KEGG_target_gene_list = KEGG["target"].to_list()

    total_gene_list = list(chain(*[KEGG_source_gene_list, KEGG_target_gene_list]))
    unique_gene = np.unique(np.array(total_gene_list))
    gene_dict = {}
    for i, g in enumerate(unique_gene):
        gene_dict[g] = i
    save_json(f"{save_dir}/gene_mapping_{species}.json", gene_dict, f"gene mapping for {species}")



def generate_KEGG_graph_data(save_dir, species="human"):
    """Generate KEGG signaling pathway graph data.
    Args:
        save_dir (str): path to the save dir for processed data.
        species (str): The specie of the data, choose from human/mouse
    """
    gene_dict = load_json(f"{save_dir}/gene_mapping_{species}.json")
    KEGG = pd.read_csv(f"data/KEGG/full_kegg_pathway_list_{species}.csv", index_col=0)
    gene_list = list(gene_dict.keys())
    source_keep_index = [True if g in gene_list else False for g in KEGG["source"]]
    target_keep_index = [True if g in gene_list else False for g in KEGG["target"]]
    KEGG_filter = KEGG.loc[np.logical_and(source_keep_index, target_keep_index), ]

    # save pathway
    unique_pathway = np.unique(KEGG_filter["pathway_name"])
    KEGG_dict = {}
    for i, p in enumerate(unique_pathway):
        KEGG_dict[p] = i
    save_json(f"{save_dir}/KEGG_mapping_{species}.json", KEGG_dict, f"KEGG signaling pathway mapping for {species}")

    #save edge type:
    unique_edge_type = np.unique(KEGG_filter["edge_type"])
    edge_dict = {}
    for i, e in enumerate(unique_edge_type):
        edge_dict[e] = i
    save_json(f"{save_dir}/KEGG_edge_mapping_{species}.json", edge_dict, f"KEGG edge type mapping for {species}")

    # gene to pathway
    gene_pathway = {}
    for gene_ensembl in gene_dict.keys():
        gene_pathway[gene_ensembl] = set()

    for i in range(KEGG_filter.shape[0]):
        source_gene = KEGG_filter.iloc[i, 1]
        target_gene = KEGG_filter.iloc[i, 3]
        pathway = KEGG_filter.iloc[i, 6]
        gene_pathway[source_gene].add(KEGG_dict[pathway])
        gene_pathway[target_gene].add(KEGG_dict[pathway])

    for k in gene_pathway.keys():
        gene_pathway[k] = list(gene_pathway[k])
    save_json(f"{save_dir}/gene_KEGG_mapping_{species}.json", gene_pathway, f"gene to KEGG signaling pathway mapping for {species}")

    # save edge list
    G = nx.MultiDiGraph()
    for i in range(KEGG_filter.shape[0]):
        source_gene = KEGG_filter.iloc[i, 1]
        target_gene = KEGG_filter.iloc[i, 3]
        direct = KEGG_filter.iloc[i, 4]
        edge_type = KEGG_filter.iloc[i, 5]
        pathway = KEGG_filter.iloc[i, 6]
        if direct == "directed":
            G.add_edge(gene_dict[source_gene], gene_dict[target_gene], edge_type=edge_dict[edge_type], pathway=KEGG_dict[pathway])
        else:
            G.add_edge(gene_dict[source_gene], gene_dict[target_gene], edge_type=edge_dict[edge_type], pathway=KEGG_dict[pathway])
            G.add_edge(gene_dict[target_gene], gene_dict[source_gene], edge_type=edge_dict[edge_type], pathway=KEGG_dict[pathway])

    data = nx.node_link_data(G)
    save_json(f"{save_dir}/KEGG_graph_{species}.json", data, f"KEGG graph for {species}")



def construct_KEGG_graph_data(gene_ids, G, num_pathways, num_edge_type):
    """Construct graph from KEGG signaling pathways
    Args:
        gene_ids (list): Gene index list.
        G (nx.MultiDiGraph): The overall KEGG signaling pathway graph.
        num_pathways (int): Number of pathways.
        num_edge_type (int): Number of edge type in KEGG.
    """
    # process graph data
    sub_G = G.subgraph(gene_ids)
    sub_G = reindex_nx_graph(sub_G, gene_ids)
    edges = sub_G.edges
    edge_list = []
    edge_attr_list = []
    for u, v, k in edges:
        if (u, v) in edge_list:
            continue
        edge = (u, v)
        edge_list.append(edge)
        edge_pathway = torch.zeros([1, num_pathways])
        edge_attr = torch.zeros([1, num_edge_type])
        edge_data = sub_G.get_edge_data(u, v)
        for result in edge_data.values():
            edge_attr[0, result["edge_type"]] = 1.0
            edge_pathway[0, result["pathway"]] = 1.0
        edge_attr_list.append(edge_attr)

    edge_attr = torch.cat(edge_attr_list, dim=0).float()
    edge_index = torch.from_numpy(np.array(edge_list).T).long()


    return edge_attr, edge_index
