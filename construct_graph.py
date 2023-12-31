"""
Zitian Tang
Reconstruct graph using attention values saved during model training.
"""

import pandas as pd
import numpy as np
import json

import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, t, ttest_ind_from_stats


def find_path(source, target, attention, edge_index, Gk):
    """Find (among all paths between 2hop neighbors) the highest attention edge.
    Args:
        source: source node.
        target: target node.
        attention: entire attention ndarray saved from training.
        edge_index: include source & target node of each edge.
        Gk: 1hop graph including all nodes of interest.
    Returns:
        path with the highest attention score.
    """
    path = np.array([p for p in nx.all_shortest_paths(Gk, source=source, target=target)])
    # print(path.shape)

    if path.shape[1] == 2:
        return np.array([[path[0,0],path[0,1]],[path[0,0],path[0,1]]])
    if path.shape[0] == 1:
        return np.array([[path[0,0],path[0,1]],[path[0,1],path[0,2]]])
    else:
        attention_score = []
        for i in range(path.shape[0]):
            edge1 = np.array([path[i,0],path[i,1]])
            edge2 = np.array([path[i, 1], path[i, 2]])
            edge1_index = np.where((edge_index[0, :] == edge1[0]) &(edge_index[1, :] == edge1[1]))
            edge1_attention = attention[edge1_index,0]+attention[edge1_index,1]
            edge2_index = np.where((edge_index[0, :] == edge2[0]) & (edge_index[1, :] == edge2[1]))
            edge2_attention = attention[edge2_index,0] + attention[ edge2_index,1]
            attention_score.append(edge1_attention*edge2_attention)
        k = attention_score.index(max(attention_score))
        return np.array([[path[k,0],path[k,1]],[path[k,1],path[k,2]]])


def load_data(data_path, attention_path, loss_path):
    """Load all saved data.
    Args:
        data_path: data saving directory.
        attention_path: attention saving directory.
    Returns:
        loaded edge_index, hop_mask, graph, loss_list, and attention.
    """
    # edge_index
    edge_index = np.load(f'{data_path}/processed_data_Gr4_allsamples.npz', allow_pickle=True)["edge_index"]  # (2, num edges)
    hop_mask = np.load(f'{data_path}/processed_data_Gr4_allsamples.npz', allow_pickle=True)["mask"]

    # saved graphs
    G = nx.read_gpickle(f'{data_path}/graph_1hop_4224_43071.gpickle')
    # saved loss list
    loss_list = pd.read_csv(f'{loss_path}').to_numpy()  # (10,16) (num batches, num sample)
    # attention list
    attention_all = np.load(f'{attention_path}', allow_pickle=True)["attention"]

    return edge_index, hop_mask, G, loss_list, attention_all


def calculate_weighted_attention(edge_index, loss_list, attention_all):
    """Calculate a weighted attention (weighted by loss).
    Args:
        edge_index: include source & target node for each edge.
        loss_list: include loss for each batch during training.
        attention_all: attention saved for all edges.
    Returns:
        the weighted attention as ndarray.
    """
    if len(attention_all.shape) == 1:
        for i in range(attention_all.shape[0]):
            attention_all[i] = np.delete(attention_all[i], -1, axis=1)
    else:
        attention_all = np.delete(attention_all, -1, axis=2)

    weighted_attention = np.zeros((edge_index.shape[1], 2))
    weights = 1 / np.exp(loss_list)
    for i in range(loss_list.shape[0]):
        for j in range(loss_list.shape[1]):
            if np.isnan(loss_list[i][j]):
                continue
            weighted_attention += weights[i, j] * attention_all[i][j].sum(2) / 4

    # E * 2
    weighted_attention = weighted_attention/ np.nansum(weights[~np.isnan(weights)])

    return weighted_attention


def find_2hop_edges(edge_index, weighted_attention, G):
    """Extend all edges (including 1hop and 2hop) to fit with proper attention scores.
    Args:
        edge_index: include source & target node for each edge.
        weighted_attention: loss-weighted attention score for each edge.
        G: 1hop graph containing all nodes of interest.
    Returns:
        all edges.
    """
    total_edges = np.array([])
    for m in range(edge_index.shape[1]):
        edge_paths = find_path(edge_index[0, m], edge_index[1, m], weighted_attention, edge_index, G)
        if len(total_edges) == 0:
            total_edges = edge_paths
        total_edges = np.vstack((total_edges, edge_paths))
    return total_edges


def add_attention_to_graph(total_edges, edge_index, weighted_attention):
    attention_list = []
    # loop through total_edges, add attention scores to each edges
    for i in range(0, total_edges.shape[0], 2):
        if np.array_equal(total_edges[i], total_edges[i + 1]):
            # 1-hop edge
            source = total_edges[i][0]
            target = total_edges[i][1]
            idx = np.where((edge_index[0, :] == source) & (edge_index[1, :] == target))[0][0]
            attention_list.append(weighted_attention[idx, 0])
            attention_list.append(weighted_attention[idx, 0])
        else:
            # 2-hop edge
            source = total_edges[i][0]
            target = total_edges[i + 1][1]
            idx = np.where((edge_index[0, :] == source) & (edge_index[1, :] == target))[0][0]
            attention_list.append(weighted_attention[idx, 1])
            attention_list.append(weighted_attention[idx, 1])
    edges_with_attention = np.concatenate([total_edges, np.expand_dims(attention_list, axis=1)], axis=1)

    # construct graph
    edge_weights = {}
    result_G = nx.DiGraph()
    for r in edges_with_attention:
        source, target, attention_score = r
        source = int(source)
        target = int(target)

        if (source, target) not in edge_weights:
            edge_weights[(source, target)] = attention_score
        else:
            edge_weights[(source, target)] += attention_score

    for edge, weight in edge_weights.items():
        result_G.add_weighted_edges_from([edge + (weight,)])  # (source, target, weight)

    return edges_with_attention, result_G


def add_edges_to_graph(result_G, data_path, fold_change, gene_description, k):
    # add source of change column
    fold_change['source_of_change'] = ""
    fold_change.loc[(fold_change['md_fc'] > 0.01) & (fold_change['md_pv'] < fold_change['ep_pv']), 'source_of_change'] \
        = 'methyl'
    fold_change.loc[(fold_change['md_fc'] > 0.01) & (fold_change['ep_pv'] < fold_change['md_pv']), 'source_of_change'] \
        = 'express'
    fold_change.loc[fold_change['md_fc'] < 0.01, 'source_of_change'] = 'express'

    # add node attributes
    gene_attr_dict = {r['gene_name']: {'methyl_fold_change': r['md_fc'],
                                       'express_fold_change': r['ep_fc'],
                                       'methyl_p_val': r['md_adj_pv'],
                                       'express_p_val': r['ep_adj_pv'],
                                       'source_of_change': r['source_of_change']} for _, r in fold_change.iterrows()}

    sorted_edges = sorted(result_G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
    top_k_edges = sorted_edges[:k]
    # # filter edges with gene p-val >= 0.05
    # filtered_edges = []
    # for edge in top_k_edges:
    #     if edge[0] in gene_attr_dict.keys() and edge[1] in gene_attr_dict.keys():
    #         print('yes its in')
    #         if (min(gene_attr_dict[edge[0]]['methyl_p_val'], gene_attr_dict[edge[0]]['express_p_val']) < 0.05) \
    #                 or (min(gene_attr_dict[edge[1]]['methyl_p_val'], gene_attr_dict[edge[1]]['express_p_val']) < 0.05):
    #             filtered_edges.append((edge[0], edge[1], edge[2]['weight']))

    top_k_graph = nx.DiGraph()
    top_k_graph.add_edges_from([(edge[0], edge[1], {'weight': edge[2]['weight']}) for edge in top_k_edges])

    gene_ids = np.load(f'{data_path}/gene_ids.npy')
    ids = list(range(len(gene_ids)))
    ids_mapping = {ids[i]: gene_ids[i] for i in range(len(ids))}
    top_k_2 = nx.relabel_nodes(top_k_graph, ids_mapping)

    with open(f"{data_path}/gene_dict.json") as gd:
        gene_dict = json.load(gd)
    gene_dict_inv = {v: k for k, v in gene_dict.items() if v in gene_ids}

    H = nx.relabel_nodes(top_k_2, gene_dict_inv)

    for gene in gene_attr_dict.keys():
        if gene in gene_description['human_gene_symbol'].values:
            # gene_ds = gene_description.loc[gene_description['human_gene_symbol'] == gene, 'gs_index'].iloc[0]
            gene_ds = gene_description.loc[gene_description['human_gene_symbol'] == gene, 'gs_index'].tolist()
            gene_ds = list(set(gene_ds))
        else:
            gene_ds = [0]
        # Add the gene_description attribute to the gene_attr_dict
        gene_attr_dict[gene]['gene_description'] = gene_ds

    for n in H.nodes():
        if n in gene_attr_dict:
            H.nodes[n]['methyl_fold_change'] = gene_attr_dict[n]['methyl_fold_change']
            H.nodes[n]['express_fold_change'] = gene_attr_dict[n]['express_fold_change']
            H.nodes[n]['source_of_change'] = gene_attr_dict[n]['source_of_change']
            H.nodes[n]['gene_description'] = gene_attr_dict[n]['gene_description']

            ds = gene_attr_dict.get(n, {}).get('gene_description', [])
            if ds:
                new_name = f"{n} - {','.join(str(desc) for desc in ds)}"
                # H = nx.relabel_nodes(H, {n: new_name})

    return H


def add_gene_sets(top_k_graph, gene_description):
    gene_set_nodes = {}

    for index, row in gene_description.iterrows():
        gene_set = row['gs_desc_short']
        gene = row['human_gene_symbol']
        if gene in top_k_graph:
            if gene_set not in gene_set_nodes:
                top_k_graph.add_node(gene_set, gene_set=True)
                gene_set_nodes[gene_set] = []
            top_k_graph.add_edge(gene_set, gene, edge_type='gene_to_gene_set')
            gene_set_nodes[gene_set].append(gene)

    return top_k_graph


def count_pathway_freq(lst):
    freq_dict = {}
    for elem in lst:
        gene, nums = elem.split(' - ')
        nums_list = nums.split(',')
        if len(nums_list) == 1:
            num = nums_list[0]
            if num in freq_dict:
                freq_dict[num] += 1
            else:
                freq_dict[num] = 1
        else:
            for num in nums_list:
                num = num.strip()
                if num in freq_dict:
                    freq_dict[num] += 1
                else:
                    freq_dict[num] = 1
    df = pd.DataFrame.from_dict(freq_dict, orient='index', columns=['value'])

    return df


def find_common_genes(plot_path, k):
    # Find same genes #
    graph1 = nx.read_gpickle(f'{plot_path}/Gr3_allsamples_{k}e_wgs.gpickle')
    graph2 = nx.read_gpickle(f'{plot_path}/Gr4_allsamples_{k}e_wgs.gpickle')

    # graph1_genes = list(graph1.nodes())
    # graph2_genes = list(graph2.nodes())
    graph1_genes = [n for n, d in graph1.nodes(data=True) if 'gene_set' not in d]
    graph2_genes = [n for n, d in graph2.nodes(data=True) if 'gene_set' not in d]
    common_genes = set(graph1_genes).intersection(graph2_genes)
    graph1_only_genes = set(graph1_genes) - set(graph2_genes)
    graph2_only_genes = set(graph2_genes) - set(graph1_genes)

    # geneset_ct = count_pathway_freq(graph1_genes + graph2_genes)

    g3f = nx.read_gpickle(f'{plot_path}/Gr3_F_{k}e_wgs.gpickle')
    g3m = nx.read_gpickle(f'{plot_path}/Gr3_M_{k}e_wgs.gpickle')
    g4f = nx.read_gpickle(f'{plot_path}/Gr4_F_{k}e_wgs.gpickle')
    g4m = nx.read_gpickle(f'{plot_path}/Gr4_M_{k}e_wgs.gpickle')

    f_genes = [n for n, d in g3f.nodes(data=True) if 'gene_set' not in d] + [n for n, d in g4f.nodes(data=True)
                                                                             if 'gene_set' not in d]
    m_genes = [n for n, d in g3m.nodes(data=True) if 'gene_set' not in d] + [n for n, d in g4m.nodes(data=True)
                                                                             if 'gene_set' not in d]
    # f_genes = list(g3f.nodes()) + list(g4f.nodes())
    # m_genes = list(g3m.nodes()) + list(g4m.nodes())
    fm_common_genes = set(f_genes).intersection(m_genes)
    f_only_genes = set(f_genes) - set(m_genes)
    m_only_genes = set(m_genes) - set(f_genes)

    # fem_geneset_ct = count_pathway_freq(f_genes)
    # mal_geneset_ct = count_pathway_freq(m_genes)
    #
    # cc = fm_common_genes - set(common_genes).intersection(fm_common_genes)

    return fm_common_genes, f_only_genes, m_only_genes


def remove_above_thres(top_k_graph, fold_change, p_val):

    md_gene_pval_dict = dict(zip(fold_change['gene_name'], fold_change['md_adj_pv']))
    ep_gene_pval_dict = dict(zip(fold_change['gene_name'], fold_change['ep_adj_pv']))

    # p_val = 0.05
    # nodes_to_remove = []
    # edges_to_remove = []
    # for node in top_k_graph.nodes():
    #     if (node in md_gene_pval_dict and md_gene_pval_dict[node] >= p_val) \
    #             or (node in ep_gene_pval_dict and ep_gene_pval_dict[node] >= p_val):
    #         in_edges = list(top_k_graph.in_edges(node))
    #         out_edges = list(top_k_graph.out_edges(node))
    #         nodes_to_remove.append(node)
    #         edges_to_remove.extend(in_edges)
    #         edges_to_remove.extend(out_edges)
    # top_k_graph.remove_nodes_from(nodes_to_remove)
    # top_k_graph.remove_edges_from(edges_to_remove)

    # remove only when both nodes are >= 0.05
    # p_val = 0.05
    nodes_to_remove = []
    edges_to_remove = []
    for source, target in top_k_graph.edges():
        if md_gene_pval_dict.get(source, 1.0) < 0.01:
            source_p_val = ep_gene_pval_dict.get(source, 1.0)
            target_p_val = min(md_gene_pval_dict.get(target, 1.0), ep_gene_pval_dict.get(target, 1.0))
        elif md_gene_pval_dict.get(target, 1.0) < 0.01:
            source_p_val = min(md_gene_pval_dict.get(source, 1.0), ep_gene_pval_dict.get(source, 1.0))
            target_p_val = ep_gene_pval_dict.get(target, 1.0)
        else:
            source_p_val = min(md_gene_pval_dict.get(source, 1.0), ep_gene_pval_dict.get(source, 1.0))
            target_p_val = min(md_gene_pval_dict.get(target, 1.0), ep_gene_pval_dict.get(target, 1.0))

        if source_p_val >= p_val and target_p_val >= p_val:
            nodes_to_remove.append(source)
            nodes_to_remove.append(target)
            edges_to_remove.append((source, target))

    top_k_graph.remove_nodes_from(nodes_to_remove)
    top_k_graph.remove_edges_from(edges_to_remove)

    # remove isolated nodes
    isolated_nodes = []
    for node in top_k_graph.nodes():
        in_edges = list(top_k_graph.in_edges(node))
        out_edges = list(top_k_graph.out_edges(node))
        if len(in_edges) == 0 and len(out_edges) == 0:
            isolated_nodes.append(node)
    top_k_graph.remove_nodes_from(isolated_nodes)

    filtered_g = top_k_graph.copy()

    return filtered_g


def prepare_long(data, filtered_genes, split_num):
    data_filtered = data.loc[filtered_genes, :]
    data_control = data_filtered.iloc[:, :split_num]
    data_case = data_filtered.iloc[:, split_num:]
    data_control = data_control.reset_index(drop=True)
    data_case = data_case.reset_index(drop=True)
    data_control_long = pd.melt(data_control.reset_index(),
                                id_vars=['index'],
                                value_vars=data_control.columns,
                                var_name='sample',
                                value_name='data_level')
    data_case_long = pd.melt(data_case.reset_index(),
                             id_vars=['index'],
                             value_vars=data_case.columns,
                             var_name='sample',
                             value_name='data_level')
    data_control_long['condition'] = 'Control'
    data_case_long['condition'] = 'Case'
    data_long = pd.concat([data_control_long, data_case_long])
    data_long['sample_number'] = data_long['sample'].str.split('_').str[-1]
    data_long['condition'] = data_long['condition'].replace({'Control': 'Control', 'Case': 'Case'})

    return data_long


def violin_plot(plot_path, data_path, case_type, gender_type, k):
    gg = nx.read_gpickle(f'{plot_path}/{case_type}_{gender_type}_{k}e.gpickle')
    md_data = pd.read_csv(f'{data_path}/fold_change_new/{case_type}_{gender_type}_md_v2.csv', index_col=0)
    ep_data = pd.read_csv(f'{data_path}/fold_change_new/{case_type}_{gender_type}_ep_v2.csv', index_col=0)

    methyl_genes = []
    express_genes = []
    for node in gg.nodes():
        if 'source_of_change' in gg.nodes[node] and gg.nodes[node]['source_of_change'] == 'methyl':
            methyl_genes.append((node, gg.nodes[node]['methyl_fold_change']))
        elif 'source_of_change' in gg.nodes[node] and gg.nodes[node]['source_of_change'] == 'express':
            express_genes.append((node, gg.nodes[node]['express_fold_change']))
    methyl_genes = sorted(methyl_genes, key=lambda x: x[1], reverse=True)
    express_genes = sorted(express_genes, key=lambda x: x[1], reverse=True)
    top_methyl_genes = methyl_genes[:10]
    md_filtered_genes = [x[0] for x in top_methyl_genes]
    top_express_genes = express_genes[:10]
    ep_filtered_genes = [x[0] for x in top_express_genes]

    # filtered_genes = list([n for n, d in gg.nodes(data=True) if 'gene_set' not in d])

    split_num = 293  # 157 for M, 117 for F, 293 for all

    md_data_long = prepare_long(md_data, md_filtered_genes, split_num)
    ep_data_long = prepare_long(ep_data, ep_filtered_genes, split_num)

    # violin plot
    sns.set_style('whitegrid')
    plt.figure(figsize=(9, 8))
    sns.violinplot(data=md_data_long, x='index', y='data_level', hue='condition', split=True,
                   palette={'Control': 'grey', 'Case': 'blue'})
    plt.title(f'Methylation Data Comparison for {case_type}_{gender_type}', fontsize=16)
    plt.xlabel('Genes')
    plt.xticks(ticks=range(len(md_filtered_genes)), labels=md_filtered_genes, fontsize=9, rotation=45)
    plt.ylabel('Methylation Level')
    plt.savefig(f'MB_project/violin_plots/{case_type}_{gender_type}_methyl.png', dpi=300)

    sns.set_style('whitegrid')
    plt.figure(figsize=(9, 8))
    sns.violinplot(data=ep_data_long, x='index', y='data_level', hue='condition', split=True,
                   palette={'Control': 'grey', 'Case': 'yellow'})
    plt.title(f'Expression Data Comparison for {case_type}_{gender_type}', fontsize=16)
    plt.xlabel('Genes')
    plt.xticks(ticks=range(len(ep_filtered_genes)), labels=ep_filtered_genes, fontsize=9, rotation=45)
    plt.ylabel('Expression Level')
    plt.savefig(f'MB_project/violin_plots/{case_type}_{gender_type}_express.png', dpi=300)


def main():
    # Define paths #
    case_type = "Gr3"  # Gr3 or Gr4
    gender_type = 'allsamples'  # 'M', 'F', or 'allsamples'
    data_path = 'MB_project/data'
    k = 200
    plot_path = 'MB_project/GNN_GAT_GCN/graphs/0401new'
    save_path = f'{data_path}/0327_new'

    # ================================================================================================== #
    attention_path = f'MB_project/GNN_GAT_GCN/save/attention/attention_{case_type}_{gender_type}.npz'
    loss_path = f'{data_path}/{case_type}_{gender_type}_loss_list.csv'

    edge_index, hop_mask, G, loss_list, attention_all = load_data(data_path, attention_path, loss_path)

    weighted_attention = calculate_weighted_attention(edge_index, loss_list, attention_all)
    total_edges = find_2hop_edges(edge_index, weighted_attention, G)
    np.save(f"{save_path}/{case_type}_{gender_type}_total_edges.npy", total_edges)
    # total_edges_saved = np.load(f"{data_path}/{case_type}_{gender_type}_total_edges.npy")
    print('total edges saved!')

    edges_with_attention, result_G = add_attention_to_graph(total_edges, edge_index, weighted_attention)
    nx.write_gpickle(result_G, f'{save_path}/{case_type}_{gender_type}_result_G.gpickle')
    print('result_G saved!')

    # Add node attr #
    result_G = nx.read_gpickle(f'{save_path}/{case_type}_{gender_type}_result_G.gpickle')
    gene_description = pd.read_csv(f'{data_path}/mSigDB_hallmark_human.csv',
                                   usecols=['human_gene_symbol', 'gs_description', 'gs_index', 'gs_desc_short'])
    # fold_change = pd.read_csv(f'{data_path}/{case_type}_{gender_type}_fold_change.csv')
    # fold_change = fold_change.rename(columns={fold_change.columns[0]: 'gene_name'})

    # Generate combined fold change and p value for each group #
    md_fc_pval = pd.read_csv(f'{data_path}/fold_change_new/{case_type}_{gender_type}_md_fc_pval.csv', index_col=0)
    ep_fc_pval = pd.read_csv(f'{data_path}/fold_change_new/{case_type}_{gender_type}_ep_fc_pval.csv', index_col=0)
    # Count above threshold
    num_genes = 4224  # or 15361
    pval_thres = 0.001
    total_md_count = len(md_fc_pval[md_fc_pval['adj_pv'] < pval_thres])
    total_ep_count = len(ep_fc_pval[ep_fc_pval['adj_pv'] < pval_thres])
    print(total_md_count / num_genes, total_ep_count / num_genes)
    ###
    combined_df = pd.concat([md_fc_pval, ep_fc_pval], axis=1)
    combined_df.columns = ['md_fc', 'md_pv', 'md_adj_pv', 'ep_fc', 'ep_pv', 'ep_adj_pv']
    combined_df.to_csv(f'{data_path}/fold_change_new/{case_type}_{gender_type}_combined_fc_pv.csv')

    fold_change = pd.read_csv(f'{data_path}/fold_change_new/{case_type}_{gender_type}_combined_fc_pv.csv')
    fold_change = fold_change.rename(columns={fold_change.columns[0]: 'gene_name'})
    top_k_graph = add_edges_to_graph(result_G, data_path, fold_change, gene_description, k)
    # Count above threshold (top k)
    top_k_genes = [n for n, d in top_k_graph.nodes(data=True) if 'gene_set' not in d]
    md_fc_pval_top_k = md_fc_pval.loc[top_k_genes]
    ep_fc_pval_top_k = ep_fc_pval.loc[top_k_genes]
    top_k_md_count = len(md_fc_pval_top_k[md_fc_pval_top_k['adj_pv'] < 0.05])
    top_k_ep_count = len(ep_fc_pval_top_k[ep_fc_pval_top_k['adj_pv'] < 0.05])

    print(top_k_md_count / len(top_k_genes), top_k_ep_count / len(top_k_genes))
    ###
    filtered_g = remove_above_thres(top_k_graph, fold_change, p_val=0.05)
    filtered_g = add_gene_sets(filtered_g, gene_description)

    # Save for later use #
    nx.write_gpickle(filtered_g, f'{plot_path}/{case_type}_{gender_type}_{k}e.gpickle')
    # ================================================================================================== #
    top_k_graph = nx.read_gpickle(f'{plot_path}/{case_type}_{gender_type}_{k}e_wgs.gpickle')
    # ================================================================================================== #
    fm_common_genes, f_only_genes, m_only_genes, fem_geneset_ct, mal_geneset_ct = find_common_genes(plot_path, k)
