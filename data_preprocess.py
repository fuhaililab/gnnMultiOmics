"""
Zitian Tang
Preprocess the raw data in order to be fed into the model.
"""

import pandas as pd
import numpy as np
import json


def process_columns(md_data, ep_data, all_ids, gender=None):
    """Step 1: subsitute colnames (i.e. sample names) in both methylation data and expression
    data with sample_1, sample_2, ... etc.
    Args:
        md_data: methylation data.
        ep_data: expression data.
        all_ids: csv file contains sample IDs from both methylation and expression data.
        gender: (optional) If specified, returns only the data corresponding to the given gender.
    Returns:
        If gender is None: (md_data, ep_data)
        If gender is 'M': (md_male_data, ep_male_data)
        If gender is 'F': (md_female_data, ep_female_data)
    """

    if gender == 'M':
        all_ids = all_ids[all_ids['Gender'] == 'M']
    elif gender == 'F':
        all_ids = all_ids[all_ids['Gender'] == 'F']

    colnames_dict = {methyl_colname: 'sample' + str(i+1)
                     for i, methyl_colname in enumerate(all_ids['MAGIC Cohort Sample Name'])}
    md_data.rename(columns=colnames_dict, inplace=True)
    colnames_dict.update({express_colname: 'sample' + str(i+1)
                          for i, express_colname in enumerate(all_ids['85218 GSM IDs'])})
    ep_data.rename(columns=colnames_dict, inplace=True)

    md_data = md_data[md_data.columns.drop(list(md_data.filter(regex='MB')))]
    ep_data = ep_data[ep_data.columns.drop(list(ep_data.filter(regex='MB')))]

    # save for later use
    # md_data.to_csv('md_modified.csv', index=False)
    # ep_data.to_csv('ep_modified.csv', index=False)

    return all_ids, md_data, ep_data


def process_rows(md_data, ep_data, md_probe_to_gene, ep_probe_to_gene):
    """Step 2: substitute probe ids with corresponding gene.
    Args:
        md_data: methylation data (already processed columns).
        ep_data: expression data (already processed columns).
        md_probe_to_gene: csv contains methylation data probe ID -> gene info.
        ep_probe_to_gene: csv contains expression data probe ID -> gene info.
    Returns:
        md_data: methylation data with each row representing a gene.
        ep_data: expression data with each row representing a gene.
    """

    md_probe_to_gene['UCSC_RefGene_Name'] = md_probe_to_gene['UCSC_RefGene_Name'].str.split(';').str[0]
    probe_to_gene_dict = dict(zip(md_probe_to_gene['Probe_ID'], md_probe_to_gene['UCSC_RefGene_Name']))
    md_data.rename(index=probe_to_gene_dict, inplace=True)

    probe_to_gene_dict = dict(zip(ep_probe_to_gene['Probe.Set.Name'].str.replace('_', '-'),
                                  ep_probe_to_gene['HGNC_symbol_from_ensemblv77']))
    ep_data.rename(index=probe_to_gene_dict, inplace=True)

    return md_data, ep_data


def combine_md_ep_data(md_data, ep_data, gene_list, label=False):
    """Step 3: concatenate data in md_data and ep_data into one 3D np array and save npz.
    Args:
        md_data: methylation data (rows and columns processed).
        ep_data: expression data (rows and columns processed).
        gene_list: list of genes used to filter Gr3 data.
        label: only equals True when subtype is Gr3, otherwise False.
    Returns:
        data: 3d numpy array with dimension (num_sample, num_gene=15361, num_feature=2)
        md_mean: filtered md_data (save in case later use).
        ep_mean: filtered ep_data (save in case later use).
    """

    if label:  # process Gr3 using different method
        # Testing code if column won't match:
        mdcols = set(md_data.columns)
        epcols = set(ep_data.columns)
        print((mdcols - epcols).pop())
        md_data = md_data.drop(str((mdcols - epcols).pop()), axis=1)
        # assert(md_mean.index == ep_mean.index).all(), 'Rows in the two dataframes do not match'
        md_filtered = md_data.loc[md_data.index.isin(gene_list)]
        ep_filtered = ep_data.loc[ep_data.index.isin(gene_list)]
        md_mean = md_filtered.groupby(md_filtered.index).mean()
        ep_mean = ep_filtered.groupby(ep_filtered.index).mean()
        # add missing genes in md_mean
        missing_idx = set(gene_list) - set(md_mean.index)
        missing_md = pd.DataFrame(0, index=missing_idx, columns=md_mean.columns)
        added_md = pd.concat([md_mean, missing_md])
        added_md = added_md.reindex(ep_mean.index)
        data = np.dstack([np.transpose(added_md.values), np.transpose(ep_mean.values)])
        return data, added_md, ep_mean

    common_genes = ep_data.index.intersection(md_data.index).dropna()
    md_filtered = md_data.loc[common_genes]
    ep_filtered = ep_data.loc[common_genes]

    md_mean = md_filtered.groupby(md_filtered.index).mean()
    ep_mean = ep_filtered.groupby(ep_filtered.index).mean()

    data = np.dstack([np.transpose(md_mean.values), np.transpose(ep_mean.values)])
    # WNT - (70, 15361, 2)
    # SHH - (223, 15361, 2)
    # Gr3 - (143, 15361, 2)
    # Gr4 - (326, 15361, 2)
    return data, md_mean, ep_mean


def compute_foldchange(subtype_name, gender, save_path):
    # if len(gender) > 1:
    #     wnt_md = pd.read_csv(f'{save_path}/wnt_md_filtered.csv', index_col=0)
    #     wnt_ep = pd.read_csv(f'{save_path}/wnt_ep_filtered.csv', index_col=0)
    #     shh_md = pd.read_csv(f'{save_path}/shh_md_filtered.csv', index_col=0)
    #     shh_ep = pd.read_csv(f'{save_path}/shh_ep_filtered.csv', index_col=0)
    #     case_md = pd.read_csv(f'{save_path}/{subtype_name}_md_filtered.csv', index_col=0)
    #     case_ep = pd.read_csv(f'{save_path}/{subtype_name}_ep_filtered.csv', index_col=0)
    # else:
    subtype_name = "gr3"  # one of wnt, shh, gr3, gr4
    gender = 'M'  # one of 'F', 'M', allsamples
    save_path = "MB_project/data/preprocessed_data/0321new"
    filtered_genes = np.load(f'{save_path}/filtered_genes.npy', allow_pickle=True).tolist()

    wnt_md = pd.read_csv(f'{save_path}/wnt_{gender}_md_filtered.csv', index_col=0).loc[filtered_genes]
    wnt_ep = pd.read_csv(f'{save_path}/wnt_{gender}_ep_filtered.csv', index_col=0).loc[filtered_genes]
    shh_md = pd.read_csv(f'{save_path}/shh_{gender}_md_filtered.csv', index_col=0).loc[filtered_genes]
    shh_ep = pd.read_csv(f'{save_path}/shh_{gender}_ep_filtered.csv', index_col=0).loc[filtered_genes]
    case_md = pd.read_csv(f'{save_path}/{subtype_name}_{gender}_md_filtered.csv', index_col=0).loc[filtered_genes]
    case_ep = pd.read_csv(f'{save_path}/{subtype_name}_{gender}_ep_filtered.csv', index_col=0).loc[filtered_genes]

    control_md = pd.concat([wnt_md, shh_md], axis=1)
    control_ep = pd.concat([wnt_ep, shh_ep], axis=1)

    md_saved = pd.concat([control_md, case_md], axis=1)
    ep_saved = pd.concat([control_ep, case_ep], axis=1)
    md_saved.to_csv(f'MB_project/data/fold_change_new/{subtype_name}_{gender}_md_v2.csv')
    ep_saved.to_csv(f'MB_project/data/fold_change_new/{subtype_name}_{gender}_ep_v2.csv')

    # compute mean
    control_md_mean = control_md.mean(axis=1)
    control_ep_mean = control_ep.mean(axis=1)
    case_md_mean = case_md.mean(axis=1)
    case_ep_mean = case_ep.mean(axis=1)

    # compute fold change and log fc
    md_fc = case_md_mean / control_md_mean
    ep_fc = case_ep_mean / control_ep_mean
    md_fc[md_fc == 0] = 1e-10
    log_md_fc = np.log2(md_fc)
    log_ep_fc = np.log2(ep_fc)
    # t-test (if necessary)

    result = pd.DataFrame({'methyl_fold_change': md_fc, 'express_fold_change': ep_fc,
                           'methyl_fold_change_log': log_md_fc, 'express_fold_change_log': log_ep_fc},
                          index=md_fc.index)
    # see which one is source of change:
    result.loc[result['methyl_fold_change'] > result['express_fold_change'], 'source_of_change'] = 'methyl'
    result.loc[result['methyl_fold_change'] < result['express_fold_change'], 'source_of_change'] = 'express'
    result.loc[result['methyl_fold_change'] == result['express_fold_change'], 'source_of_change'] = 'equal'

    # save for later use
    result.to_csv(f'MB_project/data/{subtype_name}_{gender}_fold_change.csv')

    return result


def convert_txt_to_csv():
    import csv
    filename = 'GSE37384_dChip'
    outname = 'GSE37384'
    input_file = f'data/raw_data/{filename}.txt'
    output_file = f'data/raw_data/{outname}.csv'

    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.reader(f_in, delimiter='\t')
        writer = csv.writer(f_out)
        for row in reader:
            writer.writerow(row)

    # for SNP data, remove columns other than MB dataset
    snp_data = pd.read_csv(output_file, index_col=0)
    columns_to_keep = [c for c in snp_data.columns if 'MB' in c and 'call' not in c]
    filtered_snp = snp_data[columns_to_keep]
    filtered_snp.to_csv(f'data/raw_data/{outname}_filtered.csv')


def separate_subtypes(data_path, new_data_path):
    md_total = pd.read_csv(f'{new_data_path}/GSE85212.csv')
    ep_total = pd.read_csv(f'{new_data_path}/GSE85217.csv')
    wnt_cols = list(pd.read_csv(f'{data_path}/Normalized Methylation Data Before T-Test/WNTMethyl.csv', index_col=0).columns)
    shh_cols = list(pd.read_csv(f'{data_path}/Normalized Methylation Data Before T-Test/SHHMethyl.csv', index_col=0).columns)
    g3_cols = list(pd.read_csv(f'{data_path}/Normalized Methylation Data Before T-Test/Gr3Methyl.csv', index_col=0).columns)
    g4_cols = list(pd.read_csv(f'{data_path}/Normalized Methylation Data Before T-Test/Gr4Methyl.csv', index_col=0).columns)

    wnt_md = md_total.loc[:, ['Unnamed: 0'] + wnt_cols]
    wnt_md.to_csv(f'{new_data_path}/wnt_md.csv')
    shh_md = md_total.loc[:, ['Unnamed: 0'] + shh_cols]
    shh_md.to_csv(f'{new_data_path}/shh_md.csv')
    gr3_md = md_total.loc[:, ['Unnamed: 0'] + g3_cols]
    gr3_md.to_csv(f'{new_data_path}/gr3_md.csv')
    gr4_md = md_total.loc[:, ['Unnamed: 0'] + g4_cols]
    gr4_md.to_csv(f'{new_data_path}/gr4_md.csv')

    wnt_ep = ep_total.loc[:, ['Unnamed: 0'] + wnt_cols]
    wnt_ep.to_csv(f'{new_data_path}/wnt_ep.csv')
    shh_ep = ep_total.loc[:, ['Unnamed: 0'] + shh_cols]
    shh_ep.to_csv(f'{new_data_path}/shh_ep.csv')
    gr3_ep = ep_total.loc[:, ['Unnamed: 0'] + g3_cols]
    gr3_ep.to_csv(f'{new_data_path}/gr3_ep.csv')
    gr4_ep = ep_total.loc[:, ['Unnamed: 0'] + g4_cols]
    gr4_ep.to_csv(f'{new_data_path}/gr4_ep.csv')


def main():
    # Define data paths #
    subtype_name = "gr4"  # one of wnt, shh, gr3, gr4
    gender = 'allsamples'  # one of 'F', 'M', allsamples
    data_path = "MB_project/data/normalized_data"
    new_data_path = "MB_project/data/normalized_data_new"
    save_path = "MB_project/data/preprocessed_data/0321new"
    # if gender=allsamples
    # save_path = "MB_project/data/preprocessed_data/0223new"

    md_data = pd.read_csv(f'{new_data_path}/{subtype_name}_md.csv', index_col=1)
    md_data = md_data.drop(md_data.columns[0], axis=1)
    ep_data = pd.read_csv(f'{new_data_path}/{subtype_name}_ep.csv', index_col=1)
    ep_data = ep_data.drop(ep_data.columns[0], axis=1)

    all_ids = pd.read_csv(f'{data_path}/Patient_Sample_Name_to_GSM_ID.csv')
    clinical_data = pd.read_csv(f'{data_path}/PatientIDInfo_and_ClinicalData.csv')
    md_probe_to_gene = pd.read_csv(f'{data_path}/Gene_Annotation_GSE85212_MAGIC_MethylData.csv')
    ep_probe_to_gene = pd.read_csv(f'{data_path}/Gene_Annotation_GSE85217_MAGIC_ExpressData.csv')

    # merge gender info into all_ids
    all_ids = pd.merge(all_ids, clinical_data[['Study_ID', 'Gender']], left_on='MAGIC Cohort Sample Name',
                       right_on='Study_ID', how='left')

    with open(f'{data_path}/gene_list.json', 'r') as file:
        gene_list = json.load(file)

    # Data preprocessing #
    all_ids, md_data, ep_data = process_columns(md_data, ep_data, all_ids, gender=gender)
    print(f"column processing complete for {subtype_name}!")
    md_data, ep_data = process_rows(md_data, ep_data, md_probe_to_gene, ep_probe_to_gene)
    print(f"row processing complete for {subtype_name}!")
    # if subtype_name == "Gr3":
    #     data, md_filtered, ep_filtered = combine_md_ep_data(md_data, ep_data, gene_list, label=True)
    # else:
    data, md_filtered, ep_filtered = combine_md_ep_data(md_data, ep_data, gene_list)
    print(f"data combination complete for {subtype_name}!")

    # Save for later use #
    md_filtered.to_csv(f'{save_path}/{subtype_name}_{gender}_md_filtered.csv')
    ep_filtered.to_csv(f'{save_path}/{subtype_name}_{gender}_ep_filtered.csv')
    np.savez_compressed(f'{save_path}/{subtype_name}_{gender}_preprocessed_data.npz', data=data)
    # Save gene list to use on Gr3 processing (saved using Gr4)
    # gene_list = list(ep_filtered.index)
    # with open(f'{data_path}/gene_list.json', 'w') as file:
    #     json.dump(gene_list, file)

    # Concatenate WNT & SHH together to generate control set #
    wnt_data = np.load(f'{save_path}/wnt_{gender}_preprocessed_data.npz')['data']
    shh_data = np.load(f'{save_path}/shh_{gender}_preprocessed_data.npz')['data']
    control_set = np.concatenate([wnt_data, shh_data], axis=0)
    np.savez_compressed(f'{save_path}/control_{gender}_wnt_shh.npz', data=control_set)


if __name__ == "__main__":
    main()


