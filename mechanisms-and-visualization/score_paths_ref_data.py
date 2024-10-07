'''
Script to generate paths in KG using embeddings and discovery patterns with scoring/ranking and cosine similarity.
For reference data, each pattern result is not saved separately, but all results are saved in a single file.
Data is subset to results with protein targets available.
Steps:
1. Load KG, embeddings, and discovery patterns (discovery patterns saved from database already)
2. For each discovery pattern, calculate path weight score and path embedding score (embedding summation along path)
3. Rank paths based on path weight score and path embedding score
4. For each discovery pattern, get most similar pathways and processes based on cosine similarity of path embeddings
5. Save results in a file (top 10 results only)

TODO: add error handling and test
'''

import pandas as pd
import numpy as np
from grape import Graph, EmbeddingResult
import pickle
import argparse, logging

use_edge_weight = True
KG_PATH = '../../np-kg/resources/knowledge_graphs/'
KG_FILE_NAME = 'NP-KG_BETWEENNESS-WEIGHTED-PK-CURIE-HASHED_v3.0.0.tsv'
EMBED_PATH = '../embedding-evaluation/embeddings/'
DATA_PATH = '../data/reference-data/'
DIR_PATH = 'mech-results/'
PATTERN_1_PATH = '/home/sanya/npkg-workspace/output_files/neo4j-patterns/discovery-patterns-processed/'
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='mech-scoring-pattern2.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ComplEx', help='model name')
    parser.add_argument('--output_dir', type=str, default='mech-results/', help='Output directory')
    parser.add_argument('--pattern_number', type=int, default=1, help='Discovery pattern number')
    args = parser.parse_args()
    return args

def load_graph():
    npkg = Graph.from_csv(
    node_path=KG_PATH+'nodeTypes_CURIE_v3.0.0.tsv',
    node_list_node_types_column_number=1,
    nodes_column_number=0,
    node_list_separator='\t',
    node_list_header=True,
    edge_path=KG_PATH+KG_FILE_NAME,
    edge_list_separator='\t',
    edge_list_header=True,
    edge_list_edge_types_column_number=1,
    sources_column_number=0,
    destinations_column_number=2,
    weights_column_number=3,
    directed=True,
    verbose=True
    )
    npkg = npkg.remove_disconnected_nodes()
    logging.info('Graph loaded.')
    return npkg

def load_embeddings(model_name):
    embed_file_name = 'npkg_'+model_name+'_embeddings.pickle'
    with open(EMBED_PATH+embed_file_name, 'rb') as filep:
        embedding_dict = pickle.load(filep)
    embedding_result = EmbeddingResult.load(embedding_dict)
    node_embeddings = embedding_result.get_node_embedding_from_index(0)
    edge_type_embeddings = embedding_result.get_edge_type_embedding_from_index(0)
    logging.info('Embeddings loaded.')
    return node_embeddings, edge_type_embeddings

def normalize_edge_weight(edge_weight, min_weight, max_weight):
    return (edge_weight - min_weight) / (max_weight - min_weight)

def get_path_embeddings_pattern1(path, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map):
    source_node = path['n_id']
    drug_id = path['drug_uri']
    adr_id = path['adr_uri']
    prot_node = path['prot_id']
    ##get node curies
    source_node_curie = uri_to_curie_map.get(source_node)
    drug_curie = uri_to_curie_map.get(drug_id)
    prot_node_curie = uri_to_curie_map.get(prot_node)

    path_embeddings = []
    ##get node embeddings
    path_embeddings.append(node_embeddings.loc[source_node_curie].to_numpy().reshape(1, -1))
    path_embeddings.append(node_embeddings.loc[prot_node_curie].to_numpy().reshape(1, -1))
    path_embeddings.append(node_embeddings.loc[drug_curie].to_numpy().reshape(1, -1))
    
    ##get node ids
    source_node_id = npkg.get_node_id_from_node_name(source_node_curie)
    drug_node_id = npkg.get_node_id_from_node_name(drug_curie)
    prot_node_id = npkg.get_node_id_from_node_name(prot_node_curie)
    edge_1 = npkg.get_edge_id_from_node_ids(source_node_id, prot_node_id)
    edge_1_type = npkg.get_edge_type_name_from_edge_id(edge_1)
    path_embeddings.append(edge_type_embeddings.loc[edge_1_type].to_numpy().reshape(1, -1))
    if drug_node_id in npkg[prot_node_id]:
        edge_2 = npkg.get_edge_id_from_node_ids(prot_node_id, drug_node_id)
        edge_2_type = npkg.get_edge_type_name_from_edge_id(edge_2)
        path_embeddings.append(edge_type_embeddings.loc[edge_2_type].to_numpy().reshape(1, -1))
    elif prot_node_id in npkg[drug_node_id]:
        edge_2 = npkg.get_edge_id_from_node_ids(drug_node_id, prot_node_id)
        edge_2_type = npkg.get_edge_type_name_from_edge_id(edge_2)
        path_embeddings.append(edge_type_embeddings.loc[edge_2_type].to_numpy().reshape(1, -1))
    if pd.notna(adr_id):
        adr_curie = uri_to_curie_map.get(adr_id)
        path_embeddings.append(node_embeddings.loc[adr_curie].to_numpy().reshape(1, -1))
        adr_node_id = npkg.get_node_id_from_node_name(adr_curie)
        edge_3 = npkg.get_edge_id_from_node_ids(drug_node_id, adr_node_id)
        edge_3_type = npkg.get_edge_type_name_from_edge_id(edge_3)
        path_embeddings.append(edge_type_embeddings.loc[edge_3_type].to_numpy().reshape(1, -1))
    return path_embeddings

def get_path_embeddings_pattern2(path, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map):
    source_node = path['n_id']
    drug_id = path['drug_uri']
    adr_id = path['adr_uri']
    prot_node = path['prot_id']
    process_node = path['process_id']
    ##get node curies
    source_node_curie = uri_to_curie_map.get(source_node)
    drug_curie = uri_to_curie_map.get(drug_id)
    prot_node_curie = uri_to_curie_map.get(prot_node)
    process_node_curie = uri_to_curie_map.get(process_node)

    path_embeddings = []
    ##get node embeddings
    path_embeddings.append(node_embeddings.loc[source_node_curie].to_numpy().reshape(1, -1))
    path_embeddings.append(node_embeddings.loc[prot_node_curie].to_numpy().reshape(1, -1))
    path_embeddings.append(node_embeddings.loc[drug_curie].to_numpy().reshape(1, -1))
    path_embeddings.append(node_embeddings.loc[process_node_curie].to_numpy().reshape(1, -1))
    
    ##get node ids
    source_node_id = npkg.get_node_id_from_node_name(source_node_curie)
    drug_node_id = npkg.get_node_id_from_node_name(drug_curie)
    prot_node_id = npkg.get_node_id_from_node_name(prot_node_curie)
    process_node_id = npkg.get_node_id_from_node_name(process_node_curie)
    
    ##different edges in pattern2
    #edge 1: NP -> protein
    edge_1 = npkg.get_edge_id_from_node_ids(source_node_id, prot_node_id)
    edge_1_type = npkg.get_edge_type_name_from_edge_id(edge_1)
    path_embeddings.append(edge_type_embeddings.loc[edge_1_type].to_numpy().reshape(1, -1))
    
    ##edge 2: process -> protein
    edge_2 = npkg.get_edge_id_from_node_ids(process_node_id, prot_node_id)
    edge_2_type = npkg.get_edge_type_name_from_edge_id(edge_2)
    path_embeddings.append(edge_type_embeddings.loc[edge_2_type].to_numpy().reshape(1, -1))

    ##edge 3: drug -> process
    edge_3 = npkg.get_edge_id_from_node_ids(drug_node_id, process_node_id)
    edge_3_type = npkg.get_edge_type_name_from_edge_id(edge_3)
    path_embeddings.append(edge_type_embeddings.loc[edge_3_type].to_numpy().reshape(1, -1))

    ##edge 4: drug -> ADR
    if pd.notna(adr_id):
        adr_curie = uri_to_curie_map.get(adr_id)
        path_embeddings.append(node_embeddings.loc[adr_curie].to_numpy().reshape(1, -1))
        adr_node_id = npkg.get_node_id_from_node_name(adr_curie)
        edge_4 = npkg.get_edge_id_from_node_ids(drug_node_id, adr_node_id)
        edge_4_type = npkg.get_edge_type_name_from_edge_id(edge_4)
        path_embeddings.append(edge_type_embeddings.loc[edge_4_type].to_numpy().reshape(1, -1))
    return path_embeddings

def get_path_embeddings_score(path_embeddings):
    path_embeddings_new = []
    for p_embed in path_embeddings:
        norm = np.linalg.norm(p_embed, ord=2)
        if norm != 0:
            p_embed = p_embed / norm
        path_embeddings_new.append(p_embed)
    path_sum = np.sum(path_embeddings_new, axis=0)
    path_norm = np.linalg.norm(path_sum, ord=2)
    return path_norm

def score_results_pattern1(pattern_results, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map):
    ##if weights sum column exists, path weight score = weights_sum or r1_weight+r2_weight+r3_weight
    #if 'weights_sum' in pattern_results.columns:
    #    pattern_results['path_weight_score'] = pattern_results['weights_sum']
    pattern_results['r3_weight'] = pattern_results['r3_weight'].fillna(0)
    pattern_results['path_weight_score'] = pattern_results['r1_weight'] + pattern_results['r2_weight'] + pattern_results['r3_weight']
    pattern_results['path_embedding_score'] = pd.NA
    for i in range(len(pattern_results.index)):
        path = pattern_results.iloc[i]
        path_embeddings = get_path_embeddings_pattern1(path, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map)
        path_score = get_path_embeddings_score(path_embeddings)
        pattern_results.at[i, 'path_embedding_score'] = path_score
    return pattern_results

def score_results_pattern2(pattern_results, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map):
    if 'weights_sum' in pattern_results.columns:
        pattern_results['path_weight_score'] = pattern_results['weights_sum']
    pattern_results['path_embedding_score'] = pd.NA
    for i in range(len(pattern_results.index)):
        path = pattern_results.iloc[i]
        path_embeddings = get_path_embeddings_pattern2(path, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map)
        path_score = get_path_embeddings_score(path_embeddings)
        pattern_results.at[i, 'path_embedding_score'] = path_score
    return pattern_results

def add_metadata_to_results_pattern1(results, edge_metadata):
    results['edge1_metadata'] = pd.NA
    results['edge2_metadata'] = pd.NA
    for i in range(len(results.index)):
        n_id = results.at[i, 'n_id']
        prot_id = results.at[i, 'prot_id']
        drug_uri = results.at[i, 'drug_uri']
        edge1_data = edge_metadata[(edge_metadata['subject_uri'] == n_id) & (edge_metadata['object_uri'] == prot_id)]
        edge2_data = edge_metadata[(edge_metadata['subject_uri'] == drug_uri) & (edge_metadata['object_uri'] == prot_id)]
        if not edge1_data.empty:
            ##convert to dictionary with keys: pmid, sentence, section, pubtype, belief, year
            edge1_dict = edge1_data.to_dict(orient='records')[0]
            edge1_dict = {k: v for k, v in edge1_dict.items() if k in ['pmid', 'source_section', 'pubtype', 'belief', 'year']}
            results.at[i, 'edge1_metadata'] = edge1_dict
        if not edge2_data.empty:
            edge2_dict = edge2_data.to_dict(orient='records')[0]
            edge2_dict = {k: v for k, v in edge2_dict.items() if k in ['pmid', 'source_section', 'pubtype', 'belief', 'year']}
            results.at[i, 'edge2_metadata'] = edge2_dict
    return results

def main(args):
    args = parse_arguments()

    npkg = load_graph()
    ##load node labels
    #nodeLabelsdf = pd.read_csv(KG_PATH+'nodeLabels_CURIE_v3.0.0.tsv', sep='\t')
    #nodeLabels = nodeLabelsdf.set_index('source')['entity_label'].to_dict()
    uri_df = pd.read_csv(KG_PATH+'uri_to_curie_map.tsv', sep='\t', header=None)
    uri_to_curie_map = uri_df.set_index(0)[1].to_dict()
    ##load embeddings
    node_embeddings, edge_type_embeddings = load_embeddings(args.model)

    ##load discovery patterns
    pattern_num = args.pattern_number
    logging.info(f'Loading discovery patterns for pattern number: {pattern_num}')
    if pattern_num == 1:

        ##get discovery patterns from files
        results_protein_positive = 'pattern1_results_reference_data_protein_positive_20240924.tsv'
        results_no_protein_positive = 'pattern1_results_reference_data_no_protein_positive_20240924.tsv'
        results_negative = 'pattern1_results_reference_data_negative_20240924.tsv'
        edge_metadata_filename = 'edge_metadata_pattern1_reference_data_20240925.tsv'
        discovery_patterns_protein_positive = pd.read_csv(PATTERN_1_PATH+results_protein_positive, sep='\t')
        discovery_patterns_no_protein_positive = pd.read_csv(PATTERN_1_PATH+results_no_protein_positive, sep='\t')
        discovery_patterns_negative = pd.read_csv(PATTERN_1_PATH+results_negative, sep='\t')

        ###add metadata to results
        edge_metadata = pd.read_csv(PATTERN_1_PATH+edge_metadata_filename, sep='\t')
        discovery_patterns_protein_positive = add_metadata_to_results_pattern1(discovery_patterns_protein_positive, edge_metadata)
        discovery_patterns_no_protein_positive = add_metadata_to_results_pattern1(discovery_patterns_no_protein_positive, edge_metadata)
        discovery_patterns_negative = add_metadata_to_results_pattern1(discovery_patterns_negative, edge_metadata)

        ##get scores for each path in discovery patterns
        logging.info('Scoring discovery patterns.')
        discovery_patterns_protein_positive_scored = score_results_pattern1(discovery_patterns_protein_positive, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map)
        discovery_patterns_no_protein_positive_scored = score_results_pattern1(discovery_patterns_no_protein_positive, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map)
        discovery_patterns_negative_scored = score_results_pattern1(discovery_patterns_negative, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map)
        
        ##rank paths based on path weight score and path embedding score
        ##save results
        logging.info('Results saved in: {DIR_PATH}')
        discovery_patterns_protein_positive_scored.to_csv(DIR_PATH+'pattern1_results_protein_positive_scored.tsv', sep='\t', index=False)
        discovery_patterns_no_protein_positive_scored.to_csv(DIR_PATH+'pattern1_results_no_protein_positive_scored.tsv', sep='\t', index=False)
        discovery_patterns_negative_scored.to_csv(DIR_PATH+'pattern1_results_negative_scored.tsv', sep='\t', index=False)

    ##pattern 2: NP -> process -> protein -> drug -> ADR: TODO
    elif pattern_num == 2:
        ##load from database
        results_protein_positive = 'pattern2_results_reference_data_protein_positive_20241005.tsv'
        results_negative = 'pattern2_results_reference_data_negative_20241005.tsv'
        discovery_patterns_protein_positive = pd.read_csv(PATTERN_1_PATH+results_protein_positive, sep='\t')
        discovery_patterns_negative = pd.read_csv(PATTERN_1_PATH+results_negative, sep='\t')

        ##get scores for each path in discovery patterns
        logging.info('Scoring discovery patterns.')
        discovery_patterns_protein_positive_scored = score_results_pattern2(discovery_patterns_protein_positive, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map)
        discovery_patterns_negative_scored = score_results_pattern2(discovery_patterns_negative, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map)
        
        ##rank paths based on path weight score and path embedding score
        ##save results
        logging.info('Results saved in: {DIR_PATH}')
        discovery_patterns_protein_positive_scored.to_csv(DIR_PATH+'pattern2_results_protein_positive_scored.tsv', sep='\t', index=False)
        discovery_patterns_negative_scored.to_csv(DIR_PATH+'pattern2_results_negative_scored.tsv', sep='\t', index=False)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)