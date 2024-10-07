'''
Script to generate paths in KG using embeddings and discovery patterns with scoring/ranking and cosine similarity.
1. Load KG, embeddings, and discovery patterns
2. For each discovery pattern, calculate path weight score and path embedding score (embedding summation along path)
3. Rank paths based on path weight score and path embedding score
4. For each discovery pattern, get most similar pathways and processes based on cosine similarity of path embeddings
5. Save results in a file
'''

import pandas as pd
import numpy as np
from grape import Graph, EmbeddingResult
import pickle, os
import argparse, logging
from get_discovery_pattern import get_discovery_patterns_1_pv

use_edge_weight = True
KG_PATH = '../../np-kg/resources/knowledge_graphs/'
KG_FILE_NAME = 'NP-KG_BETWEENNESS-WEIGHTED-PK-CURIE-HASHED_v3.0.0.tsv'
EMBED_PATH = '../embedding-evaluation/embeddings/'
DATA_PATH = '../data/pv-data/'
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='mech-scoring-pv.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ComplEx', help='model name')
    parser.add_argument('--data', type=str, default='pv', help='data type (pv or reference)')
    parser.add_argument('--output_dir', type=str, default='../data/pv-data/', help='Output directory')
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

def add_metadata_to_results_pattern1(results, edge_metadata):
    results['edge1_metadata'] = pd.NA
    results['edge2_metadata'] = pd.NA
    results['path_metadata_score'] = 0
    for i in range(len(results.index)):
        n_id = results.at[i, 'n_id']
        prot_id = results.at[i, 'prot_id']
        drug_uri = results.at[i, 'drug_uri']
        edge1_data = edge_metadata[(edge_metadata['subject_uri'] == n_id) & (edge_metadata['object_uri'] == prot_id)]
        edge2_data = edge_metadata[(edge_metadata['subject_uri'] == drug_uri) & (edge_metadata['object_uri'] == prot_id)]
        if not edge1_data.empty:
            ##convert to dictionary with keys: pmid, sentence, section, pubtype, belief, year
            edge1_dict = edge1_data.to_dict(orient='records')[0]
            ##if pmid exists, score=1, else score=0
            if 'pmid' in edge1_dict and edge1_dict['pmid'] != 'None':
                results.at[i, 'path_metadata_score'] +=1
            edge1_dict = {k: v for k, v in edge1_dict.items() if k in ['pmid', 'source_section', 'pubtype', 'belief', 'year']}
            results.at[i, 'edge1_metadata'] = edge1_dict
        if not edge2_data.empty:
            edge2_dict = edge2_data.to_dict(orient='records')[0]
            if 'pmid' in edge2_dict and edge2_dict['pmid'] != 'None':
                results.at[i, 'path_metadata_score'] += 1
            edge2_dict = {k: v for k, v in edge2_dict.items() if k in ['pmid', 'source_section', 'pubtype', 'belief', 'year']}
            results.at[i, 'edge2_metadata'] = edge2_dict
    return results

def normalize_edge_weight(edge_weight, min_weight, max_weight):
    return (edge_weight - min_weight) / (max_weight - min_weight)

def get_path_embeddings(path, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map):
    source_node = path['n_id']
    drug_id = path['drug_uri']
    adr_id = path['adr_uri']
    prot_node = path['prot_id']
    ##get node curies
    source_node_curie = uri_to_curie_map.get(source_node)
    drug_curie = uri_to_curie_map.get(drug_id)
    adr_curie = uri_to_curie_map.get(adr_id)
    prot_node_curie = uri_to_curie_map.get(prot_node)

    path_embeddings = []
    ##get node embeddings
    path_embeddings.append(node_embeddings.loc[source_node_curie].to_numpy().reshape(1, -1))
    path_embeddings.append(node_embeddings.loc[prot_node_curie].to_numpy().reshape(1, -1))
    path_embeddings.append(node_embeddings.loc[drug_curie].to_numpy().reshape(1, -1))
    path_embeddings.append(node_embeddings.loc[adr_curie].to_numpy().reshape(1, -1))
    
    ##get node ids
    source_node_id = npkg.get_node_id_from_node_name(source_node_curie)
    drug_node_id = npkg.get_node_id_from_node_name(drug_curie)
    adr_node_id = npkg.get_node_id_from_node_name(adr_curie)
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
    edge_3 = npkg.get_edge_id_from_node_ids(drug_node_id, adr_node_id)
    edge_3_type = npkg.get_edge_type_name_from_edge_id(edge_3)
    path_embeddings.append(edge_type_embeddings.loc[edge_3_type].to_numpy().reshape(1, -1))
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

def score_results(pattern_results, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map):
    ##if weights sum column exists, path weight score = weights_sum or r1_weight+r2_weight+r3_weight
    if 'weights_sum' in pattern_results.columns:
        pattern_results['path_weight_score'] = pattern_results['weights_sum']
    else:
        pattern_results['path_weight_score'] = pattern_results['r1_weight'] + pattern_results['r2_weight'] + pattern_results['r3_weight']
    pattern_results['path_embedding_score'] = pd.NA
    for i in range(len(pattern_results.index)):
        path = pattern_results.iloc[i]
        path_embeddings = get_path_embeddings(path, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map)
        path_score = get_path_embeddings_score(path_embeddings)
        pattern_results.at[i, 'path_embedding_score'] = path_score
    return pattern_results

def main(args):
    args = parse_arguments()
    DATA_PATH = args.output_dir
    npkg = load_graph()
    ##load node labels
    #nodeLabelsdf = pd.read_csv(KG_PATH+'nodeLabels_CURIE_v3.0.0.tsv', sep='\t')
    #nodeLabels = nodeLabelsdf.set_index('source')['entity_label'].to_dict()
    uri_df = pd.read_csv(KG_PATH+'uri_to_curie_map.tsv', sep='\t', header=None)
    uri_to_curie_map = uri_df.set_index(0)[1].to_dict()
    ##load embeddings
    node_embeddings, edge_type_embeddings = load_embeddings(args.model)
    pattern_num = args.pattern_number
    '''
    ##Option 1: input NP, Drug, ADR names and URIs
    np_name = 'ginger'
    drug_name = 'morphine'
    adr_name = 'skin_reaction'
    np_uri = 'http://napdi.org/napdi_srs_imports:zingiber_officinale'
    drug_uri = 'http://purl.obolibrary.org/obo/CHEBI_17303'
    adr_uri = ['http://purl.obolibrary.org/obo/MONDO_0005093']
    np_id = -7012387
    drug_id = 1110410
    adr_meddra_id = [10040914]
    logging.info(f'Generating paths for NP: {np_name}, Drug: {drug_name}, ADR: {adr_name}')

    np_curie = uri_to_curie_map.get(np_uri)
    drug_curie = uri_to_curie_map.get(drug_uri)
    adr_curie = [uri_to_curie_map.get(uri) for uri in adr_uri]
    logging.info(f'Node curies: {np_curie}, {drug_curie}, {adr_curie}')

    ##load discovery patterns
    logging.info(f'Loading discovery patterns for pattern number: {pattern_num}')
    if pattern_num == 1:
        ###pattern 1: NP -> protein -> drug -> ADR
        filename = DATA_PATH+np_name+'_'+drug_name+'_'+adr_name+'_discovery_pattern1.tsv'
        if os.path.exists(filename):
            discovery_patterns = pd.read_csv(filename, sep='\t')
        else:
            discovery_patterns = get_discovery_patterns_1_pv(np_id, drug_id, adr_meddra_id)
        
        if discovery_patterns is None:
            logging.info('No discovery patterns found.')
            exit()
        else:
            logging.info('Discovery patterns loaded with metadata: {discovery_patterns.shape[0]}')
            discovery_patterns.to_csv(DATA_PATH+np_name+'_'+drug_name+'_'+adr_name+'_discovery_pattern1.tsv', sep='\t', index=False)
            ##add curies
            discovery_patterns['n_curie'] = np_curie
            discovery_patterns['drug_curie'] = drug_curie
            discovery_patterns['adr_curie'] = '|'.join(adr_curie)

        ##get scores for each path in discovery patterns
        logging.info('Scoring discovery patterns.')
        discovery_patterns_scored = score_results(discovery_patterns, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map)
        
        ##rank paths based on path weight score and path embedding score
        ##save results
        discovery_patterns_scored.to_csv(DATA_PATH+np_name+'_'+drug_name+'_'+adr_name+'_discovery_pattern1_scored.tsv', sep='\t', index=False)
        logging.info('Results saved in: {DATA_PATH}{np_name}_{drug_name}_{adr_name}_discovery_pattern1_scored.tsv}')

    ##pattern 2: NP -> process -> protein -> drug -> ADR: TODO
    elif pattern_num == 2:
        pass
    '''
    ##Option 2: load data with patterns and metadata (already generated in npkg-workspace/pv-signals-related/)
    if pattern_num == 1:
        logging.info(f'Loading discovery patterns for pattern number: {pattern_num}')
        discovery_patterns = 'pv-signals-subset-pattern1-results_20241003.tsv'
        edge_metadata_filename = 'pv-signals-subset-edge-metadata_20241003.tsv'
        discovery_patterns = pd.read_csv(DATA_PATH+discovery_patterns, sep='\t')
        edge_metadata = pd.read_csv(DATA_PATH+edge_metadata_filename, sep='\t')

        logging.info(f'Discovery patterns loaded with metadata: {discovery_patterns.shape[0]}')
        logging.info(f'Edge metadata loaded: {edge_metadata.shape[0]}')
        ##add metadata to results
        discovery_patterns = add_metadata_to_results_pattern1(discovery_patterns, edge_metadata)

        ##get scores for each path in discovery patterns
        logging.info('Scoring discovery patterns.')
        discovery_patterns_scored = score_results(discovery_patterns, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map)

        ##rank paths based on path weight score and path embedding score
        ##save results
        discovery_patterns_scored.to_csv(DATA_PATH+'pv-signals-subset-pattern1-results_20241003_scored.tsv', sep='\t', index=False)
        logging.info(f'Results saved in: {DATA_PATH}pv-signals-subset-pattern1-results_20241003_scored.tsv')

if __name__ == "__main__":
    args = parse_arguments()
    main(args)