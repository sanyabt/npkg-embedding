'''
Script to explore nodes that are similar to other nodes in the path and nodes that are similar to the path itself using cosine similarity of embeddings.
'''

import pandas as pd
import numpy as np
from grape import Graph, EmbeddingResult
import pickle
from generate_and_score_paths import get_path_embeddings

KG_PATH = '../../np-kg/resources/knowledge_graphs/'
KG_FILE_NAME = 'NP-KG_BETWEENNESS-WEIGHTED-PK-CURIE-HASHED_v3.0.0.tsv'
EMBED_PATH = '../embedding-evaluation/embeddings/'
DATA_PATH = '../data/'
DIR_PATH = 'mech-results/'

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
    return npkg

def load_embeddings(model_name):
    embed_file_name = 'npkg_'+model_name+'_embeddings.pickle'
    with open(EMBED_PATH+embed_file_name, 'rb') as filep:
        embedding_dict = pickle.load(filep)
    embedding_result = EmbeddingResult.load(embedding_dict)
    node_embeddings = embedding_result.get_node_embedding_from_index(0)
    edge_type_embeddings = embedding_result.get_edge_type_embedding_from_index(0)
    return node_embeddings, edge_type_embeddings

def get_l2_norm_edge_embedding(
    edge_embedding: np.ndarray
) -> np.ndarray:
    assert edge_embedding.ndim == 2
    return np.sqrt(np.power(edge_embedding, 2.0).sum(axis=1, keepdims=True))

def get_cosine_similarity_nodes(
    source_node_embedding: np.ndarray,
    destination_node_embedding: np.ndarray
) -> np.ndarray:
    """Return cosine similarity of the two nodes."""
    hadamard_product = source_node_embedding * destination_node_embedding
    norm = (
        get_l2_norm_edge_embedding(source_node_embedding) *
        get_l2_norm_edge_embedding(destination_node_embedding)
    )
    norm[norm < 1e-6] = 1e-6  # Prevent division by very small numbers
    return np.sum(hadamard_product, axis=1, keepdims=True) / norm

def get_connected_nodes(prot_curie, drug_curie, npkg):
    neighbors_all = []
    neighbors_all_filtered = []
    neighbors_intersect = []
    if prot_curie is not None:
        prot_neighbors = npkg.get_neighbour_node_names_from_node_name(prot_curie).tolist()
        if prot_neighbors:
            neighbors_all.extend(prot_neighbors)
    drug_neighbors = npkg.get_neighbour_node_names_from_node_name(drug_curie).tolist()
    if drug_neighbors:
        neighbors_all.extend(drug_neighbors)
    ##add to neighbors_all_filtered if gene, pathway, or process
    for node in neighbors_all:
        node_type = npkg.get_node_type_from_node_name(node)
        if node_type in ['gene', 'pathway', 'process']:
            neighbors_all_filtered.append(node)
    ##add to neighbors_intersect if in prot_neighbors and drug_neighbors
    for node in neighbors_all_filtered:
        if node in prot_neighbors and node in drug_neighbors:
            neighbors_intersect.append(node)
    return neighbors_all_filtered, neighbors_intersect

def get_similarity_nodes(path, connected_nodes, connected_nodes_intersect, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map):
    path_embeddings = get_path_embeddings(path, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map)[0:3]


if __name__ == "__main__":

    np_name = 'cinnamon'
    np_id = -7012332
    np_uri = 'http://napdi.org/napdi_srs_imports:cinnamomum_verum'
    drug_name = 'ranitidine'
    drug_id = 961047
    drug_uri = 'http://purl.obolibrary.org/obo/CHEBI_8776'
    adr_name = 'renal failure'
    adr_id = [10038435]
    adr_uri = ['http://purl.obolibrary.org/obo/HP_0000083']

    discovery_patterns_filename = np_name+'_'+drug_name+'_'+adr_name+'_discovery_patterns_scored.tsv'
    discovery_patterns = pd.read_csv(DATA_PATH+discovery_patterns_filename, sep='\t')

    npkg = load_graph()
    ##load node labels
    nodeLabelsdf = pd.read_csv(KG_PATH+'nodeLabels_CURIE_v3.0.0.tsv', sep='\t')
    nodeLabels = nodeLabelsdf.set_index('source')['entity_label'].to_dict()
    uri_df = pd.read_csv(KG_PATH+'uri_to_curie_map.tsv', sep='\t', header=None)
    uri_to_curie_map = uri_df.set_index(1)[0].to_dict()
    ##load embeddings
    node_embeddings, edge_type_embeddings = load_embeddings(model_name='ComplEx')

    ##get most similar pathways and processes based on cosine similarity of path embeddings
    for i in range(len(discovery_patterns.index)):
        path = discovery_patterns.iloc[i]
        prot_node = path['prot_id']
        prot_curie = uri_to_curie_map.get(prot_node)
        drug_curie = path['drug_curie']
        connected_nodes, connected_nodes_intersect = get_connected_nodes(prot_curie, drug_curie, npkg)

        connected_nodes_similarities = get_similarity_nodes(path, connected_nodes, connected_nodes_intersect, npkg, node_embeddings, edge_type_embeddings, uri_to_curie_map)