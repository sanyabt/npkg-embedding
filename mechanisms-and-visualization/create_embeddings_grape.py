import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from grape import Graph
from grape.embedders import TransEEnsmallen
from grape.embedders import TransRPyKEEN, TransDPyKEEN, RotatEPyKEEN

KG_PATH = '/home/sbt12/sbt12-work/NPKG-files/'
DIR_PATH = 'grape-embeddings/'

def load_graph(filename):
    npkg = Graph.from_csv(
        edge_path=KG_PATH+'NP-KG_BETWEENNESS-WEIGHTED-PK-CURIE_v2.0.0.tsv',
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
    ##print graph report
    #print(npkg)
    return npkg

if __name__ == "__main__":

    #load graph
    npkg_filename = 'NP-KG_BETWEENNESS-WEIGHTED-PK-CURIE_v2.0.0.tsv'
    npkg = load_graph(npkg_filename)

    #Defaults: embedding_size=100, epochs=100, relu_bias=1.0, learning_rate=0.05, learning_rate_decay=0.9, random_state=42
    ##very fast - 8 mins on CPU (implemented in Rust, not all models are that quick. Most random walk ones are)
    transe_embed_model = TransEEnsmallen(
        embedding_size=100,
        epochs=100, 
        random_state=42,
        verbose=True,
        enable_cache=True
    )
    time1 = datetime.now()
    transe_embedding = transe_embed_model.fit_transform(npkg)

    print(transe_embed_model.parameters())
    time2 = datetime.now()
    ##save embeddings
    transe_embedding_dict = transe_embedding.dump()
    with open(DIR_PATH+'npkg_transe_embedding_dict_default.pickle', 'wb') as file_transe:
        pickle.dump(transe_embedding_dict, file_transe)
    print('TransE time taken: {}'.format(time2-time1))
    
    ##Run other models (much slower, wrappers for pykeen models)
    #Default: embedding_size=100, relation_dim=30, epochs=100, random_state=42, training_loop=stochastic local closed world assumption, scoring_fct_norm (gamma) = 2, batch_size=2**10
    transr_embed_model = TransRPyKEEN(
        embedding_size=100,
        relation_dim=30,
        epochs=100,
        random_state=42,
        verbose=True,
        enable_cache=True
    )
    time3 = datetime.now()
    transr_embed_model.parameters()
    transr_embedding = transr_embed_model.fit_transform(npkg)
    time4 = datetime.now()
    print('TransR time taken: {}'.format(time4-time3))
    ##save embeddings (if no path provided above)
    transr_embedding_dict = transr_embedding.dump()
    with open(DIR_PATH+'npkg_transr_embedding_dict_default.pickle', 'wb') as file_transr:
        pickle.dump(transr_embedding_dict, file_transr)