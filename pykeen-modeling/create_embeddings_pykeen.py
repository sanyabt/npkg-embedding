'''
Create and evaluate translational embeddings using PyKEEN
'''

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import TransE, TransR, TransD, RotatE
from datetime import datetime
import torch

KG_PATH = '/home/sbt12/sbt12-work/NPKG-files/'
DIR_PATH = 'pykeen-models/'

def load_graph(filename):
    tf = TriplesFactory.from_path(KG_PATH+filename)
    return tf

if __name__ == "__main__":
    ##load graph
    time1 = datetime.now()
    npkg_file = 'NP-KG-CURIE-only-v2.0.0.tsv'
    npkg = load_graph(npkg_file)

    print('Number of nodes, relations, triples: ', npkg.num_entities, npkg.num_relations, npkg.num_triples)
    time2 = datetime.now()
    print('Graph loaded in time: {}'.format(time2-time1))
    ##split data
    npkg_train, npkg_test, npkg_val = npkg.split([0.8, 0.1, 0.1], random_state=42)
    print('Size of split data: ', npkg_train.num_triples, npkg_test.num_triples, npkg_val.num_triples)

    ##create pipeline with default params
    '''
    Default params: embedding_dim=50, scoring_fct_norm=1, 
    Default pipeline parameters: regularizer=None, lr_scheduler=ExponentialLR, optimizer=Adagrad, training_loop='slcwa', negative_sampler='basic',  
    stopper=None, evaluator='RankBasedEvaluator'
    '''
    ##add time info
    time3 = datetime.now()
    transe_result = pipeline(
        training=npkg_train,
        testing=npkg_test,
        validation=npkg_val,
        model=TransE,
        model_kwargs=dict(embedding_dim=100),
        epochs=100,
        stopper='early',
        random_seed=42
    )

    ##save results
    transe_result.save_to_directory(DIR_PATH+'npkg_transe_default_pykeen')
    time4 = datetime.now()
    print('Model training and saving completed in time: {}'.format(time4-time3))

    ##load results
    #transe_model = torch.load(DIR_PATH+'npkg_transe_default_pykeen/npkg_transe_default_pykeen.pkl')

    ##get evaluation results (intrinsic)