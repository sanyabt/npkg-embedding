import pandas as pd
import numpy as np
import pickle, os
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
import argparse, logging

EMBED_PATH = 'embeddings/'
DATA_PATH = '../data/'
LOG_FILE = 'edge_prediction_embeddings.log'
logging.basicConfig(filename='edge_prediction.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Evaluate embeddings')
    parser.add_argument('--embed_model', type=str, default='TransE_l1', help='model name')
    parser.add_argument('--edge_feature', type=str, default='concat', help='Edge feature to use')
    parser.add_argument('--classifier', type=str, default='lr', help='Classifier to use')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--output_dir', type=str, default='results/', help='Output directory')
    args = parser.parse_args()
    return args

def load_embeddings(model_name):
    embed_file_name = 'npkg_'+model_name+'_embeddings.pickle'
    with open(EMBED_PATH+embed_file_name, 'rb') as filep:
        embedding_dict = pickle.load(filep)
    return embedding_dict

def load_data(filename):
    refdf = pd.read_csv(DATA_PATH+filename, sep='\t')
    le = LabelEncoder()
    refdf['interaction_type_encoded'] = le.fit_transform(refdf['interaction_type'])
    return refdf

def get_embeddings_df(embedding_dict):
    node_embeddings_df = embedding_dict['node_embeddings'][0]
    #edge_type_embeddings_df = embedding_dict['edge_type_embeddings'][0]
    return node_embeddings_df

def extract_features(edges, node_embeddings_df, edge_feature_method='concat'):
    features = []
    for u, v, edge_type in edges:
        try:
            node_u_embedding = node_embeddings_df.loc[u].values
            node_v_embedding = node_embeddings_df.loc[v].values
            #edge_type_embedding = edge_type_embeddings_df.loc[edge_type].values
            if edge_feature_method == 'concat':
                node_feature = np.concatenate([node_u_embedding, node_v_embedding])
            elif edge_feature_method == 'hadamard':
                node_feature = np.multiply(node_u_embedding,node_v_embedding)
            features.append(node_feature)
        except KeyError:
            continue
    return np.array(features)

def save_classifier_results(clf, X, y, kf, refdf, output_dir, model_name, classifier_name='lr'):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    aucs = []
    avg_precision_scores = []
    refdf_predict = refdf.copy()
    refdf_predict['edge_proba'] = np.nan
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
        recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
        f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
        aucs.append(roc_auc_score(y_test, y_proba, multi_class='ovr'))
        avg_precision_scores.append(average_precision_score(y_test, y_proba, average='weighted'))
        refdf_predict.loc[test_index, 'edge_proba'] = y_proba[:, 1]
    results = {
        'roc_auc': np.mean(aucs),
        'accuracy': np.mean(accuracy_scores),
        'precision': np.mean(precision_scores),
        'recall': np.mean(recall_scores),
        'f1': np.mean(f1_scores),
        'avg_precision': np.mean(avg_precision_scores)
    }
    refdf_predict.to_csv(output_dir+model_name+'_'+classifier_name+'_edge_predictions.tsv', sep='\t', index=False)
    return results

def get_baseline_performance():
    pass

def main(args):
    input_filename = 'reference-data/reference_data_curie_edges_20240725.tsv'
    args = parse_arguments()
    model_name = args.embed_model
    classifier = args.classifier
    n_folds = args.n_folds
    edge_feature_method = args.edge_feature
    output_dir = args.output_dir
    logging.info(f"Model: {model_name}, Classifier: {classifier}, Edge Feature: {edge_feature_method}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    embedding_dict = load_embeddings(model_name)
    refdf = load_data(input_filename)
    node_embeddings_df = get_embeddings_df(embedding_dict)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    edges = refdf[['source', 'target', 'interaction_type_encoded']].values
    X = extract_features(edges, node_embeddings_df, edge_feature_method)
    y = refdf['interaction_type_encoded'].values
    #Handle imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    if classifier == 'lr':
        clf = LogisticRegression(class_weight=class_weight_dict, max_iter=1000)
    elif classifier == 'mlp':
        clf = MLPClassifier(
        hidden_layer_sizes=(100,), 
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size='auto',
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        validation_fraction=0.1,
        shuffle=True,
        tol=1e-4,
        random_state=42
    )
        
    results = save_classifier_results(clf, X, y, kf, refdf, output_dir, model_name, classifier)
    logging.info(results)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)