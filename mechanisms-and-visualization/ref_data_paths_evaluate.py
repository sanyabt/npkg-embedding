import pandas as pd
import numpy as np
import pickle
import logging

#TODO: ####need to handle the case where the protein target is not in the dictionary

DIR_IN = 'mech-results/'
KG_PATH = '/home/sanya/np-kg/resources/'
logger = logging.getLogger()
fhandler = logging.FileHandler(filename='ref-data-path-eval-pattern2-test.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)

def hits_at_k(ground_truth, predictions, k):
    top_k_predictions = predictions[:k]
    return 1 if any(uri in ground_truth for uri in top_k_predictions) else 0

def evaluate_positive_interactions_with_proteins(df, pkdict):
    precisions = []
    precisions_top_10_weight = []
    precisions_top_10_embedding = []
    recalls = []
    recalls_top_10_weight = []
    recalls_top_10_embedding = []
    f1_scores = []
    f1_scores_top_10_weight = []
    f1_scores_top_10_embedding = []
    hits_at_5_list_embedding = []
    hits_at_5_list_weight = []
    hits_at_10_list_embedding = []
    hits_at_10_list_weight = []
    npdi_count = 0
    for source_id, group in df.groupby('source_id'):
        npdi_count += 1
        protein_target_names = group['protein_target'].iloc[0]
        protein_target_names = [name.strip() for name in protein_target_names.split('|')]
        ground_truth_uris = set()
        for name in protein_target_names:
            if name == 'CYP3A':
                name = 'CYP3A4'
            uris = pkdict.get(name, [])
            if uris:
                ground_truth_uris.update(uris)
        if not ground_truth_uris:
            logging.warning(f'No protein targets found for source_id: {source_id}')
            logging.info(f'Protein target names: {protein_target_names}')
            continue

        ##convert ground truth uris to str
        ground_truth_uris = set([str(uri) for uri in ground_truth_uris])
        
        ###if all results are considered
        '''predicted_uris = group['prot_id'].dropna().tolist()
        predicted_uris = [uri for uri in predicted_uris if pd.notna(uri)]

        predicted_uris_set = set(predicted_uris)
        tp = len(predicted_uris_set & ground_truth_uris)
        fp = len(predicted_uris_set - ground_truth_uris)
        fn = len(ground_truth_uris - predicted_uris_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)'''

        ####if only top 10 results are considered based on weight score
        sorted_group_weight = group.sort_values(by=['path_weight_score'], ascending=False)
        predicted_uris_subset_weight = sorted_group_weight['prot_id'].head(20).tolist()
        predicted_uris_subset_weight = [uri for uri in predicted_uris_subset_weight if pd.notna(uri)]
        predicted_uris_subset_weight = predicted_uris_subset_weight[:20]

        hits_at_5 = hits_at_k(ground_truth_uris, predicted_uris_subset_weight, 5)
        hits_at_10 = hits_at_k(ground_truth_uris, predicted_uris_subset_weight, 10)

        hits_at_5_list_weight.append(hits_at_5)
        hits_at_10_list_weight.append(hits_at_10)
        predicted_uris_subset_set_weight = set(predicted_uris_subset_weight)
        tp_top_10 = len(predicted_uris_subset_set_weight & ground_truth_uris)
        fp_top_10 = len(predicted_uris_subset_set_weight - ground_truth_uris)
        fn_top_10 = len(ground_truth_uris - predicted_uris_subset_set_weight)
        precision_top_10 = tp_top_10 / (tp_top_10 + fp_top_10) if (tp_top_10 + fp_top_10) > 0 else 0
        recall_top_10 = tp_top_10 / (tp_top_10 + fn_top_10) if (tp_top_10 + fn_top_10) > 0 else 0
        f1_score_top_10 = 2 * precision_top_10 * recall_top_10 / (precision_top_10 + recall_top_10) if (precision_top_10 + recall_top_10) > 0 else 0
        precisions_top_10_weight.append(precision_top_10)
        recalls_top_10_weight.append(recall_top_10)
        f1_scores_top_10_weight.append(f1_score_top_10)

        ####if only top 10 results are considered based on embedding score
        sorted_group_embedding = group.sort_values(by=['path_embedding_score'], ascending=False)
        predicted_uris_subset_embedding = sorted_group_embedding['prot_id'].head(20).tolist()
        predicted_uris_subset_embedding = [uri for uri in predicted_uris_subset_embedding if pd.notna(uri)]
        predicted_uris_subset_embedding = predicted_uris_subset_embedding[:20]

        predicted_uris_subset_set_embedding = set(predicted_uris_subset_embedding)
        tp_top_10 = len(predicted_uris_subset_set_embedding & ground_truth_uris)
        fp_top_10 = len(predicted_uris_subset_set_embedding - ground_truth_uris)
        fn_top_10 = len(ground_truth_uris - predicted_uris_subset_set_embedding)
        precision_top_10 = tp_top_10 / (tp_top_10 + fp_top_10) if (tp_top_10 + fp_top_10) > 0 else 0
        recall_top_10 = tp_top_10 / (tp_top_10 + fn_top_10) if (tp_top_10 + fn_top_10) > 0 else 0
        f1_score_top_10 = 2 * precision_top_10 * recall_top_10 / (precision_top_10 + recall_top_10) if (precision_top_10 + recall_top_10) > 0 else 0
        precisions_top_10_embedding.append(precision_top_10)
        recalls_top_10_embedding.append(recall_top_10)
        f1_scores_top_10_embedding.append(f1_score_top_10)

        hits_at_5 = hits_at_k(ground_truth_uris, predicted_uris_subset_embedding, 5)
        hits_at_10 = hits_at_k(ground_truth_uris, predicted_uris_subset_embedding, 10)
        hits_at_5_list_embedding.append(hits_at_5)
        hits_at_10_list_embedding.append(hits_at_10)
        
    logging.info(f'NPDI count: {npdi_count}')
    '''average_precision = sum(precisions) / npdi_count if npdi_count > 0 else 0
    average_recall = sum(recalls) / npdi_count if npdi_count > 0 else 0
    average_f1 = sum(f1_scores) / npdi_count if npdi_count > 0 else 0'''
    average_precision_top_10 = sum(precisions_top_10_weight) / npdi_count if npdi_count > 0 else 0
    average_recall_top_10 = sum(recalls_top_10_weight) / npdi_count if npdi_count > 0 else 0
    average_f1_top_10 = sum(f1_scores_top_10_weight) / npdi_count if npdi_count > 0 else 0
    hits_at_5 = sum(hits_at_5_list_weight) / npdi_count if npdi_count > 0 else 0
    hits_at_10 = sum(hits_at_10_list_weight) / npdi_count if npdi_count > 0 else 0
    logging.info(f'Positive interactions with protein targets: ')
    '''logging.info(f'Average precision: {average_precision}')
    logging.info(f'Average recall: {average_recall}')
    logging.info(f'Average F1 score: {average_f1}')'''

    logging.info(f'Sorting predictions by weight score')
    logging.info(f'Average precision at 10: {average_precision_top_10}')
    logging.info(f'Average recall at 10: {average_recall_top_10}')
    logging.info(f'Average F1 score at 10: {average_f1_top_10}')
    logging.info(f'Hits at 5: {hits_at_5}')
    logging.info(f'Hits at 10: {hits_at_10}')

    logging.info(f'Sorting predictions by embedding score')
    average_precision_top_10 = sum(precisions_top_10_embedding) / npdi_count if npdi_count > 0 else 0
    average_recall_top_10 = sum(recalls_top_10_embedding) / npdi_count if npdi_count > 0 else 0
    average_f1_top_10 = sum(f1_scores_top_10_embedding) / npdi_count if npdi_count > 0 else 0
    hits_at_5 = sum(hits_at_5_list_embedding) / npdi_count if npdi_count > 0 else 0
    hits_at_10 = sum(hits_at_10_list_embedding) / npdi_count if npdi_count > 0 else 0
    logging.info(f'Average precision at 10: {average_precision_top_10}')
    logging.info(f'Average recall at 10: {average_recall_top_10}')
    logging.info(f'Average F1 score at 10: {average_f1_top_10}')
    logging.info(f'Hits at 5: {hits_at_5}')
    logging.info(f'Hits at 10: {hits_at_10}')
    
    results = {
        'count': npdi_count,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'precisions_top_10': precisions_top_10_embedding,
        'recalls_top_10': recalls_top_10_embedding,
        'f1_scores_top_10': f1_scores_top_10_embedding,
        'hits_at_5': hits_at_5_list_embedding,
        'hits_at_10': hits_at_10_list_embedding
    }
    return results

def evaluate_negative_interactions(df):
    total_negative_npdis = negative_df['source_id'].nunique()
    negative_with_predictions = 0
    negative_without_predictions = 0
    for source_id, group in df.groupby('source_id'):
        predicted_targets = group['prot_id'].dropna().tolist()
        if predicted_targets:
            negative_with_predictions += 1  # False Positive
        else:
            negative_without_predictions += 1  # True Negative
    specificity = negative_without_predictions / total_negative_npdis if total_negative_npdis > 0 else 0
    false_positive_rate = negative_with_predictions / total_negative_npdis if total_negative_npdis > 0 else 0
    logging.info(f'Negative interactions: ')
    logging.info(f'Specificity: {specificity}')
    logging.info(f'False positive rate: {false_positive_rate}')
    logging.info(f'Negative Examples with Incorrect Predictions (False Positives): {negative_with_predictions}')
    logging.info(f'Negative Examples with Correct Predictions (True Negatives): {negative_without_predictions}')
    results = {
        'count': total_negative_npdis,
        'negative_with_predictions': negative_with_predictions,
        'negative_without_predictions': negative_without_predictions
    }
    return results

def evaluate_positive_interactions_no_proteins(df):
    total_positive_no_protein_npdis = df['source_id'].nunique()
    positive_no_protein_with_predictions = 0
    positive_no_protein_without_predictions = 0
    for source_id, group in df.groupby('source_id'):
        predicted_targets = group['prot_id'].dropna().tolist()
        if predicted_targets:
            positive_no_protein_with_predictions += 1
        else:
            positive_no_protein_without_predictions += 1
    logging.info(f'Positive interactions with no protein targets: ')
    logging.info(f'Total Positive NPDIs without Protein Targets: {total_positive_no_protein_npdis}')
    logging.info(f'Number with Predicted Mechanisms: {positive_no_protein_with_predictions}')
    logging.info(f'Number without Predicted Mechanisms: {positive_no_protein_without_predictions}')
    results = {
        'count': total_positive_no_protein_npdis,
        'positive_no_protein_with_predictions': positive_no_protein_with_predictions,
        'positive_no_protein_without_predictions': positive_no_protein_without_predictions
    }
    return results

if __name__ == '__main__':
    ##Pattern 1 files
    '''positive_with_protein_file = 'pattern1_results_protein_positive_scored.tsv'
    positive_with_no_protein_file = 'pattern1_results_no_protein_positive_scored.tsv'
    negative_file = 'pattern1_results_negative_scored.tsv'''

    ##Pattern 2 files
    positive_with_protein_file = 'pattern2_results_protein_positive_scored.tsv'
    negative_file = 'pattern2_results_negative_scored.tsv'

    pk_file = KG_PATH + 'pk_dict.pickle'
    with open(pk_file, 'rb') as f:
        pkdict = pickle.load(f)
    negative_df = pd.read_csv(DIR_IN+negative_file, sep='\t')
    positive_with_protein_df = pd.read_csv(DIR_IN+positive_with_protein_file, sep='\t')
    #positive_no_protein_df = pd.read_csv(DIR_IN+positive_with_no_protein_file, sep='\t')
    logging.info('Loaded data')
    
    pos_protein_results = evaluate_positive_interactions_with_proteins(positive_with_protein_df, pkdict)
    #pos_no_protein_results = evaluate_positive_interactions_no_proteins(positive_no_protein_df)
    negative_results = evaluate_negative_interactions(negative_df)

    ##pattern 1
    #total_count = pos_protein_results['count'] + pos_no_protein_results['count'] + negative_results['count']
    #tp_samples = sum(1 for p, r in zip(pos_protein_results['hits_at_10'], pos_protein_results['precisions']) if p > 0 and r > 0)
    #fn_samples = sum(1 for p, r in zip(pos_protein_results['hits_at_10'], pos_protein_results['precisions']) if p == 0 and r == 0)
    
    ##pattern 2
    total_count = 714
    tp_samples = sum(1 for p, r in zip(pos_protein_results['hits_at_10'], pos_protein_results['precisions_top_10']) if p > 0 and r > 0)
    fn_samples = sum(1 for p, r in zip(pos_protein_results['hits_at_10'], pos_protein_results['precisions_top_10']) if p == 0 and r == 0)
    
    fp_samples = negative_results['negative_with_predictions']
    tn_samples = negative_results['negative_without_predictions']

    tp_samples_top_10 = sum(1 for p, r in zip(pos_protein_results['hits_at_10'], pos_protein_results['precisions_top_10']) if p > 0 and r > 0)
    fn_samples_top_10 = sum(1 for p, r in zip(pos_protein_results['hits_at_10'], pos_protein_results['precisions_top_10']) if p == 0 and r == 0)

    sample_accuracy = (tp_samples + tn_samples) / (pos_protein_results['count'] + negative_results['count']) if (pos_protein_results['count'] + negative_results['count']) > 0 else 0
    sample_precision = tp_samples / (tp_samples + fp_samples) if (tp_samples + fp_samples) > 0 else 0
    sample_recall = tp_samples / (tp_samples + fn_samples) if (tp_samples + fn_samples) > 0 else 0
    sample_specificity = tn_samples / (tn_samples + fp_samples) if (tn_samples + fp_samples) > 0 else 0
    sample_precision_top_10 = tp_samples_top_10 / (tp_samples_top_10 + negative_results['negative_with_predictions']) if (tp_samples_top_10 + negative_results['negative_with_predictions']) > 0 else 0
    sample_recall_top_10 = tp_samples_top_10 / (tp_samples_top_10 + fn_samples_top_10) if (tp_samples_top_10 + fn_samples_top_10) > 0 else 0
    logging.info('Sample-level evaluation')
    logging.info(f'Total NPDI count: {total_count}')
    logging.info(f'Sample Accuracy: {sample_accuracy}')
    logging.info(f'Sample Precision: {sample_precision}')
    logging.info(f'Sample Recall (Sensitivity): {sample_recall}')
    logging.info(f'Sample Specificity: {sample_specificity}')

    logging.info(f'Sample Precision at 10: {sample_precision_top_10}')
    logging.info(f'Sample Recall at 10: {sample_recall_top_10}')
    logging.info(f'TP: {tp_samples}, FN: {fn_samples}, FP: {fp_samples}, TN: {tn_samples}')