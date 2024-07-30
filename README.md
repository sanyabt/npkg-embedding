# npkg-embedding
Embeddings related scripts and notebooks for NP-KG

## Workflow

### Note: Steps 1-5 in dglke-modeling/ and 6-8 in embeddings-evaluation/

1. **Split NP-KG into Train, Test, and Validation Data**:
    - Ensure the splits are connected to avoid isolated subgraphs (connected holdout)
    - Save as TSV files for train, validation, test data
    - Notebook: process-and-split-npkg.ipynb
    - Output: data/
2. **Train KGE Models with DGL-KE using SLURM with default parameters**:
    - Create SLURM job scripts for each model (TransE, RotatE, DistMult, ComplEx, Conv).
    - Evaluate on test data
    - Script: {model_name}_train_eval_default.slurm
    - Output: test data performance in log, embeddings in output/
3. **HPO**
    - HPO with quasi random trials and evaluation with validation data
    - Report metrics and choose the best hyperparameters for each model based on MRR on validation data.
    - Scripts: change model name in “hpo_model_train_script.py” and run “hpo_train_validate.slurm”
    - Output: best config parameters saved to log with validation MRR
4. **Intrinsic evaluation**
    - With best parameters, run models again (embeddings are not saved) and evaluate on test data.
    - Script: {model_name}_train_eval_best.slurm
    - Output: test data performance in logs
5. **Generate embeddings for all data**
    - With best parameters, run models on all data and save embeddings
    - Script: {model_name}_train_all_data.slurm
    - Output: embeddings and config saved in output/
6. **Load embeddings and reference dataset**
    - Process and save embeddings to pickle (load-process-embeddings.ipynb)
    - Save reference data as edges (npkg-data-process-for-embeddings.ipynb)
7. **Edge prediction**
    - Feature extraction methods: concatenation or hadamrd
    - 5-fold cross validation
    - Classification with logistic regression and multilayer perceptron
    - Script: edge_prediction_embeddings.py
    - Input parameters: embed_model, edge_feature, classifier, n_folds
    - Command example: ```python edge_prediction_embeddings.py --embed_model ComplEx --edge_feature concat --classifier lr --output_dir results/ ```
    - Output: prediction probabilities in results/, edge_prediction.log
8. **Edge Type Prediction**: TODO
9. **Visualization**
    - Embeddings TSNE visualization for node degrees, node types, edge types, edge weights: transe-edge-prediction-vis.ipynb
    - Results: plot_results.ipynb
    - Output: images/