"""
modelTraining.py
Alyssa Ang alysappleseed@gmail.com

This script contains the model training in the pipeline
1. Train and test datasets are loaded from processedData dir
2. Main algorithms are Random Forest and XGBoost algorithms
>> Inclusive of additional models like Decision Tree, Extra Trees, and Logistic Regression for comparison
3. Models are trained with specified hyperparams in config.py
4. Models are evaluated by AUC score and have metrics such as cross-validation, validation accuracy, and confusion matrixes
>> Feature importances are also looked at and considered towards understanding predictions, and consumer preferences

"""

# Import libraries
import sys 
import os

import pandas as pd
import numpy as np
import configparser
import pickle

# Append the parent directory of the current script to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration from config.py
from config import (
    enrichedData_path, targetCol, 
    testSize, randomState, valSize,
    processedData_path, output_dir
)

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
from imblearn.combine import SMOTETomek
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

import configparser

# Declare relevant functions here
def measure_error(y_true, y_pred, label):
    return pd.Series({
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'), 
        'recall': recall_score(y_true, y_pred, average='weighted'),       
        'f1': f1_score(y_true, y_pred, average='weighted'
                      )}, name=label)

def get_model_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print('Classification Report of model:', report)

def get_confusion_matrix(y_true, y_pred):
    # compute and print the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion matrix:\n', cm)
    TP = cm[0,0]
    TN = cm[1,1]
    FP = cm[0,1]
    FN = cm[1,0]
    
    classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
    classification_error = (FP + FN) / float(TP + TN + FP + FN)
    recall = TP / float(TP + FN)
    true_positive_rate = TP / float(TP + FN)
    false_positive_rate = FP / float(FP + TN)
    specificity = TN / (TN + FP)
    print()
    print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
    print('Classification error : {0:0.4f}'.format(classification_error))
    print()
    print('Sensitivity : {0:0.4f}'.format(recall))
    print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
    print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
    print('Specificity : {0:0.4f}'.format(specificity))
    print()

def get_validation_accuracy(clf, X_val, y_val):
    accuracy = clf.score(X_val, y_val)
    print(f"\nValidation Accuracy: {accuracy}")
    print()

def model_accuracy(classifier, X_train, y_train):
    classifier.fit(X_train, y_train)
    acr = classifier.score(X_test, y_test)
    print("Accuracy = ", acr)
    
# Predict and evaluate
def evaluation_metrics(clf, X_train, y_train):
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print(get_confusion_matrix(y_train, y_train_pred))
    print()
    print("Classification Report")
    print(classification_report(y_train, y_train_pred))
    print()

    y_probs = clf.predict_proba(X_test)
    y_probs = y_probs[:, 1]
    
    train_test_full_error = pd.concat([measure_error(y_train, y_train_pred, 'train'),
                                   measure_error(y_test, y_test_pred, 'test')],axis=1)
    print(train_test_full_error)
    print()
    auc_score = roc_auc_score(y_test, y_probs)
    print("AUC score = {:.5f}".format(auc_score))
    # Compute cross-validated ROC AUC
    cv_roc_auc = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')

    # Print the average ROC AUC score
    print(f'Cross-validated ROC AUC: {cv_roc_auc.mean()}')

# Returns the top 5 feature importances
def top5_feature_importances(clf):
    feature_imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Top 5 feature importances")
    print(feature_imp.head())
    print()
    
# Save the trained models to file
def save_models_pickle(clf, output_dir, model_name):
    # Create a valid filename
    file_path = os.path.join(output_dir, f"{model_name}.pkl")
    
    # Save the trained model
    with open(file_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved successfully at: {file_path}")
    
def decision_tree_clf():
    # Firstly compare with Decision Tree Classifier
    # DT to compare with RF
    print("Decision Tree Classifier for model comparison")
    print()
    print("Training DT CLF..")
    # Train a basic DT CLF
    dt_clf = DecisionTreeClassifier(random_state=42)
    print("> Training Complete!")
    print()
    model_accuracy(dt_clf, X_train, y_train)
    print()
    print(f"Tree node count = {dt_clf.tree_.node_count}\nMax depth = {dt_clf.tree_.max_depth}")
    print()

    # Compute and compare between Accuracy, Precision, F1 Score and Recall for train and test
    # Predict and evaluate
    evaluation_metrics(dt_clf, X_train, y_train)
    top5_feature_importances(dt_clf)
    print("> Next step, optimization")
    
    print()
    print("Finding the best params..")
    best_auc_score = -1  # Initialize with a value lower than possible AUC scores
    best_model = None
    best_params = {}
    
    # Faster and less computationally resources in comparison to GridSearchCV
    for depth in range(1, 12):
        for features in range(20, 100, 5):  # Vary max_features from 20 to 95 with step 5
            opt_model = DecisionTreeClassifier(max_depth=depth, max_features=features, random_state=42)
            opt_model.fit(X_train, y_train)

            # Calculate accuracy and AUC score on test set
            accuracy = opt_model.score(X_test, y_test)
            y_probs = opt_model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_probs)

            # Check if current model has higher AUC score than the best found so far
            if auc_score > best_auc_score:
                best_auc_score = auc_score
                best_model = opt_model
                best_params = {'max_depth': depth, 'max_features': features}

    print("Best Model:")
    print(f"max_depth={best_params['max_depth']}, max_features={best_params['max_features']}, best_auc_score={best_auc_score:.4f}")

    print()
    print("Train DT CLF with best values")
    print()
    dt_best_model = DecisionTreeClassifier(max_depth=best_params['max_depth'], max_features=best_params['max_features'], random_state=42)
    dt_best_model.fit(X_train, y_train)
    print("> Training Complete!")
    print()
    model_accuracy(dt_best_model, X_train, y_train)
    print
    # Compute and compare between Accuracy, Precision, F1 Score and Recall for train and test
    y_train_pred = dt_best_model.predict(X_train)
    y_test_pred = dt_best_model.predict(X_test)
    train_test_full_error = pd.concat([measure_error(y_train, y_train_pred, 'train'),
                                       measure_error(y_test, y_test_pred, 'test')],axis=1)
    print(train_test_full_error)
    print()

    # Predict and evaluate
    evaluation_metrics(dt_best_model, X_train, y_train)
    top5_feature_importances(dt_best_model)
    print("> DT CLF Done")
    save_models_pickle(dt_best_model, output_dir,"decisiontree")
    
def random_forest_clf():
    global X_train_smt, y_train_smt
    # Build a RF CLF
    print("Random Forest Classifier")
    print()
    print("Training RF CLF..")
    # Train a basic RF CLF
    rf_clf = RandomForestClassifier(oob_score=True,random_state=42,warm_start=True,n_jobs=-1)
    rf_clf = rf_clf.set_params(n_estimators=542)
    print("> Training Complete!")
    print()
    model_accuracy(rf_clf, X_train, y_train)
    print()

    # Compute and compare between Accuracy, Precision, F1 Score and Recall for train and test
    y_train_pred = rf_clf.predict(X_train)
    y_test_pred = rf_clf.predict(X_test)

    # Predict and evaluate
    evaluation_metrics(rf_clf, X_train, y_train)
    top5_feature_importances(rf_clf)
    print("> Next step, optimization")
    
    print()
    print(">>> Optimize by using SMOTEomek for oversampling and undersampling strategies")
    smt = SMOTETomek(random_state=42)
    X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)

    print(f"Original training set shape: {X_train.shape}, {y_train.shape}")
    print(f"Resampled training set shape: {X_train_smt.shape}, {y_train_smt.shape}")
    
    print()
    print("Optimize RF CLF with SMOTEomek")
    print()
    # with SMOTEomek
    print("Training RF CLF..")
    params = model_hyperparams['rf']
    rf_clf_smt = RandomForestClassifier(**params)
    rf_clf_smt.fit(X_train_smt, y_train_smt)

    y_train_pred = rf_clf_smt.predict(X_train_smt)
    y_test_pred = rf_clf_smt.predict(X_test)

    # Predict and evaluate
    evaluation_metrics(rf_clf_smt, X_train_smt, y_train_smt)
    top5_feature_importances(rf_clf_smt)
    print()
    print("> Optimizing metrics further..")
    
    # Further optimization with ExtraTrees CLF
    extra_trees = ExtraTreesClassifier(n_estimators=111, max_depth=42, bootstrap=True, n_jobs=-1, 
                                   min_samples_split=2,random_state=42, warm_start=True)
    extra_trees.n_estimators += 10
    extra_trees.fit(X_train_smt, y_train_smt)
    
    y_train_pred = extra_trees.predict(X_train_smt)
    y_test_pred = extra_trees.predict(X_test)
    # Predict and evaluate
    evaluation_metrics(extra_trees, X_train_smt, y_train_smt)
    top5_feature_importances(extra_trees)
    print("> Extra Trees CLF Done!")
    print("> This portion of CLF has been completed.")
    save_models_pickle(extra_trees, output_dir,"extratree")
    
def log_reg_clf():
    logreg = LogisticRegression(solver='sag', random_state=42)

    # Train the model
    logreg.fit(X_train_smt, y_train_smt)

    y_train_pred = logreg.predict(X_train_smt)
    y_test_pred = logreg.predict(X_test)
    # Predict and evaluate
    evaluation_metrics(logreg, X_train_smt, y_train_smt)
    
def xgboost_clf():
    # Init xgb
    # Convert X_train.columns to list (required for feature_names parameter in xgb.DMatrix)
    feature_names = X_train.columns.tolist()
    # Create DMatrix
    dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_names)
    
    print("Training first XGB Model...")
    params = model_hyperparams['xgb']
    xgb_clf = xgb.XGBClassifier(**params)
    xgb_clf.fit(X_train, y_train)
    y_train_pred = xgb_clf.predict(X_train)
    y_test_pred = xgb_clf.predict(X_test)
    # Predict and evaluate
    evaluation_metrics(xgb_clf, X_train, y_train)
    top5_feature_importances(xgb_clf)
    print("> Next step, optimization")
    print()
    
    # Define individual XGB models
    model1 = xgb.XGBClassifier(n_estimators=300, max_depth=20, random_state=42)
    model2 = xgb.XGBClassifier(n_estimators=200, max_depth=30, random_state=42)

    # Initialize VotingClassifier
    ensemble_clf = VotingClassifier(estimators=[('model1', model1), ('model2', model2)], voting='soft')

    # Fit the ensemble classifier
    ensemble_clf.fit(X_train, y_train)

    # Evaluate the classifier
    accuracy = ensemble_clf.score(X_test, y_test)

    # Predict probabilities for ROC AUC score
    y_probs = ensemble_clf.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, y_probs)

    # Print evaluation metrics
    print("Accuracy =", accuracy)
    print("AUC score = {:.5f}".format(auc_score))

    # Determine which model is better based on AUC score
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    y_probs_model1 = model1.predict_proba(X_test)[:, 1]
    y_probs_model2 = model2.predict_proba(X_test)[:, 1]
    auc_score_model1 = roc_auc_score(y_test, y_probs_model1)
    auc_score_model2 = roc_auc_score(y_test, y_probs_model2)

    if auc_score_model1 > auc_score_model2:
        print("Model 1 is better.")
        evaluation_metrics(model1, X_train, y_train)
        print()
        top5_feature_importances(model1)
    elif auc_score_model2 > auc_score_model1:
        print("Model 2 is better.")
        evaluation_metrics(model2, X_train, y_train)
        print()
        top5_feature_importances(model1)
    else:
        print("Both models have the same AUC score.")
        evaluation_metrics(ensemble_clf, X_train, y_train)
        print()
        top5_feature_importances(ensemble_clf)
    
    # Fit model with SMOTEomek
    model1.fit(X_train_smt, y_train_smt)
    y_train_pred = model1.predict(X_train_smt)
    y_test_pred = model1.predict(X_test)
    # Predict and evaluate
    evaluation_metrics(model1, X_train_smt, y_train_smt)
    print()
    top5_feature_importances(model1)
    print()
    print("> XGB CLF done!")
    save_models_pickle(model1, output_dir,"xgboost")

# Model training 
def model_training():
    # Read X values
    X_train = pd.read_csv(f'{processedData_path}/X_train.csv')
    X_test = pd.read_csv(f'{processedData_path}/X_test.csv')
    X_val = pd.read_csv(f'{processedData_path}/X_val.csv')

    # Read y values
    y_train = pd.read_csv(f'{processedData_path}/y_train.csv', header=None)
    y_test = pd.read_csv(f'{processedData_path}/y_test.csv', header=None)
    y_val = pd.read_csv(f'{processedData_path}/y_val.csv', header=None)
    y_train, y_test, y_val = np.ravel(y_train), np.ravel(y_test), np.ravel(y_val)

    # Initial look into the shapes
    print("Shape of X train = ", X_train.shape)
    print("Shape of y train = ", y_train.shape)
    print()
    
    # Model training for different models
    decision_tree_clf()
    # log_reg_clf()
    xgboost_clf()

# Main
model_training()