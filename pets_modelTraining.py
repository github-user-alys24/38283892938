"""
Alyssa Ang 211082M

This script contains the Model Training pipeline
1. Train and test datasets are loaded from processedData dir
2. Includes Random Forest and XGBoost algorithms
3. Models are trained with specified hyperparams in config.py
4. Models are evaluated by accuracy and have metrics such as cross-validation,
   validation accuracy, and confusion matrixes
   Feature importances are also looked at and considered towards understanding
   prediction
"""
# Import libraries
import os 
import sys

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc
import xgboost as xgb
from xgboost import plot_importance
import configparser
import pickle
from sklearn.model_selection import train_test_split

# Append the parent directory of the current script to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configuration from params.py
from config import processedData_path, trainModels, model_hyperparams, output_dir

class modelTrain:
    def __init__(self):
        self.models = trainModels
        
    # Get the corresponding X and y train, test and val values
    def get_TTV_datasets(self):
        # Read X values
        X_train = pd.read_csv(f'{processedData_path}/X_train.csv')
        X_test = pd.read_csv(f'{processedData_path}/X_test.csv')
        X_val = pd.read_csv(f'{processedData_path}/X_val.csv')

        # Read y values
        y_train = pd.read_csv(f'{processedData_path}/y_train.csv', header=None)
        y_test = pd.read_csv(f'{processedData_path}/y_test.csv', header=None)
        y_val = pd.read_csv(f'{processedData_path}/y_val.csv', header=None)
        y_train, y_test, y_val = np.ravel(y_train), np.ravel(y_test), np.ravel(y_val)
        return X_train, X_test, X_val, y_train, y_test, y_val
    
    # Returns a train/test table on accuracy, precision, recall, f1 
    def measure_error(y_true, y_pred, label):
        return pd.Series({
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'), 
            'recall': recall_score(y_true, y_pred, average='weighted'),       
            'f1': f1_score(y_true, y_pred, average='weighted'
                          )}, name=label)
    
    def get_model_classification_report(self, y_true, y_pred):
        report = classification_report(y_true, y_pred)
        print('Classification Report of model:', report)

    def get_confusion_matrix(self, y_true, y_pred, plot_heatmap=True):
        # compute and print the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print('Confusion matrix:\n', cm)
        if plot_heatmap:
            # plot the confusion matrix as a heatmap
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
    
    def get_validation_accuracy(self, clf, X_val, y_val):
        accuracy = clf.score(X_val, y_val)
        print()
        print(f"\nValidation Accuracy: {accuracy}")
        print()
    
    def model_training(self, model, X_train, y_train):
        # Random Forest Classifier
        if model == 'rf':

            feature_cols=[x for x in X.columns]
            print("Information on features and X, y shape")
            print()
            print("Number of features= {}".format(len(feature_cols)))
            print("Shape of X train = ", X_train.shape)
            print("Shape of y train = ", y_train.shape)
            print()

            # Firstly compare with Decision Tree Classifier
            # DT to compare with RF
            from config import processedData_path, trainModels, model_hyperparams, output_dir
            from sklearn.model_selection import cross_val_score
            print()
            print("Decision Tree Classifier for model comparison")
            print()
            print("Training DT CLF...")
            dt_clf = DecisionTreeClassifier(random_state=42)
            print("Done! :D")
            model_accuracy(dt_clf, X_train, y_train)
            print()
            
            # Compute and compare between Accuracy, Precision, F1 Score and Recall for train and test
            print("Acc, Precision, F1 Score and Recall table")
            y_train_pred = dt_clf.predict(X_train)
            y_test_pred = dt_clf.predict(X_test)
            train_test_full_error = pd.concat([measure_error(y_train, y_train_pred, 'train'),
            measure_error(y_test, y_test_pred, 'test')],axis=1)
            print(train_test_full_error)
            print()
            
            # Visualize DT tree
            plt.figure(figsize=(15,10))
            plot_tree(dt_clf, filled=True)
            plt.title('Decision Tree Visualization', fontsize=24)
            plt.show()
            
            # RF classifier
            print()
            print("Random Forest Classifier")
            print()
            print("Training RF CLF...")
            params = model_hyperparams['rf']
            rf_clf = RandomForestClassifier(**params)
            rf_clf.fit(X_train, y_train)
            print("Done! :D")
            print()

            # Visualize feature importances to better understand each feature's weight
            feature_imp = pd.Series(rf_clf.feature_importances_, index=feature_cols).sort_values(ascending=False)
            fig = plt.figure(figsize=(12,5))
            ax = feature_imp.plot(kind='bar',color="maroon")
            ax.set_title('Feature Importances for RF')
            ax.set(ylabel='Relative Importance')
            print()
            
            # Cross-validation score 
            scores = cross_val_score(rf_clf, X_train, y_train, cv=4, scoring='accuracy')
            print("Cross-validation scores:", scores)
            print()
            y_train_pred = rf_clf.predict(X_train)
            y_test_pred = rf_clf.predict(X_test)
            train_test_full_error = pd.concat([measure_error(y_train, y_train_pred, 'train'),
            measure_error(y_test, y_test_pred, 'test')],axis=1)
            print("Acc, Precision, F1 Score and Recall table")
            print(train_test_full_error)
            print()
            
            # Perform gridSearch
            # Define the parameter grid
            param_grid = {
                'n_estimators': [100, 200, 300, 400],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'max_features': ['sqrt', 'log2']
            }

            # Instantiate RandomForestClassifier
            rf_clf = RandomForestClassifier(random_state=42)

            # Perform Grid Search CV
            grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid, cv=4, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            # Get best parameters and best score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            print("Performing GridSearch to find best params...")
            print()
            print("Found the best parameters!")
            print()
            print("Best Parameters:", best_params)
            print("Best Cross-validation Accuracy:", best_score)
            
            # Create a new RandomForestClassifier with the best parameters
            clf = RandomForestClassifier(random_state=42, **best_params)
            clf.fit(X_train, y_train)
            print()
            print("RF clf fitted with best params")
            model_accuracy(clf, X_train, y_train)
            print()
            
        # XGBoost
        elif model == 'xgb':
            
            cleaned_columns = []
            for col in X_train.columns:
                cleaned_col = col.replace('[', '').replace(']', '').replace('<', '')
                cleaned_columns.append(cleaned_col)

            # Update DataFrame with cleaned column names
            X_train.columns = cleaned_columns
            X_test.columns = cleaned_columns

            # Convert X_train.columns to list (required for feature_names parameter in xgb.DMatrix)
            feature_names = X_train.columns.tolist()

            # Create DMatrix
            dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_names)
            
            # Instantiate XGB
            print()
            print("Training first XGB Model...")
            params = model_hyperparams['xgb']
            xgb_clf = xgb.XGBClassifier(**params)
            
            xgb_clf.fit(X_train, y_train)
            print("Done! :D")

            # Evaluate the classifier
            accuracy = xgb_clf.score(X_test, y_test)
            print("Accuracy:", accuracy)
            print()
            print("Will Voting Classifiers help derive better XGB models?")
            model1 = xgb.XGBClassifier(n_estimators=300, max_depth=20, random_state=42)
            model2 = xgb.XGBClassifier(n_estimators=200, max_depth=30, random_state=42)

            ensemble_clf = VotingClassifier(estimators=[('model1', model1), ('model2', model2)], voting='soft')
            ensemble_clf.fit(X_train, y_train)
            # Evaluate the classifier
            accuracy = ensemble_clf.score(X_test, y_test)
            print("Accuracy:", accuracy)
            
            # Create DMatrix
            dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_names)

            # Define the parameter grid
            param_grid = { 
                'max_depth': [10, 20, 30, 40],
                'subsample': [0.5, 0.7, 0.9],
                'colsample_bytree': [0.5, 0.7, 0.9]
            }

            # Perform Grid Search CV
            print()
            print("Performing GridSearch to find best params...")
            grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            
            # Get best parameters and best score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            print()
            print("Found the best parameters!")
            
            # Print the best parameters and the corresponding accuracy
            print("Best Parameters:", grid_search.best_params_)
            print("Best Accuracy:", grid_search.best_score_)
            print()
            print("XGB clf fitted with best params")
            clf = xgb.XGBClassifier(**best_params)
            clf.fit(X_train, y_train)
            
            # PLot feature importance for xgb
            plt.figure(figsize=(10, 8))
            plot_importance(clf, max_num_features=10)  
            plt.title('Feature Importance for XGB fitted with best params')
            plt.show()
        
        clf.fit(X_train, y_train)
        # Save the trained models
        with open(f'{output_dir}/{model}.pkl', 'wb') as f:
            pickle.dump(clf, f)
        return clf

    def evaluate_model(self, clf, X_test, y_test):
        y_pred = clf.predict(X_test)
        # Get Model classification report
        self.get_model_classification_report(y_test, y_pred)
        # Get Confusion Matrix
        self.get_confusion_matrix(y_test, y_pred)
    
if __name__ == "__main__":
    Trainer = modelTrain()
    X_train, X_test, X_val, y_train, y_test, y_val = Trainer.get_TTV_datasets()
    for model in trainModels:
        print("Training and evaluating {} model ...".format(model))
        clf = Trainer.model_training(model, X_train, y_train)
        Trainer.evaluate_model(clf, X_test, y_test)
        Trainer.get_validation_accuracy(clf, X_val, y_val)
        print("Training and evaluation for {} model has completed!".format(model))
        