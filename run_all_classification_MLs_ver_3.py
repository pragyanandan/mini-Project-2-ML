import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier


def get_model_and_param_grid(model_type):
    if model_type == "LogisticRegression":
        model = LogisticRegression()
        apply_scaling = True
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.01, 0.1, 1, 10, 100]
        }
        is_ensemble_allowed = True
    elif model_type == "DecisionTree":
        model = DecisionTreeClassifier()
        apply_scaling = False
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40, 50]
        }
        is_ensemble_allowed = True
    elif model_type == "RandomForest":
        model = RandomForestClassifier()
        apply_scaling = False
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'criterion': ['gini'],
            'max_depth': [None, 10, 20, 30, 40, 50]
        }
        is_ensemble_allowed = True
    elif model_type == "KNeighbors":
        model = KNeighborsClassifier()
        apply_scaling = True
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance']
        }
        is_ensemble_allowed = False
    elif model_type == "SVM":
        model = SVC(probability=True)
        apply_scaling = True
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
        is_ensemble_allowed = False
    elif model_type == "AdaBoost":
        model = AdaBoostClassifier()
        apply_scaling = False
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1]
        }
        is_ensemble_allowed = False
    elif model_type == "XGBoost":
        model = XGBClassifier()
        apply_scaling = False
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        is_ensemble_allowed = False
    else:
        raise ValueError("Invalid model type provided.")
    
    return model, param_grid, apply_scaling, is_ensemble_allowed


def scale_features(X, apply_scaling):
    if apply_scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X


def split_data(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def perform_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def perform_ensemble_grid_search(X_train, y_train, model_type):
    
    model1, param_grid1, _, _ = get_model_and_param_grid('LogisticRegression')
    model2, param_grid2, _, _ = get_model_and_param_grid('DecisionTree')
    model3, param_grid3, _, _ = get_model_and_param_grid('RandomForest')
    
    model1 = perform_grid_search(model1, param_grid1, X_train, y_train)
    model2 = perform_grid_search(model2, param_grid2, X_train, y_train)
    model3 = perform_grid_search(model3, param_grid3, X_train, y_train)
    
    ensemble_model = VotingClassifier(estimators=[('lr', model1), ('dt', model2), ('rf', model3)], voting='soft')
    ensemble_model.fit(X_train, y_train)
    return ensemble_model


def evaluate_metrics(y_true, y_pred, y_prob):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, target_labels, model_type):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cm_labels = np.array([[f'TN\n{tn}', f'FP\n{fp}'], [f'FN\n{fn}', f'TP\n{tp}']])

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=cm_labels, fmt='', cmap='Blues',
                xticklabels=[f'Predicted: {target_labels[0]}', f'Predicted: {target_labels[1]}'],
                yticklabels=[f'Actual: {target_labels[0]}', f'Actual: {target_labels[1]}'])
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix for {model_type}')
    plt.show()


def plot_roc_curve(y_true, y_prob, model_type):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f'{model_type} (AUC = {roc_auc_score(y_true, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()


def evaluate_classification_model(df, target_var, feature_list, model_type, target_labels, run_ensemble_mode=False):
    X = df[feature_list]
    y = df[target_var]
    
    try:
        if not run_ensemble_mode:
            # Initialize the model and parameter grid
            model, param_grid, apply_scaling, is_ensemble_allowed = get_model_and_param_grid(model_type)
            # Apply scaling if necessary
            X = scale_features(X, apply_scaling)
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = split_data(X, y)
            # Perform GridSearchCV
            optimized_clf = perform_grid_search(model, param_grid, X_train, y_train)
            ensemble_model_name = model_type
        
        
        
        if run_ensemble_mode:
            # Use an ensemble voting classifier
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = split_data(X, y)
            optimized_clf = perform_ensemble_grid_search(X_train, y_train, model_type)
            ensemble_model_name = model_type
     
        
        # Predictions
        train_preds = optimized_clf.predict_proba(X_train)[:, 1]
        test_preds = optimized_clf.predict_proba(X_test)[:, 1]
        train_class_preds = optimized_clf.predict(X_train)
        test_class_preds = optimized_clf.predict(X_test)
        
        # Evaluate metrics
        train_metrics = evaluate_metrics(y_train, train_class_preds, train_preds)
        test_metrics = evaluate_metrics(y_test, test_class_preds, test_preds)

        print(f"Model: {ensemble_model_name}")
        print("The accuracy on train data is ", train_metrics['accuracy'])
        print("The accuracy on test data is ", test_metrics['accuracy'])
        print("The precision on train data is ", train_metrics['precision'])
        print("The precision on test data is ", test_metrics['precision'])
        print("The recall on train data is ", train_metrics['recall'])
        print("The recall on test data is ", test_metrics['recall'])
        print("The f1 on train data is ", train_metrics['f1'])
        print("The f1 on test data is ", test_metrics['f1'])
        print("The roc_auc_score on test data is ", test_metrics['roc_auc'])
        
        plot_confusion_matrix(y_test, test_class_preds, target_labels, ensemble_model_name)
        plot_roc_curve(y_test, test_preds, ensemble_model_name)
        
        # Return metrics and test predictions for ROC curve
        return {
            'model_type': ensemble_model_name,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'test_preds': test_preds,
            'y_test': y_test
        }
    except ValueError as e:
        print(e)
        return None


def evaluate_classification_models(models, df, target_var, feature_list, target_labels):
    results = []

    run_ensemble_mode = False

    ## Run for all modles
    for model in models:
        result = evaluate_classification_model(df, target_var, feature_list, model, target_labels, run_ensemble_mode)
        if result is not None:
            results.append(result)
    ## Run for Ensemble model 
    run_ensemble_mode = True
    if run_ensemble_mode:
        model = 'ensemble_ML'
        result = evaluate_classification_model(df, target_var, feature_list, model, target_labels, run_ensemble_mode)
        if result is not None:
            results.append(result)

    # Convert results to a DataFrame for easier plotting
    metrics_df = pd.DataFrame({
        'model_type': [res['model_type'] for res in results],
        'test_accuracy': [res['test_metrics']['accuracy'] for res in results],
        'test_precision': [res['test_metrics']['precision'] for res in results],
        'test_recall': [res['test_metrics']['recall'] for res in results],
        'test_f1': [res['test_metrics']['f1'] for res in results],
        'test_roc_auc': [res['test_metrics']['roc_auc'] for res in results]
    })

    # Plot the metrics comparison for all models
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model_type', y=f'test_{metric}', data=metrics_df)
        plt.title(f'Test {metric.capitalize()} Comparison')
        plt.ylabel(f'Test {metric.capitalize()}')
        plt.xlabel('Model Type')
        plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
        plt.show()

    # Plot ROC AUC curves for all models
    plt.figure(figsize=(10, 8))
    for result in results:
        model_type = result['model_type']
        fpr, tpr, _ = roc_curve(result['y_test'], result['test_preds'])
        plt.plot(fpr, tpr, label=f"{model_type} (AUC = {result['test_metrics']['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve')
    plt.legend(loc='best')
    plt.show()


'''
target_var = 'def_pay'
feature_list = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE','AGE',\
                'PAY_1','PAY_2','PAY_3','PAY_4', 'PAY_5','PAY_6', \
                'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6', \
                'PAY_AMT1','PAY_AMT2', 'PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',  ]
target_labels = ['Yes','No']

models = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'KNeighbors', 'SVM', 'AdaBoost', 'XGBoost']
# Run without ensemble mode
evaluate_classification_models(models, defaulters, target_var, feature_list, target_labels, run_ensemble_mode=False)
# Run with ensemble mode
evaluate_classification_models(models, defaulters, target_var, feature_list, target_labels, run_ensemble_mode=True)
'''
