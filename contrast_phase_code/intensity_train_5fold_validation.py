# Overview
# This Python script is designed for machine learning (ML) tasks involving the training, evaluation, and comparison of Random Forest and Gradient Boosting models. The script performs K-Fold Cross-Validation and outputs various metrics. It also incorporates statistical tests to compare the performance of the two models.


import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load
from datetime import datetime
from sklearn.model_selection import KFold
import numpy as np
from scipy import stats

def load_data(filename):
    data = pd.read_csv(filename)
    X = data.iloc[:, 1:-1]
    y = data.iloc[:, -1]
    sample_name = data.iloc[:, 0]
    return X, y, sample_name

def train_model(model, X_train, y_train, model_name):
    model.fit(X_train, y_train)
    dump(model, model_name + '.joblib')  # Save the trained model to a file
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    prediction_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, digits=3)
    cm = confusion_matrix(y_test, predictions)
    return accuracy, report, cm, predictions, prediction_proba

def save_results(sample_id_test, predictions, y_test, save_path):
    df = pd.DataFrame({'sample_id': sample_id_test, 'prediction': predictions, 'label': y_test})
    df.to_csv(save_path, index=False)

def main():
    # !!! rename the train test and val data path accordingly
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path_root = 'intensity_trained_models/'
    output_folder_for_this_run = 'intensity_trained_models_and_results_knn_'+timestamp
    output_path = os.path.join(output_path_root, output_folder_for_this_run)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    #!!! paths for thor server
    # X_train, y_train, _ = load_data('/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics_mean_imputed.csv')
    # X_val, y_val, _ = load_data('/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/validation_set_respacing113_radiomics_mean_imputed.csv')
    # X_test, y_test, sample_id_test = load_data('/home/lance/NAS/users/Akshaya/My_Datasets/DeepLesion509_original_scan_only_keep_key_organs_radiomics_respacing113/DeepLesion509_original_scan_only_keep_key_organs_radiomics_respacing113_radiomics_mean_imputed.csv')  # Load test data

    #!!! paths for helix and biowulf
    X_train, y_train, _ = load_data('/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics_mean_imputed.csv')
    X_val, y_val, _ = load_data('/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/validation_set_respacing113_radiomics_mean_imputed.csv')
    X_test, y_test, sample_id_test = load_data('/data/drdcad/lance/dataset/contrast_phase_dataset/DeepLesion509_original_scan_only_keep_key_organs_radiomics_respacing113/DeepLesion509_original_scan_only_keep_key_organs_radiomics_respacing113_radiomics_mean_imputed.csv')  # Load test data
    # X_test, y_test, sample_id_test = load_data('/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/test_set_respacing113_radiomics_knn_imputed.csv')  # Load test data

    # X_train, y_train, _ = load_data('/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics_knn_imputed.csv')
    # X_val, y_val, _ = load_data('/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/validation_set_respacing113_radiomics_knn_imputed.csv')
    # X_test, y_test, sample_id_test = load_data('/data/drdcad/lance/dataset/contrast_phase_dataset/DeepLesion509_original_scan_only_keep_key_organs_radiomics_respacing113/DeepLesion509_original_scan_only_keep_key_organs_radiomics_respacing113_radiomics_knn_imputed.csv')  # Load test data
    # X_test, y_test, sample_id_test = load_data('/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/test_set_respacing113_radiomics_knn_imputed.csv')  # Load test data


    X = pd.concat([X_train, X_val])
    y = pd.concat([y_train, y_val])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    rf_accuracies = []
    gb_accuracies = []
    
    fold_count = 1
    for train_index, test_index in kf.split(X):
        X_train, X_val_5fold = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val_5fold = y.iloc[train_index], y.iloc[test_index]
        
        rf_model = RandomForestClassifier(random_state=42)
        gb_model = GradientBoostingClassifier(random_state=42)
        
        # Train and evaluate Random Forest model
        rf_model = train_model(rf_model, X_train, y_train, os.path.join(output_path, f'rf_model_fold_{fold_count}'))
        rf_accuracy, rf_report, rf_cm, rf_predictions, rf_prediction_proba = evaluate_model(rf_model, X_val_5fold, y_val_5fold)
        rf_accuracies.append(rf_accuracy)
        
        # Save results
        save_results(test_index, rf_prediction_proba, rf_predictions, y_val_5fold, os.path.join(output_path, f'rf_model_results_fold_{fold_count}.csv'))

        # Train and evaluate Gradient Boosting model
        gb_model = train_model(gb_model, X_train, y_train, os.path.join(output_path, f'gb_model_fold_{fold_count}'))
        gb_accuracy, gb_report, gb_cm, gb_predictions, gb_prediction_proba = evaluate_model(gb_model, X_val_5fold, y_val_5fold)
        gb_accuracies.append(gb_accuracy)

        # Save results
        save_results(test_index, gb_prediction_proba, gb_predictions, y_val_5fold, os.path.join(output_path, f'gb_model_results_fold_{fold_count}.csv'))

        
        # Print validation results
        print("Random Forest accuracy on validation set: ", rf_accuracy)
        print("Gradient Boosting accuracy on validation set: ", gb_accuracy)
        print("Random Forest classification report on validation set:\n", rf_report)
        print("Gradient Boosting classification report on validation set:\n", gb_report)
        print("Random Forest confusion matrix on validation set:\n", rf_cm)
        print("Gradient Boosting confusion matrix on validation set:\n", gb_cm)

        fold_count += 1
        
    # Perform t-test
    t_stat, p_value = stats.ttest_rel(rf_accuracies, gb_accuracies)
    rf_accuracy_std = np.std(rf_accuracies, ddof=1)  # ddof=1 for sample standard deviation
    gb_accuracy_std = np.std(gb_accuracies, ddof=1)
    
    print("Random Forest mean accuracy: ", np.mean(rf_accuracies))
    print("Gradient Boosting mean accuracy: ", np.mean(gb_accuracies))
    print("Random Forest accuracy standard deviation: ", rf_accuracy_std)
    print("Gradient Boosting accuracy standard deviation: ", gb_accuracy_std)
    print("t-statistic: ", t_stat)
    print("p-value: ", p_value)
    
    
    
    # rf_model = RandomForestClassifier(random_state=42)
    # gb_model = GradientBoostingClassifier(random_state=42)

    #  #!!! rename the model saving name and path

    # rf_model = train_model(rf_model, X_train, y_train, os.path.join(output_path,'rf_model'))
    # gb_model = train_model(gb_model, X_train, y_train, os.path.join(output_path,'gb_model'))
    # # rf_model = train_model(rf_model, X_train, y_train, os.path.join(output_path,'rf_model_knn_imputed_data'))
    # # gb_model = train_model(gb_model, X_train, y_train, os.path.join(output_path,'gb_model_knn_imputed_data'))

    # # Evaluate models on the val set
    # rf_val_accuracy, rf_val_report, rf_val_cm, _ = evaluate_model(rf_model, X_val, y_val)
    # gb_val_accuracy, gb_val_report, gb_val_cm, _ = evaluate_model(gb_model, X_val, y_val)

    # # Evaluate models on the test set
    # rf_test_accuracy, rf_test_report, rf_test_cm, rf_test_predictions = evaluate_model(rf_model, X_test, y_test)
    # gb_test_accuracy, gb_test_report, gb_test_cm, gb_test_predictions = evaluate_model(gb_model, X_test, y_test)

    #  #!!! rename the results saving name and path
    # save_results(sample_id_test, rf_test_predictions, y_test, os.path.join(output_path,'rf_model_test_results'))
    # save_results(sample_id_test, gb_test_predictions, y_test, os.path.join(output_path,'gb_model_test_results'))
    # # save_results(sample_id_test, rf_test_predictions, y_test, os.path.join(output_path,'rf_model_test_results_knn_imputed_data'+datetime+'.csv'))
    # # save_results(sample_id_test, gb_test_predictions, y_test, os.path.join(output_path,'gb_model_test_results_knn_imputed_data'+datetime+'.csv'))
    

    # print("Random Forest accuracy on validation set: ", rf_val_accuracy)
    # print("Gradient Boosting accuracy on validation set: ", gb_val_accuracy)
    # print("Random Forest classification report on validation set:\n", rf_val_report)
    # print("Gradient Boosting classification report on validation set:\n", gb_val_report)
    # print("Random Forest confusion matrix on validation set:\n", rf_val_cm)
    # print("Gradient Boosting confusion matrix on validation set:\n", gb_val_cm)

    # # Print test results
    # print("Random Forest accuracy on test set: ", rf_test_accuracy)
    # print("Gradient Boosting accuracy on test set: ", gb_test_accuracy)
    # print("Random Forest classification report on test set:\n", rf_test_report)
    # print("Gradient Boosting classification report on test set:\n", gb_test_report)
    # print("Random Forest confusion matrix on test set:\n", rf_test_cm)
    # print("Gradient Boosting confusion matrix on test set:\n", gb_test_cm)

if __name__ == '__main__':
    main()
