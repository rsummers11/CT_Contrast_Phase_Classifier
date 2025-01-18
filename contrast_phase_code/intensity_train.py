# Overview
# Load preprocessed training, validation, and test datasets from specified paths.
# Train Random Forest and Gradient Boosting models on the training dataset.
# Evaluate the models on the validation and test datasets.
# Save trained models and evaluation results, such as accuracy, classification reports, and confusion matrices, to specified output directories.


import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump, load
from datetime import datetime


def load_data(filename):
    data = pd.read_csv(filename)
    sample_names = data.iloc[:, 0]
    X, y = data.drop(columns=['fullpath', 'GT']
                     ), data['GT'] if 'GT' in data.columns else None
    return X, y, sample_names


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


def save_results(sample_id_test, prediction_proba, predictions, y_test, save_path):
    # Assuming prediction_proba is an n by 5 matrix
    # Create a DataFrame where each probability is in its own column
    proba_df = pd.DataFrame(prediction_proba, columns=[
                            f'prob_class_{i}' for i in range(5)])

    # Convert the sample_id_test to a DataFrame
    sample_id_test.rename('sample_id', inplace=True)
    y_test.rename('label', inplace=True)
    predictions_df = pd.DataFrame(predictions, columns=['prediction'])
    final_df = pd.concat(
        [sample_id_test, proba_df, predictions_df, y_test], axis=1)
    final_df.to_csv(save_path, index=False)


def main():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # !!! example: output_path_root = 'intensity_trained_models/'
    output_path_root = 'contrast_phase_code/intensity_trained_models/'
    # output_folder_for_this_run = 'intensity_trained_models_and_results_mean'+timestamp
    output_folder_for_this_run = 'intensity_trained_models_and_results_knn_'+timestamp
    # output_folder_for_this_run = 'intensity_trained_models_and_results_knn_within_class'+timestamp
    output_path = os.path.join(output_path_root, output_folder_for_this_run)
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    # !! change the path to our own from previous step
    X_train, y_train, _ = load_data('contrast_phase_code/dataset/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics_intensity_w_hu_std_knn_imputed.csv')
    X_val, y_val, _ = load_data('contrast_phase_code/dataset/Attempt2_only_keep_key_organs_radiomics_respacing113/validation_set_respacing113_radiomics_intensity_w_hu_std_knn_imputed.csv')
    X_test, y_test, sample_id_test = load_data(
        'contrast_phase_code/dataset/Attempt2_only_keep_key_organs_radiomics_respacing113/test_set_respacing113_radiomics_intensity_w_hu_std_knn_imputed.csv')  # Load test data
    
    #!!path example:
    # X_train, y_train, _ = load_data(
    #     '/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics_intensity_w_hu_std_knn_imputed.csv')
    # X_val, y_val, _ = load_data(
    #     '/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/validation_set_respacing113_radiomics_intensity_w_hu_std_knn_imputed.csv')
    # X_test, y_test, sample_id_test = load_data(
    #     '/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/test_set_respacing113_radiomics_intensity_w_hu_std_knn_imputed.csv')  # Load test data

    #!! make a reasonable model saving name and path
    rf_model_name = 'rf_model_knn_imputed_data'
    gb_model_name = 'gb_model_knn_imputed_data'

    #!! make a reasonable name for your output prediction csv
    # example:  
    # rf_output_csv_name = 'rf_model_test_results_knn_imputed_data.csv'    
    # gb_output_csv_name = 'gb_model_test_results_knn_imputed_data.csv' 
    rf_output_csv_name = 'rf_model_test_results_knn_imputed_data.csv'    
    gb_output_csv_name = 'gb_model_test_results_knn_imputed_data.csv' 


    rf_model = RandomForestClassifier(random_state=42)
    gb_model = GradientBoostingClassifier(random_state=42)

    # rf_model = train_model(rf_model, X_train, y_train, os.path.join(
    #     output_path, 'rf_model_knn_within_class_imputed_data'))
    # gb_model = train_model(gb_model, X_train, y_train, os.path.join(
    #     output_path, 'gb_model_knn_within_class_imputed_data'))
    # rf_model = train_model(rf_model, X_train, y_train, os.path.join(output_path,'rf_model'))
    # gb_model = train_model(gb_model, X_train, y_train, os.path.join(output_path,'gb_model'))
    rf_model = train_model(rf_model, X_train, y_train, os.path.join(
        output_path, rf_model_name))
    gb_model = train_model(gb_model, X_train, y_train, os.path.join(
        output_path, gb_model_name))
    # rf_model = train_model(rf_model, X_train, y_train, os.path.join(
    #     output_path, 'rf_model_mean_imputed_data'))
    # gb_model = train_model(gb_model, X_train, y_train, os.path.join(
    #     output_path, 'gb_model_mean_imputed_data'))

    # Evaluate models on the val set
    rf_val_accuracy, rf_val_report, rf_val_cm, * \
        _ = evaluate_model(rf_model, X_val, y_val)
    gb_val_accuracy, gb_val_report, gb_val_cm, * \
        _ = evaluate_model(gb_model, X_val, y_val)

    # Evaluate models on the test set
    rf_test_accuracy, rf_test_report, rf_test_cm, rf_test_predictions, rf_test_predcition_proba = evaluate_model(
        rf_model, X_test, y_test)
    gb_test_accuracy, gb_test_report, gb_test_cm, gb_test_predictions, gb_test_predcition_proba = evaluate_model(
        gb_model, X_test, y_test)

    # save_results(sample_id_test, rf_test_predictions, y_test, os.path.join(output_path,'rf_model_test_results.csv'))
    # save_results(sample_id_test, gb_test_predictions, y_test, os.path.join(output_path,'gb_model_test_results.csv'))
    # save_results(sample_id_test, rf_test_predcition_proba, rf_test_predictions, y_test,
    #              os.path.join(output_path, 'rf_model_test_results_knn_within_class_imputed_data.csv'))
    # save_results(sample_id_test, gb_test_predcition_proba, gb_test_predictions, y_test,
    #              os.path.join(output_path, 'gb_model_test_results_knn_within_class_imputed_data.csv'))
    save_results(sample_id_test, rf_test_predcition_proba, rf_test_predictions, y_test,
                 os.path.join(output_path, rf_output_csv_name))
    save_results(sample_id_test, gb_test_predcition_proba, gb_test_predictions, y_test,
                 os.path.join(output_path, gb_output_csv_name))
    # save_results(sample_id_test, rf_test_predcition_proba, rf_test_predictions, y_test, os.path.join(
    #     output_path, 'rf_model_test_results_mean_imputed_data.csv'))
    # save_results(sample_id_test, gb_test_predcition_proba, gb_test_predictions, y_test, os.path.join(
    #     output_path, 'gb_model_test_results_mean_imputed_data.csv'))

    print("Random Forest accuracy on validation set: ", rf_val_accuracy)
    print("Gradient Boosting accuracy on validation set: ", gb_val_accuracy)
    print("Random Forest classification report on validation set:\n", rf_val_report)
    print("Gradient Boosting classification report on validation set:\n", gb_val_report)
    print("Random Forest confusion matrix on validation set:\n", rf_val_cm)
    print("Gradient Boosting confusion matrix on validation set:\n", gb_val_cm)

    # Print test results
    print("Random Forest accuracy on test set: ", rf_test_accuracy)
    print("Gradient Boosting accuracy on test set: ", gb_test_accuracy)
    print("Random Forest classification report on test set:\n", rf_test_report)
    print("Gradient Boosting classification report on test set:\n", gb_test_report)
    print("Random Forest confusion matrix on test set:\n", rf_test_cm)
    print("Gradient Boosting confusion matrix on test set:\n", gb_test_cm)


if __name__ == '__main__':
    main()
