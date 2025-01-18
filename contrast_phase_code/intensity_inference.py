import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from intensity_train import load_data, evaluate_model, save_results


def load_model(model_path):
    return load(model_path)


def write_results_to_csv(results, csv_path):
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)


def main():
    # X_train, y_train, _ = load_data('/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics_mean_imputed.csv')
    # X_val, y_val, _ = load_data('/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/validation_set_respacing113_radiomics_mean_imputed.csv')

    # !!! change validation csv path for your own validation set
    X_val, y_val, _ = load_data('dataset/Attempt2_only_keep_key_organs_radiomics_respacing113/validation_set_respacing113_radiomics_intensity_w_hu_std_mean_imputed.csv')

    # !!! validation path exmaple:
    # X_val, y_val, _ = load_data(
    #     '/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/validation_set_respacing113_radiomics_intensity_w_hu_std_mean_imputed.csv')

    # !!! change test csv path for your own imputed test set
    # *** for NIH CC dataset
    X_test, y_test, sample_id_test = load_data(
        "dataset/Attempt2_only_keep_key_organs_radiomics_respacing113/test_set_respacing113_radiomics_intensity_w_hu_std_mean_imputed.csv")  # Load test data

    # !!! test path example:
    # X_test, y_test, sample_id_test = load_data(
    #     "/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/test_set_respacing113_radiomics_intensity_w_hu_std_mean_imputed.csv")  # Load test data

    # X_test, y_test, sample_id_test = load_data("/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/test_set_respacing113_radiomics_mean_imputed.csv")  # Load test data
    


    # output_path = 'intensity_trained_models/'
    # output_path = 'intensity_trained_models/intensity_trained_models_and_results20230926-000025'
    # output_path = 'intensity_trained_models/intensity_trained_models_and_results_knn_20231017-024034'
    # output_path = 'intensity_trained_models/intensity_trained_models_and_results_mean20240423-040201'

    #!! change model and output path for your own model
    model_and_output_path = 'contrast_phase_code/intensity_trained_models/intensity_trained_models_and_results_knn_within_class20240401-011939'
    #!! model and output path example:
    # model_and_output_path = 'intensity_trained_models/intensity_trained_models_and_results_knn_within_class20240401-011939'
    
    #!! change model name for your own trained model
    rf_model_name = 'rf_model_knn_within_class_imputed_data.joblib'
    gb_model_name = 'gb_model_knn_within_class_imputed_data.joblib'
    
    #!! make a reasonable name for your output prediction csv
    # example:  
    rf_output_csv_name = 'rf_test_results_with_proba_deeplesion_full_scan_1nn_within_class.csv'    
    gb_output_csv_name = 'gb_test_results_with_proba_deeplesion_full_scan_1nn_within_class.csv' 


    # rf_model = RandomForestClassifier(random_state=42)
    # gb_model = GradientBoostingClassifier(random_state=42)

    # rf_model = load_model(output_path + 'rf_model.joblib')
    # gb_model = load_model(output_path + 'gb_model.joblib')

    rf_model = load_model(os.path.join(
    model_and_output_path, rf_model_name))
    gb_model = load_model(os.path.join(
    model_and_output_path, gb_model_name))
    # rf_model = load_model(os.path.join(output_path, 'rf_model_fold_1.joblib'))
    # gb_model = load_model(os.path.join(output_path, 'gb_model_fold_1.joblib'))
    # rf_model = load_model(os.path.join(
    #     output_path, 'rf_model_knn_imputed_data.joblib'))
    # gb_model = load_model(os.path.join(
    #     output_path, 'gb_model_knn_imputed_data.joblib'))
    # rf_model = load_model(os.path.join(
    #     output_path, 'rf_model_mean_imputed_data.joblib'))
    # gb_model = load_model(os.path.join(
    #     output_path, 'gb_model_mean_imputed_data.joblib'))

    rf_val_accuracy, rf_val_report, rf_val_cm, * \
        _ = evaluate_model(rf_model, X_val, y_val)
    gb_val_accuracy, gb_val_report, gb_val_cm, * \
        _ = evaluate_model(gb_model, X_val, y_val)

    # Evaluate models on the test set
    rf_test_accuracy, rf_test_report, rf_test_cm, rf_test_prediction, rf_test_prediction_proba = evaluate_model(
        rf_model, X_test, y_test)
    gb_test_accuracy, gb_test_report, gb_test_cm, gb_test_prediction, gb_test_prediction_proba = evaluate_model(
        gb_model, X_test, y_test)


    save_results(sample_id_test, rf_test_prediction_proba, rf_test_prediction, y_test, os.path.join(
        model_and_output_path, rf_output_csv_name))  
    save_results(sample_id_test, gb_test_prediction_proba, gb_test_prediction, y_test, os.path.join(
        model_and_output_path, gb_output_csv_name))


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

    #     # Collect results and write to CSV
    # results = {
    #     'Model': ['Random Forest', 'Gradient Boosting'],
    #     'Validation Accuracy': [rf_val_accuracy, gb_val_accuracy],
    #     'Validation Report': [rf_val_report, gb_val_report],
    #     'Validation Confusion Matrix': [rf_val_cm, gb_val_cm],
    #     'Test Accuracy': [rf_test_accuracy, gb_test_accuracy],
    #     'Test Report': [rf_test_report, gb_test_report],
    #     'Test Confusion Matrix': [rf_test_cm, gb_test_cm]
    # }
    # write_results_to_csv(results, output_path + 'results.csv')


if __name__ == '__main__':
    main()
