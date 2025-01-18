# Intensity train and inference

This code is for training and inference of the intensity model. The intensity model is used to classify the contrast phase of the CT scans. The model is trained on the radiomics features of the key organs segmented by the TotalSegmentator. The model is trained on the contrast phase dataset and tested on the CTC dataset.

# FILEPATH: contrast_phase_code/readme_intensity_train_inference.md

"file" is the python file name for the project (all in the zip file)
"path" in each step indicates the original path of the python file
the paths the python file "path" in each step are the demo paths to the data and models on biowulf

<!-- # step 1 preprocessing data (only for CTC data, substract 1024 in the intensity)
**activate the TotalSegmentator environment and run the file: adjust_intensity_CTC.py**

file: adjust_intensity_CTC.py
path: /data/drdcad/lance/PycharmProjects/PycharmProjects/TotalSegmentator/bin/adjust_intensity_CTC.py

list_file = '/data/drdcad/lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/dataset_CTC_3m/test_set_CTC_3m_biowulf_path.csv'

root_dir = '/data/drdcad/lance/dataset/CTC/CTC_3mm_adjusted_intensity-1024' -->
# step 0 install the required packages for TotalSegmentator and monai environment

create the TotalSegmentator environment and install TotalSegmentator according to installation part in <https://github.com/StanfordMIMI/TotalSegmentator>
create the monai environment and install monai according to installation part in <https://github.com/Project-MONAI/MONAI>

# step 1 use TotalSegmentator to segment the key organs

**Activate the TotalSegmentator environment and run the file:**
run_in_batch_tts.py**

file: run_in_batch_tts.py
path: contrast_phase_code/run_in_batch_tts.py

Change the list_file and root_dir to the correct path you want to segment, for example:
list_file = 'contrast_phase_code/dataset/test_set_CTC_3m_adjusted_intensity-1024_biowulf_path.csv'
root_dir = 'contrast_phase_code/dataset/CTC_3mm_adjusted_intensity-1024_only_keep_key_organs_radiomics'

The TotalSegmentator will output the key organs in the folder "root_dir"
(you can also consider using the batch process in a cluster to submit the jobs of TotalSegmentator and process the segmentation in parallel. you need to write bash script (like swarm command) according to this line:
TotalSegmentator -i {case} -o {output_path} --statistics, and {case}, {output_path} should be modified to your own situation.)

# step 2 get the statistics of the key organs

**Activate the monai environment and run the file: get_statistics.py**

file: get_statistics.py
path: contrast_phase_code/get_statistics.py

Change the csv_file_path (the list of file paths to the scans) and output_path (the csv file to save the radiomics of the scans) to the correct path you want to get the statistics, for example:

csv_test_path = "/data/drdcad/lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/dataset_CTC_3m/test_set_CTC_3m_adjusted_intensity-1024_biowulf_path.csv"
output_base = "/data/drdcad/lance/dataset/CTC/CTC_3mm_adjusted_intensity-1024_only_keep_key_organs_radiomics"
csv_train_path = "/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics.csv"

Example usage:
example 1
    python get_statistics.py --cc_deeplesion \
    --csv_train_path "/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics.csv" \
    --csv_label_path "/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_label_respacing113_radiomics.csv" \
    --model "test" \
    --method "knn_within_class" \
    --n_neighbors 1 \
    --output_path "/output/path/cc_deeplesion_test_knn_within_class_imputed.csv"

# step 3 intensity train

**Activate the monai environment and run the file: intensity_train.py to train the models based on intensity. Evaluation step will also be performed. If it is not needed, the test path can be replaced by the validation path.**

file: intensity_train.py
path: contrast_phase_code/intensity_train.py

change the path to the correct path you want to save the model and the output, for example:
output_path_root = '/data/drdcad/lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/intensity_trained_models/'
output_folder_for_this_run path is for particular run under the output_path_root path, change it accordingly. 
e.g.:
output_folder_for_this_run = 'intensity_trained_models_and_results'+timestamp

change the path to our own training, validation, and testing data from previous step, for example:
X_train, y_train, _= load_data('/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/dataset/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics_intensity_w_hu_std_knn_imputed.csv')
X_val, y_val,_ = load_data('/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/dataset/Attempt2_only_keep_key_organs_radiomics_respacing113/validation_set_respacing113_radiomics_intensity_w_hu_std_knn_imputed.csv')
X_test, y_test, sample_id_test = load_data('/data/drdcad/lance/dataset/CTC/CTC_3mm_adjusted_intensity-1024_only_keep_key_organs_radiomics/test_set_CTC_3m_adjusted_intensity-1024_biowulf_path_radiomics_mean_imputed.csv')  # Load test data

make a reasonable model saving name and path
rf_model_name = 'rf_model_knn_imputed_data'
gb_model_name = 'gb_model_knn_imputed_data'

make a reasonable name for your output prediction csv
rf_output_csv_name = 'rf_model_test_results_knn_imputed_data.csv'
gb_output_csv_name = 'gb_model_test_results_knn_imputed_data.csv'

the output model will be saved in the folder: output_path
the output prediction will also be saved in the folder: output_path
the evaluation results will be reported in the console.

# step 4 intensity inference

**Activate the monai environment and run the file: intensity_inference.py for the contrast phase classification and evaluation**

file: intensity_inference.py
path: /data/drdcad/lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/intensity_inference.py

change the path to your imputed data, for example:
X_test, y_test, sample_id_test = load_data("/data/drdcad/lance/dataset/CTC/CTC_3mm_adjusted_intensity-1024_only_keep_key_organs_radiomics/test_set_CTC_3m_adjusted_intensity-1024_biowulf_path_radiomics_mean_imputed.csv")

change the model_and_output_path to your saved model, for example:
model_and_output_path = '/data/drdcad/lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/intensity_trained_models/intensity_trained_models_and_results20230926-000025'

change model name for your own model, for example:
rf_model_name = 'rf_model_knn_within_class_imputed_data.joblib'
gb_model_name = 'gb_model_knn_within_class_imputed_data.joblib'

make a reasonable name for your output prediction csv, for example:
rf_output_csv_name = 'rf_test_results_with_proba_deeplesion_full_scan_1nn_within_class.csv'
gb_output_csv_name = 'gb_test_results_with_proba_deeplesion_full_scan_1nn_within_class.csv'

the output prediction will be saved in the folder: <model_and_output_path>/<rf_output_csv_name> or <model_and_output_path>/<gb_output_csv_name>
the evaluation results will be reported in the console.
