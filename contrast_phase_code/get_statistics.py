"""
This module provides functions to extract and process statistics from a JSON file.

Functions:
- extract_intensity_values(json_file_path: str) -> pd.DataFrame:
    Extracts the intensity values from a JSON file and returns them as a pandas DataFrame.

- knn_imputation(df: pd.DataFrame, columns: List[str], n_neighbors: int = 5, imputer: Optional[KNNImputer] = None) -> Tuple[pd.DataFrame, KNNImputer]:
    Performs k-nearest neighbors imputation on specified columns of a DataFrame and returns the imputed DataFrame and the KNNImputer object used for imputation.

- mean_imputation(df: pd.DataFrame, columns: List[str], means: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    Performs mean imputation on specified columns of a DataFrame and returns the imputed DataFrame.

Example usage:
example 1
    python script.py --cc_deeplesion \
    --csv_train_path "/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics.csv" \
    --csv_label_path "/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_label_respacing113_radiomics.csv" \
    --model "test" \
    --method "knn_within_class" \
    --n_neighbors 1 \
    --output_path "/output/path/cc_deeplesion_test_knn_within_class_imputed.csv"

example 2
    python script.py --vindata \
    --csv_train_path "/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics.csv" \
    --csv_label_path "/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_label_respacing113_radiomics.csv" \
    --model "val" \
    --method "knn" \
    --n_neighbors 5 \
    --output_path "/output/path/vindata_val_knn_imputed.csv"

"""
import argparse
from typing import List, Optional, Dict, Tuple
import glob
import json
import pandas as pd
import os
import numpy as np
from sklearn.impute import KNNImputer


def extract_intensity_values(json_path):
    # Load the JSON file
    with open(json_path) as file:
        data = json.load(file)

    # Define the keys
    keys = [
        "iliac_artery_left",
        "iliac_artery_right",
        "pulmonary_artery",
        "aorta",
        "kidney_left",
        "kidney_right",
        "heart_myocardium",
        "heart_atrium_left",
        "heart_atrium_right",
        "heart_ventricle_left",
        "heart_ventricle_right",
        "portal_vein_and_splenic_vein",
        "liver",
        "urinary_bladder",
        "inferior_vena_cava",
        "iliac_vena_left",
        "iliac_vena_right",
    ]

    # Get the value of the "intensity" subkey for each key
    intensity_values = {key: data[key]["intensity"]
                        if "intensity" in data[key] else None for key in keys}

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(intensity_values, index=[0])

    return df


def knn_imputation(df, columns, n_neighbors=5, within_class=False, imputer=None):
    """
    Perform k-nearest neighbors imputation on specified columns of a DataFrame.

    Args:
        df (_type_): The DataFrame on which to perform knn imputation.
        columns (_type_): A list of column names on which to perform knn imputation.
        n_neighbors (int, optional): The number of neighbors to form as well as the number of centroids to generate. Defaults to 3.
    """
    df[columns] = df[columns].replace(0, np.nan)

    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df[columns].dropna())
    # df[columns] = df[columns].fillna(kmeans.cluster_centers_[kmeans.predict(df[columns])])

    if imputer is None:
        if within_class == False:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputer.set_output(transform="pandas")
            imputer.fit(df[columns])
            filled_df = imputer.fit_transform(df[columns])
        else:
            imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')
            imputer.set_output(transform="pandas")
            imputer.fit(df[columns])
            filled_df = imputer.fit_transform(df[columns])
    else:
        filled_df = imputer.transform(df[columns])

    filled_df = pd.concat([df.iloc[:, 0], filled_df], axis=1)
    return filled_df, imputer

# Usage
# df = extract_intensity_values("/path/to/your/file/statistics.json")
# print(df)


def load_and_merge_labels(df, labels_filepath):
    """
    Load class labels from a separate file and merge them into the main DataFrame.

    Parameters:
    - df: pandas DataFrame, the main dataset with filenames as the first column.
    - labels_filepath: str, the path to the labels file, which has filenames and labels.

    Returns:
    - pandas DataFrame with class labels merged.
    """
    # Load labels
    # Assuming filenames and labels are the first two columns
    labels_df = pd.read_csv(labels_filepath, usecols=[0, 1])
    # Rename columns for clarity
    labels_df.columns = ['fullpath', 'GT']

    # Merge the class labels into the main DataFrame based on filenames
    merged_df = pd.merge(df, labels_df, on='fullpath', how='left')

    return merged_df


def knn_imputation_within_class(df: pd.DataFrame, columns: List[str], train_labels_filepath: str, n_neighbors: int = 5, df_imputed_train: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    # def knn_imputation_within_class(df, columns, labels_filepath, n_neighbors=5, df_imputed_train=None):
    """
    Perform KNN imputation within each class of entries, with class labels coming from a separate file.

    Parameters:
    - df: pandas DataFrame, the dataset with rows as entries and columns as features. The first column is 'filename'.
    - labels_filepath: str, the path to the labels file, which has filenames and class labels.
    - n_neighbors: int, number of neighbors to use for KNN imputation.

    Returns:
    - pandas DataFrame with imputed values, imputation performed within each class.
    """
    df[columns] = df[columns].replace(0, np.nan)

    # Initialize the KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='distance')

    # Empty DataFrame to hold the imputed data, preserving the fullpath and class label columns
    imputed_data = pd.DataFrame(
        columns=['fullpath'] + list(df.columns[1:]) + ['GT'])

    if df_imputed_train is None:
        # Load and merge class labels
        df_with_labels = load_and_merge_labels(df, train_labels_filepath)

       # Separate features and class labels
        X = df_with_labels.drop(columns=['fullpath', 'GT'])
        y = df_with_labels['GT']

        # Iterate over each class, perform KNN imputation, and concatenate the results
        for GT in y.unique():
            # Subset the data for the current class
            subset_X = X[y == GT]
            # Preserve fullpaths for merging
            subset_fullpaths = df_with_labels['fullpath'][y == GT]

            # Perform KNN imputation on the subset
            imputed_subset = imputer.fit_transform(subset_X)

            # Convert the imputed numpy array back to a DataFrame
            imputed_subset_df = pd.DataFrame(imputed_subset, columns=X.columns)
            # Add fullpaths back
            imputed_subset_df.insert(0, 'fullpath', subset_fullpaths)
            # imputed_subset_df['fullpath'] = subset_fullpaths.values
            imputed_subset_df['GT'] = GT  # Add class labels back

            # Concatenate the imputed subset back to the full DataFrame
            imputed_data = pd.concat(
                [imputed_data, imputed_subset_df], ignore_index=True)
    else:
        imputer.fit(df_imputed_train[columns])
        imputed_data = imputer.transform(df[columns])
        imputed_data = pd.DataFrame(
            imputed_data, columns=columns)

    # Ensure the class label column is in the same type as in the original DataFrame
    if 'fullpath' not in imputed_data.columns:
        imputed_data.insert(0, 'fullpath', df['fullpath'])
    else:
        print("fullpath already exists in the DataFrame")
    # imputed_data['fullpath'] = df['fullpath']
    # imputed_data['GT'] = df['GT'].astype(y.dtype)

    return imputed_data


def mean_imputation(df, columns, means=None):
    """
    Perform mean imputation on specified columns of a DataFrame.
    Assumes missing values are represented as 0, and replaces these with NaN before imputation.

    Parameters:
    df (pd.DataFrame): The DataFrame on which to perform mean imputation.
    columns (list): A list of column names on which to perform mean imputation.

    Returns:
    pd.DataFrame: The DataFrame after mean imputation.
    """
    # if 'GT' in df.columns:
    #     df = df.drop(columns=['GT'])
    df[columns] = df[columns].replace(0, np.nan)

    if means is None:
        filled_df = df[columns].fillna(df[columns].mean())
        return filled_df, df[columns].mean()
    if means is not None:
        return df.fillna(means), means


def imputation_main(csv_train_path, model="train", csv_train_label_path=None, csv_val_test_path=None, method="mean", *args, **kwargs):
    """
    Perform mean imputation on the given csv file and output.
    use the mean values of the training set to impute the validation and test set.
    """
    # csv_train_path = "/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics.csv"
    for key, value in kwargs.items():
        if key == "n_neighbors":
            n_neighbors = value

    df = pd.read_csv(csv_train_path)
    if 'GT' in df.columns:
        df = df.drop(columns=['GT'])
    if method == "mean":
        df, filled_value = mean_imputation(df, df.columns[1:])
    elif method == "knn":
        df, imputer = knn_imputation(df, df.columns[1:], n_neighbors=5)
    elif method == "knn_within_class":
        df = knn_imputation_within_class(
            df, df.columns[1:], csv_train_label_path, n_neighbors=1)
    else:
        print("Wrong imputation method, please check")
        return

    # df_output_path = "/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics_mean_imputed.csv"

    if model == "train":
        df_output_path = os.path.splitext(csv_train_path)[
            0]+"_"+method+"_imputed.csv"
        if os.path.exists(df_output_path):
            print("File already exists: ", df_output_path)
        else:
            df.to_csv(df_output_path, index=False)
        return
    elif (model == "val") or (model == "test"):
        if csv_val_test_path is None:
            print("Please provide the validation or test csv file path")
            return
        df_val_test = pd.read_csv(csv_val_test_path)
        if 'GT' in df_val_test.columns:
            df_val_test_columns = df_val_test.columns[1:-1]
        else:
            df_val_test_columns = df_val_test.columns[1:]
        if method == "mean":
            df_val_test, _ = mean_imputation(
                df_val_test, df_val_test_columns, filled_value)
        elif method == "knn":
            df_val_test, _ = knn_imputation(
                df_val_test, df_val_test_columns, imputer=imputer)
        elif method == "knn_within_class":
            df_val_test = knn_imputation_within_class(
                df_val_test, df_val_test_columns, csv_train_label_path, n_neighbors=1, df_imputed_train=df)
        df_val_test.to_csv(csv_val_test_path.replace(
            ".csv", "_"+method+"_imputed.csv"), index=False)
        #!!!!!! Please manually add the GT after the last column
        print("!!! Please manually add the GT after the last column, if there are none")
    else:
        print("Wrong model name, please check")
        return

    # csv_val_path = "/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/validation_set_respacing113_radiomics.csv"
    # csv_test_path = "/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/test_set_respacing113_radiomics.csv"
    # df_val = pd.read_csv(csv_val_path)
    # df_val, _ = mean_imputation(df_val, df_val.columns[1:], df_mean)
    # df_val.to_csv("/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/validation_set_respacing113_radiomics_mean_imputed.csv", index=False)
    # df_test = pd.read_csv(csv_test_path)
    # df_test, _ = mean_imputation(df_test, df_test.columns[1:], df_mean)
    # df_test.to_csv("/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/test_set_respacing113_radiomics_mean_imputed.csv", index=False)
    return


def has_header(filename):
    df = pd.read_csv(filename, nrows=1)  # Read only the first row
    return 'img_path' in df.columns or 'filename' in df.columns


def main_cc_deeplesion(csv_paths_list: str, output_path: str):
    """
    get the radiomics from the samples in the given csv file and the prepared statistics.json from tts.

    Returns:
        str: The path to the output CSV file.

    Raises:
        FileNotFoundError: If any of the input JSON files do not exist.

    Example usage:
        # Extract radiomics features from the samples in the given CSV file and the prepared statistics.json from tts
        output_path = main()

    """
    # csv_paths = pd.read_csv("/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/train_set_respacing113.csv", header=None)
    # csv_paths = pd.read_csv("/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/validation_set_respacing113.csv", header=None)
    # csv_paths = pd.read_csv("/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/test_set_respacing113.csv", header=None)
    # csv_paths = pd.read_csv(
    # "/data/drdcad/lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/test_set_respacing113.csv", header=None)

    # csv_paths_list = "/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/validation_set_respacing113.csv"
    # csv_paths_list = "/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/test_set_DeepLesion509_original_scan_respacing113.csv"

    # output_path = "/home/lance/NAS/users/Akshaya/My_Datasets/DeepLesion509_original_scan_only_keep_key_organs_radiomics_respacing113/DeepLesion509_original_scan_only_keep_key_organs_radiomics_respacing113_radiomics.csv"
    # output_path = "/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/validation_set_respacing113_radiomics.csv"
    # output_path = "/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/test_set_respacing113_radiomics.csv"
    if os.path.exists(output_path):
        print("File already exists: ", output_path)
        return output_path

    if not has_header(csv_paths_list):
        csv_paths = pd.read_csv(csv_paths_list, header=None)
    else:
        csv_paths = pd.read_csv(csv_paths_list, header=0)

    df_full = pd.DataFrame()
    for one_path in csv_paths.iloc[:, 0]:
        # json_path = one_path.replace("Attempt2_respacing113", "Attempt2_only_keep_key_organs_radiomics_respacing113")
        json_path = one_path.replace("DeepLesion509_original_scan_respacing113",
                                     "DeepLesion509_original_scan_only_keep_key_organs_radiomics_respacing113/tts_results")
        json_path = os.path.join(os.path.dirname(
            json_path), "totalsegmentation_" + os.path.basename(json_path).split('.')[0], "statistics.json")
        if not os.path.exists(json_path):
            print("File not exists: ", json_path)
            continue
        else:
            df = extract_intensity_values(json_path)
            df.insert(0, 'fullpath', one_path)
            df_full = pd.concat([df_full, df], ignore_index=True)
            # df_full = df_full.append(df, ignore_index=True) # this is removed in new version of pandas

    print(df_full)

    # df_full.to_csv("/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics.csv", index=False)
    # df_full.to_csv("/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/validation_set_respacing113_radiomics.csv", index=False)
    # df_full.to_csv("/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/test_set_respacing113_radiomics.csv", index=False)
    df_full.to_csv(output_path, index=False)
    #!!!!!! Please manually add the GT after the last column
    print("!!! Please manually add the GT after the last column")
    return output_path


def main_CTC(csv_paths_list, output_base):
    """
    get the radiomics from the samples in the given csv file and the prepared statistics.json from tts.
    """

    # output_base = "/data/drdcad/lance/dataset/CTC/CTC_3mm_only_keep_key_organs_radiomics"
    output_csv = os.path.basename(csv_paths_list)
    output_csv = output_csv.replace(".csv", "_radiomics.csv")
    output_path = os.path.join(output_base, output_csv)
    if os.path.exists(output_path):
        print("File already exists: ", output_path)
        return output_path

    if not has_header(csv_paths_list):
        csv_paths = pd.read_csv(csv_paths_list, header=None)
    else:
        csv_paths = pd.read_csv(csv_paths_list, header=0)

    df_full = pd.DataFrame()
    for one_path in csv_paths.iloc[:, 0]:
        direct_dirname = os.path.dirname(one_path).split("/")[-1]
        json_path = one_path.replace(
            direct_dirname, direct_dirname+"_only_keep_key_organs_radiomics/"+direct_dirname)
        json_path = os.path.join(os.path.dirname(
            json_path), "totalsegmentation_" + os.path.basename(json_path).split('.')[0], "statistics.json")
        if not os.path.exists(json_path):
            print("File not exists: ", json_path)
            continue
        else:
            df = extract_intensity_values(json_path)
            df.insert(0, 'fullpath', one_path)
            print(type(df))
            print(type(df_full))
            # print(type(df_full))
            df_full = pd.concat([df_full, df], ignore_index=True)
            # df_full = df_full.append(df, ignore_index=True) # this is removed in new version of pandas
    print(df_full)
    df_full.to_csv(output_path, index=False)
    #!!!!!! Please manually add the GT after the last column
    print("!!! Please manually add the GT after the last column")
    return output_path


def main_vindata(csv_paths_list, output_base):
    """
    get the radiomics from the samples in the given csv file and the prepared statistics.json from tts.
    """

    # output_base = "/data/drdcad/lance/dataset/CTC/CTC_3mm_only_keep_key_organs_radiomics"
    output_csv = os.path.basename(csv_paths_list)
    output_csv = output_csv.replace(".csv", "_radiomics.csv")
    output_path = os.path.join(output_base, output_csv)
    if os.path.exists(output_path):
        print("File already exists: ", output_path)
        return output_path

    if not has_header(csv_paths_list):
        csv_paths = pd.read_csv(csv_paths_list, header=None)
    else:
        csv_paths = pd.read_csv(csv_paths_list, header=0)

    df_full = pd.DataFrame()
    for one_path in csv_paths.iloc[:, 0]:
        direct_dirname = os.path.dirname(one_path).split("/")[-1]

        json_path = one_path.replace(direct_dirname, direct_dirname+"_tts")
        json_path = os.path.join(os.path.dirname(json_path), os.path.basename(
            json_path).split('.')[0], "statistics.json")
        if not os.path.exists(json_path):
            print("File not exists: ", json_path)
            continue
        else:
            df = extract_intensity_values(json_path)
            df.insert(0, 'fullpath', one_path)
            print(type(df))
            print(type(df_full))
            # print(type(df_full))
            df_full = pd.concat([df_full, df], ignore_index=True)
            # df_full = df_full.append(df, ignore_index=True) # this is removed in new version of pandas
    print(df_full)
    df_full.to_csv(output_path, index=False)
    #!!!!!! Please manually add the GT after the last column
    print("!!! Please manually add the GT after the last column")
    return output_path


if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Process dataset for radiomics extraction and imputation.",
                                     usage="python get_statistics.py [options]")
    
    # Define mutually exclusive group for choosing main function
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--cc_deeplesion', action='store_true', help="Use main_cc_deeplesion function.")
    group.add_argument('--vindata', action='store_true', help="Use main_vindata function.")
    group.add_argument('--ctc', action='store_true', help="Use main_CTC function.")
    
    # Common arguments that will be used for all the methods
    parser.add_argument('--csv_path', type=str, default="contrast_phase_code/dataset/test_set_respacing113.csv",
                        help="Path to the CSV file for radiomics processing. e.g.: '/data/drdcad/lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/test_set_respacing113.csv'.")

    # Optional arguments with default values based on selected function
    parser.add_argument('--output_path', type=str, default="contrast_phase_code/dataset/DeepLesion509_original_scan_only_keep_key_organs_radiomics_respacing113/DeepLesion509_original_scan_only_keep_key_organs_radiomics_respacing113_radiomics.csv",
                        help="Output path for cc_deeplesion method. e.g.: '/data/drdcad/lance/dataset/contrast_phase_dataset/vin_abdomenContrastPhases_26Feb24/nifti_3mm_only_keep_key_organs_radiomics/test_set_vindata_radiomics_intensity_w_hu_std.csv'.")

    parser.add_argument('--output_base', type=str, default="contrast_phase_code/dataset/vin_abdomenContrastPhases_26Feb24/nifti_3mm_only_keep_key_organs_radiomics",
                        help="Output base directory for vindata and CTC methods. e.g.: '/data/drdcad/lance/dataset/contrast_phase_dataset/vin_abdomenContrastPhases_26Feb24/nifti_3mm_only_keep_key_organs_radiomics'.")

    # Optional arguments for imputation
    parser.add_argument('--csv_train_path', type=str, required=True, default="contrast_phase_code/dataset/train_set_respacing113_radiomics.csv",
                        help="Path to the training CSV file for imputation. e.g.: '/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics.csv'.")
    
    parser.add_argument('--csv_label_path', type=str, required=False, default="contrast_phase_code/dataset/train_label_respacing113_radiomics.csv",
                        help="Path to the label CSV file for imputation within class. e.g.: '/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_label_respacing113_radiomics.csv'.")
    
    parser.add_argument('--model', type=str, default="test", choices=['train', 'val', 'test'],
                        help="Specify whether this is for training, validation, or test. Default: 'train'.")
    
    parser.add_argument('--method', type=str, default="mean", choices=['mean', 'knn', 'knn_within_class'],
                        help="Specify the imputation method. Default: 'mean'.")
    
    parser.add_argument('--n_neighbors', type=int, default=5, help="Number of neighbors for KNN imputation. Default: 5.")

    # Parse the arguments
    args = parser.parse_args()

    # Choose main function and set the corresponding arguments
    if args.cc_deeplesion:
        main = main_cc_deeplesion
        output = main(csv_paths_list=args.csv_path, output_path=args.output_path)

    elif args.vindata:
        main = main_vindata
        output = main(csv_paths_list=args.csv_path, output_base=args.output_base)

    elif args.ctc:
        main = main_CTC
        output = main(csv_paths_list=args.csv_path, output_base=args.output_base)

    print("Processing completed. Output saved at:", output)

    # csv_train_path = "/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_set_respacing113_radiomics_intensity_w_hu_std.csv"
    # csv_label_path = "/data/drdcad/lance/dataset/contrast_phase_dataset/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics_respacing113/train_label_respacing113_radiomics.csv"

    # imputation_main(csv_train_path, model='train', csv_train_label_path=csv_label_path,
    # method='mean', n_neighbors=1)
    # imputation_main(csv_train_path, model='test', csv_val_test_path=output_path)
    # imputation_main(csv_train_path, model='test',
    #                 csv_val_test_path=output_path, method='mean')
    # imputation_main(csv_train_path, model='test',
    #                 csv_val_test_path=output_path, method='knn', n_neighbors=5)
    # # imputation_main(csv_train_path, model='val', csv_val_test_path=output_path, method='knn')
    # imputation_main(csv_train_path, model='test',
    # csv_val_test_path=output, csv_train_label_path=csv_label_path, method='knn_within_class', n_neighbors=1)

        # Call the imputation function based on the user-specified model, method, and other arguments
    imputation_main(
        csv_train_path=args.csv_train_path,
        csv_train_label_path=args.csv_label_path,
        csv_val_test_path=args.output_path,
        model=args.model,
        method=args.method,
        n_neighbors=args.n_neighbors
    )
    print("Processing completed. Output saved at:", output)
 
    
    
    
    
    
    
