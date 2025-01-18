import numpy as np
import os
import subprocess
import pathlib
import pandas as pd
import sys
import argparse

def has_header(filename):
    """Check if the given CSV file has a header with an 'img_path' column."""
    try:
        # Try to read the first row as header
        df = pd.read_csv(filename, nrows=1)
    except pd.errors.ParserError:
        # If it can't read the file for some reason, return None
        return None

    # The header is assumed to be a string, so if any column has a non-string
    # data type, we assume it's not a header.
    if 'img_path' in df.columns:
        return True
    else:
        return False

def get_case_list(list_file):
    """Get the list of cases from either a CSV or TXT file."""
    if list_file.endswith('.txt'):
        with open(list_file) as f:
            return [x.strip() for x in f.readlines() if not x.startswith('#')]
    else:
        has_header_switch = has_header(list_file)
        if has_header_switch:
            df = pd.read_csv(list_file)
            return df['img_path'].tolist()
        else:
            df = pd.read_csv(list_file, header=None)
            return df[0].tolist()

def run_total_segmentator(case_list, root_dir):
    """Run the TotalSegmentator command for each case in the case list."""
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for case in case_list:
        filename = case.split('/')[-1]
        filename_stem = filename.split('.')[0]
        foldername = case.split('/')[-2]

        output_path = os.path.join(root_dir, foldername, 'totalsegmentation_'+filename_stem)

        if os.path.exists(output_path):
            print(f"Output path {output_path} already exists. Skipping.")
            continue

        command = f"TotalSegmentator -i {case} -o {output_path} --statistics"
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()
        if error:
            print(f"Error in processing {case}: {error}")
        else:
            print(f"Processed {case} successfully.")

def main():
    parser = argparse.ArgumentParser(description="Run TotalSegmentator on a list of files.")
    parser.add_argument("list_file", nargs='?', default='/data/drdcad/lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/dataset_CTC_3m/test_set_CTC_3m_biowulf_path.csv', help="Path to the file containing a list of image paths.")
    parser.add_argument("root_dir", nargs='?', default='/data/drdcad/lance/dataset/CTC/CTC_3mm_only_keep_key_organs_radiomics', help="Root directory where output will be saved.")
    # list_file = '/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/test_files_linux.txt'
    # list_file = '/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/test_set_linux_respacing113.csv'
    # list_file = '/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/train_files_linux.txt'
    # list_file = '/home/lance/NAS/users/Akshaya/deepLesion_stuff/deep_lesion_test_phase_annotations_valid_original_scan.csv'
    # list_file = '/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/5Phase_all.csv'
    # list_file = '/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/test_set_DeepLesion509_original_scan_respacing113.csv'
    #list_file = '/data/drdcad/lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/dataset_CTC_3m/test_set_CTC_3m_biowulf_path.csv'

    # root_dir = '/home/lance/NAS/users/Akshaya/My_Datasets/DeepLesion509_original_scan_key_organs'
    # root_dir = '/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics'
    # root_dir = '/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/DeepLesion509_original_scan_only_keep_key_organs_radiomics_respacing113'
    #root_dir = '/data/drdcad/lance/dataset/CTC/CTC_3mm_only_keep_key_organs_radiomics'

    args = parser.parse_args()
    list_file = args.list_file
    root_dir = args.root_dir

    # Get case list from the input file
    case_list = get_case_list(list_file)

    # Run TotalSegmentator for each case in the case list
    run_total_segmentator(case_list, root_dir)

if __name__ == "__main__":
    main()
