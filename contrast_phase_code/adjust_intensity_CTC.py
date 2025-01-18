import numpy as np
import os
import subprocess
import pathlib
import pandas as pd
import sys
import pandas as pd
import nibabel as nib

def has_header(filename):
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
# Get the path of the current Python interpreter
python_exec_path = sys.executable

# Get the directory of the current Python interpreter
python_dir = os.path.dirname(python_exec_path)

# Add the Python directory to the PATH
os.environ["PATH"] = python_dir + ":" + os.getenv("PATH")

# list_file = '/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/test_files_linux.txt'
# list_file = '/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/test_set_linux_respacing113.csv'
# list_file = '/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/train_files_linux.txt'
# list_file = '/home/lance/NAS/users/Akshaya/deepLesion_stuff/deep_lesion_test_phase_annotations_valid_original_scan.csv'
# list_file = '/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/5Phase_all.csv'
list_file = '/home/lance/NAS/users/Lance/PycharmProjects/Contrast_Phase_Classifier/monai_3d_classification/test_set_DeepLesion509_original_scan_respacing113.csv'

# root_dir = '/home/lance/NAS/users/Akshaya/My_Datasets/DeepLesion509_original_scan_key_organs'
# root_dir = '/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_only_keep_key_organs_radiomics'
root_dir = '/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/DeepLesion509_original_scan_only_keep_key_organs_radiomics_respacing113'

if not os.path.exists(root_dir):
    os.makedirs(root_dir)

# check if txt or csv extension and read accordingly
if list_file.endswith('.txt'):
    case_list = []
    with open(list_file) as f:
        case_list = [x.strip() for x in f.readlines() if not x.startswith('#')]
    # file_name = 'imgCT_transformed_spacial.nii.gz'
    # root_dir = '/home/lance/NAS/datasets/Babak_Project/Lymphoma2_paired_only_obo'

    # output_name = 'totalsegmentations_' + file_name
else:
    has_header_switch = has_header(list_file)
    if has_header_switch:
        df = pd.read_csv(list_file)
        case_list = df['img_path'].tolist()
    else:
        df = pd.read_csv(list_file, header=None)
        case_list = df[0].tolist()

for case in case_list:
    filename = case.split('/')[-1]
    filename_stem = filename.split('.')[0]
    foldername = case.split('/')[-2]

    output_path = os.path.join(root_dir, foldername, 'totalsegmentation_'+filename_stem)

    img = nib.load(case).get_fdata()


pause
pause