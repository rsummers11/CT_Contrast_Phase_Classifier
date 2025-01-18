import os
import SimpleITK as sitk
import csv
# import tqdm
from tqdm import tqdm


def get_nifti_dimensions(folder_path, output_csv, max_depth=None, num_samples = None):
    """
    This function takes the path of a folder containing NIfTI files (.nii.gz) and writes the dimensions of each file as
    a row in a CSV file. The function also allows specifying the maximum depth of subfolders to search.

    Args:
        folder_path (str): The path of the folder containing NIfTI files.
        output_csv (str): The path of the output CSV file.
        max_depth (int): The maximum depth of subfolders to search. Default is 0, which means no subfolders are searched.
        num_samples (int): The number of samples to process. Default is None, which means all samples are processed.
    """
    # Initialize an empty list to store the file dimensions
    file_dimensions = []

    # Set the initial depth to 0
    current_depth = 0

    # Initialize a counter for the number of processed samples
    processed_samples = 0

    # Loop through the files and subfolders in the folder using os.walk()
    for root, dirs, files in os.walk(folder_path):
        # Calculate the current depth
        current_depth = root[len(folder_path):].count(os.path.sep)
        # If the current depth is less than or equal to the maximum depth, or max_depth is None, process the files
        if max_depth is None or current_depth <= max_depth:
            for file in tqdm(files, desc=f'Processing {root}'):
                if file.endswith('.nii.gz'):
                    try:
                        # Read the NIfTI file using SimpleITK
                        file_path = os.path.join(root, file)
                        image = sitk.ReadImage(file_path)

                        # Get the dimensions and spacing of the image
                        dimensions = image.GetSize()
                        spacing = image.GetSpacing()

                        # Append the file name and dimensions and spacing to the list
                        file_dimensions.append([file] + list(dimensions) + list(spacing))

                        # Increment the counter for processed samples
                        processed_samples += 1

                        # If the number of processed samples reaches the desired number, break the loop
                        if num_samples is not None and processed_samples >= num_samples:
                            break

                    except RuntimeError as e:
                        print(f"Error reading file {file}: {e}")


        # If the maximum depth is reached and not None, stop searching deeper subfolders
        if max_depth is not None and current_depth >= max_depth:
            dirs.clear()

        # If the number of processed samples reaches the desired number, break the loop
        if num_samples is not None and processed_samples >= num_samples:
            break

    # Write the dimensions to the CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['File', 'X', 'Y', 'Z', 'Spacing_X', 'Spacing_Y', 'Spacing_Z'])
        csv_writer.writerows(file_dimensions)

def main():
    # Example usage
    # folder_path = '/home/lance/NAS/users/Akshaya/My_Datasets/DeepLesion509_resized'
    # folder_path = '/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2'
    # folder_path = "/home/lance/NAS/users/Akshaya/My_Datasets/DeepLesion509_resized_resampled_same_size"
    # folder_path = '/home/lance/NAS/users/Akshaya/My_Datasets/5Phase_resized/Use_This/Attempt2_resampled_same_size'
    # folder_path = '/home/lance/NAS/datasets/CTC/CTC_3mm'
    folder_path = '/home/lance/NAS/users/Akshaya/My_Datasets/DeepLesion509_original_scan'
    output_csv = os.path.basename(folder_path)+ '_dimensions.csv'
    max_depth = 1  # Adjust the maximum depth as needed
    get_nifti_dimensions(folder_path, output_csv, max_depth)

if __name__ == '__main__':
    main()