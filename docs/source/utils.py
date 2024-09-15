# Install gdown if not already installed
# !pip install gdown

import gdown
import zipfile
import os

def download_and_unzip(file_id, output_dir=None):
    """
    Downloads a zipped file from Google Drive using its file ID and unzips it to a specified directory.

    Parameters:
    - file_id (str): The file ID of the Google Drive file.
    - output_dir (str): The directory where the file should be extracted. Defaults to the current working directory.

    Returns:
    - str: The path to the extracted file.
    """
    if output_dir is None:
        output_dir = os.getcwd()
        
    os.makedirs(output_dir, exist_ok=True)
    url = f'https://drive.google.com/uc?id={file_id}'
    output = os.path.join(output_dir, 'temp.zip')
    gdown.download(url, output, quiet=False)

    with zipfile.ZipFile(output, 'r') as zip_ref:
        original_name = zip_ref.namelist()[0]
        zip_ref.extractall(output_dir)

    # Remove the temporary zip file
    os.remove(output)

    # The path to the extracted file
    extracted_file = os.path.join(output_dir, original_name)

    print(f"File extracted as: {extracted_file}, saved to {output_dir}")
    return extracted_file