import os
import sys
import tarfile
import argparse
import urllib.request as request
import pathlib
import shutil
from google_drive_downloader import GoogleDriveDownloader as gdd

# The URL where the PASCAL VOC data can be downloaded.
# http://home.bharathh.info/pubs/codes/SBD/download.html
# This link does not work anymore:
# DATASET_URL = 'https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'
# So, using google drive: https://drive.google.com/file/d/1EQSKo5n2obj7tW8RytYTJ-eEYbXqtUXE/view
FILE_ID = '1EQSKo5n2obj7tW8RytYTJ-eEYbXqtUXE'
FILE_NAME = 'benchmark.tgz'
TRAIN_FILE_URL = 'http://home.bharathh.info/pubs/codes/SBD/train_noval.txt'

def download_and_uncompress_dataset(dataset_dir: str):
    """Downloads SBD, uncompresses it locally and patches train.txt 
        to exclude VOC2012 validation images from train.
    Parameters
    ----------
    dataset_dir : str
        The directory where the dataset is stored.
    """
    filename = FILE_NAME
    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(dataset_dir, filename)
    
    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading {} {:.1f}%%'.format(
            TRAIN_FILE_URL,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
        
    print()
    print("Downloading SBD from Google Drive file id", FILE_ID)
    print("Downloading SBD to", filepath)
    gdd.download_file_from_google_drive(file_id=FILE_ID,
                                    dest_path=filepath,
                                    unzip=False)
    statinfo = os.stat(filepath)
    print()
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    print('Uncompressing...')
    tarfile.open(filepath, 'r').extractall(dataset_dir)
    shutil.move(os.path.join(dataset_dir, 'benchmark_RELEASE', 'dataset'), dataset_dir)
    print('Successfully downloaded and extracted SBD')
    print('Patching the train.txt to exclude VOC2012 validation images from train...')
    # Move the original train file
    train_txt_filepath = os.path.join(dataset_dir, 'dataset', 'train.txt')
    shutil.move(train_txt_filepath, 
                os.path.join(dataset_dir, 'dataset', 'train_orig.txt'))
    # Download new file
    filepath, _ = request.urlretrieve(TRAIN_FILE_URL, train_txt_filepath, _progress)
    statinfo = os.stat(filepath)
    print()
    print('Successfully downloaded', train_txt_filepath, statinfo.st_size, 'bytes.')
    print('train.txt patch complete')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str,
        default=os.path.join(os.sep, 'mnt', 'datasets', 'public', 'research', 'sbd'),
        help='Path to the raw data')
    args = parser.parse_args()
    download_and_uncompress_dataset(args.dataset_dir)
    