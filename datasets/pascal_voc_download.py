import os
import sys
import tarfile
import argparse
import urllib.request as request
import pathlib

# The URL where the PASCAL VOC data can be downloaded.
# http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
DATASET_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'


def download_and_uncompress_dataset(dataset_dir: str):
    """Downloads PASCAL VOC and uncompresses it locally.
    Parameters
    ----------
    dataset_dir : str
        The directory where the dataset is stored.
    """
    filename = DATASET_URL.split('/')[-1]
    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(dataset_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading {} {:.1f}%%'.format(
            filename,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
        
    print()
    print("Downloading PASCAL VOC from", DATASET_URL)
    print("Downloading PASCAL VOC to", filepath)
    filepath, _ = request.urlretrieve(DATASET_URL, filepath, _progress)
    statinfo = os.stat(filepath)
    print()
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    print('Uncompressing...')
    tarfile.open(filepath, 'r').extractall(dataset_dir)
    print('Successfully downloaded and extracted PASCAL VOC')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str,
        default=os.path.join(os.sep, 'mnt', 'datasets', 'public', 'research', 'pascal'),
        help='Path to the raw data')
    args = parser.parse_args()
    download_and_uncompress_dataset(args.dataset_dir)
    