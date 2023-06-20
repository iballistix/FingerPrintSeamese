import json
import os
import pathlib
import sys
import click
import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
import cv2

from model import BinaryClassifier
from utils import circ_aug, crop_to_size

cfg = dict()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_probability_dragmark(model, img_path, device, x=None, y=None, contrast=None, black_boost=None, brightness=None,
                              debug=False, debug_dir=None):
    """
    Predicts probability of labels for input image
    """
    # Augmentation
    transform = A.Compose([
        A.Resize(width=cfg['img_size'], height=cfg['img_size']),
        A.ToGray(p=1),
    ])
    img = np.asarray(Image.open(img_path))
    img = circ_aug(img, zoom_coef=4 / 9, x=y, y=x)
    img = transform(image=img)["image"] # Augmenting
    img = circ_aug(img, zoom_coef=1) # Cropping again without zooming
    if debug:
        Image.fromarray(img.astype(np.uint8)).save(os.path.join(debug_dir, "debug_dragmark.png"), optimize=True)
    #img = np.expand_dims(img, axis=2)
    img = np.array([img[:, :, 0]])
    img = np.transpose(img, (1, 2, 0))
    img = img / np.max(img)  # Normalize image
    img = np.transpose(img, (2, 0, 1))  # Convert shape [H,W,C] -> [C,H,W]
    img = torch.from_numpy(img).type(torch.float32)
    img = np.expand_dims(img, axis=0)  # Add dimension [C,H,W] -> [1,C,H,W]
    img = torch.tensor(img).type(torch.float)  # Convert to Pytorch Tensor
    img = img.to(device)
    prediction = model(img)
    return nn.Sigmoid()(prediction).item()


def find_probability_parallel(model, img_path, device, x=None, y=None, contrast=None, black_boost=None, brightness=None,
                              debug=False, debug_dir=None):
    """
    Predicts probability of labels for input image
    """
    # Augmentation
    transform = A.Compose([
        A.Resize(width=cfg['img_size'], height=cfg['img_size'])
    ])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = crop_to_size(img, y, x, contrast, black_boost, brightness)
    img = transform(image=img)["image"] # Augmenting
    img = circ_aug(img, zoom_coef=1, crop_center=True) # Cropping again without zooming
    if debug:
        Image.fromarray(img.astype(np.uint8)).save(os.path.join(debug_dir, "debug_parallel.png"), optimize=True)
    img = np.array([img] * 3).astype('uint8')
    img = np.transpose(img, (1, 2, 0))
    img = np.array([img[:, :, 0]])
    img = np.transpose(img, (1, 2, 0))
    img = img / np.max(img)  # Normalize image
    img = np.transpose(img, (2, 0, 1))  # Convert shape [H,W,C] -> [C,H,W]
    img = torch.from_numpy(img).type(torch.float32)
    img = np.expand_dims(img, axis=0)  # Add dimension [C,H,W] -> [1,C,H,W]
    img = torch.tensor(img).type(torch.float)  # Convert to Pytorch Tensor
    img = img.to(device)
    prediction = model(img)
    return nn.Sigmoid()(prediction).item()


def get_center_coordinates_from_dat(filepath):
    """
    Gets x and y coordinates from .dat file
    """
    with open(filepath, 'r') as f:
        data = f.read()
        return (int(x) for x in data.split('\n')[:2])



@click.command()
@click.option('-c', '--config', help='Path to config file',
              default='configs/detect_config.yaml', type=click.File('rt'))
@click.option('-ap', '--avg_path', help='Path to averaged .dng image',
              required=True, type=click.Path(exists=True))
@click.option('-nmp', '--nm_path', help='Path to normal map image',
              required=True, type=click.Path(exists=True))
@click.option('-cc', '--center_coordinates', help='Path to center coordinates .dat file',
              required=True, type=click.Path(exists=True))
@click.option('-o', '--out_path', help='Path to output result .json file with predicted probabilities')
@click.option('-dp', '--dragmark_path', help='to dragmark classifier model file',
              default='saved_models/dragmark_classifier.pt', type=click.Path(exists=True))
@click.option('-pp', '--parallel_path', help='Path to parallel classifier model file',
              default='saved_models/parallel_classifier.pt', type=click.Path(exists=True))
@click.option('-cpar', '--contrast_parallel', help='Contrast factor for parallel lines identification',
              type=float, default=6)
@click.option('-bpar', '--brightness_parallel', help='Brightness factor for parallel lines identification',
              type=float, default=160)
@click.option('-bbpar', '--blackboost_parallel', help='Black boost factor for parallel lines identification',
              type=float, default=1)
@click.option('-cdrg', '--contrast_dragmark', help='Contrast factor for dragmark identification',
              type=float, default=3)
@click.option('-bdrg', '--brightness_dragmark', help='Brightness factor for dragmark identification',
              type=float, default=130)
@click.option('-bbdrg', '--blackboost_dragmark', help='Black boost factor for dragmark identification',
              type=float, default=2)
@click.option('--debug/--no-debug', help='Enable debug mode', envvar='DEBUG')
@click.option('--debug-dir',
              help='Specify folder where to put debug files',
              envvar='AZ_BATCH_TASK_WORKING_DIR',
              default='.', type=click.Path(exists=True))
def main(config, avg_path, nm_path, center_coordinates, out_path, dragmark_path, parallel_path,
         contrast_parallel, brightness_parallel, blackboost_parallel,
         contrast_dragmark, brightness_dragmark, blackboost_dragmark,
         debug, debug_dir):
    global cfg
    try:
        cfg = yaml.safe_load(config)
    except yaml.YAMLError as exc:
        print(exc)
        quit()

    # Setting random seed for reproducible results
    np.random.seed(cfg['random_seed'])
    torch.manual_seed(cfg['random_seed'])
    os.environ['PYTHONHASHSEED'] = str(cfg['random_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)

    model_dragmark = BinaryClassifier()  # Initialize model
    model_parallel = BinaryClassifier()

    model_dragmark.load_state_dict(torch.load(dragmark_path, map_location=device))  # Load saved model weights
    model_dragmark.to(device)  # GPU or CPU
    model_dragmark.eval()  # Set model to evaluation mode

    model_parallel.load_state_dict(torch.load(parallel_path, map_location=device))  # Load saved model weights
    model_parallel.to(device)  # GPU or CPU
    model_parallel.eval()  # Set model to evaluation mode

    # Reading center coordinates
    try:
        x, y = get_center_coordinates_from_dat(center_coordinates)
    except ValueError:
        print(f'Shell center file {center_coordinates} has wrong structure.')
        sys.exit()

    # Getting predicted probabilities
    pred_probabilities_parallel = find_probability_parallel(model_parallel, nm_path, device, x=x, y=y,
                                                   contrast=contrast_parallel,
                                                   brightness=brightness_parallel,
                                                   black_boost=blackboost_parallel,
                                                   debug=debug,
                                                   debug_dir=debug_dir)
    pred_probabilities_dragmark = find_probability_dragmark(model_dragmark, avg_path, device, x=x, y=y,
                                                   contrast=contrast_dragmark,
                                                   brightness=brightness_dragmark,
                                                   black_boost=blackboost_dragmark,
                                                   debug=debug,
                                                   debug_dir=debug_dir)

    final_probabilities = {
        'is_dragmark': pred_probabilities_dragmark,
        'is_parallel': pred_probabilities_parallel
    }
    print(final_probabilities)
    if out_path:
        print(f'Saving detected characteristics to {out_path}')
        # Create directories to `out_path`
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(final_probabilities, f)


if __name__ == '__main__':
    main()
