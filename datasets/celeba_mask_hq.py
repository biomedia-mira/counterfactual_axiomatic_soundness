from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import colors
from numpy.typing import NDArray
from PIL import Image
from tqdm import tqdm
from itertools import product
from datasets.utils import image_gallery

#these attributes are subjective and thus removed
subjective_attributes = ['Attractive', 'Big_Lips', 'Big_Nose', 'Blurry', 'Chubby', 'High_Cheekbones',
                         'Narrow_Eyes', 'Oval_Face', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Young']

['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache',
 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

# Colligation/Yule/Phi coefficient https://en.wikipedia.org/wiki/Coefficient_of_colligation
def colligation_coefficient(var1: NDArray, var2: NDArray) -> float:
    a = np.sum(np.logical_and(var1 == 0, var2 == 0))
    b = np.sum(np.logical_and(var1 == 1, var2 == 0))
    c = np.sum(np.logical_and(var1 == 0, var2 == 1))
    d = np.sum(np.logical_and(var1 == 1, var2 == 1))
    t1, t2 = np.sqrt(a * d), np.sqrt(b * c)
    return float((t1 - t2) / (t1 + t2))


def parwise_colligation_coefficient(vars: NDArray) -> NDArray:
    assert vars.ndim == 2
    nvars = vars.shape[0]
    table = np.zeros(shape=(nvars, nvars))
    for i, j in product(range(nvars), range(nvars)):
        if i > j:
            continue
        cc = colligation_coefficient(vars[i], vars[j])
        table[i, j] = cc
        table[j, i] = cc
    return table


def analyse_dataset(images: NDArray, parents: Dict[str, NDArray]) -> None:
    # indices = np.where(np.logical_not(parents['Male']))[0]
    # indices = np.where(parents['Male'])[0]
    # print(len(indices))
    # images = images[indices]
    # parents={key: value[indices] for key, value in parents.items()}
    parent_names = list(parents.keys())
    _parents = np.array(list(parents.values()))
    cm = np.einsum('ns,ms->nm', _parents, _parents)
    #cm = cm / np.sum(cm, axis=1, keepdims=True) * 100
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, plt.get_cmap('Blues'))
    plt.colorbar()
    tick_marks = np.arange(len(parent_names))
    plt.xticks(tick_marks, parent_names, rotation=90)
    plt.yticks(tick_marks, parent_names)
    thresh = cm.max() / 2
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:d}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()

    ##
    yule_coefficient = parwise_colligation_coefficient(_parents)
    plt.figure(figsize=(20, 20))
    plt.imshow(np.abs(yule_coefficient), cmap='seismic', norm=colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1))
    plt.colorbar()
    tick_marks = np.arange(len(parent_names))
    plt.xticks(tick_marks, parent_names, rotation=90)
    plt.yticks(tick_marks, parent_names)
    for i, j in product(range(cm.shape[0]), range(yule_coefficient.shape[1])):
        plt.text(j, i, "{:0.2f}".format(yule_coefficient[i, j]), horizontalalignment="center", color="black")
    plt.tight_layout()
    plt.show()

    for i, j in zip(*np.unravel_index(np.argsort(-np.abs(yule_coefficient), axis=None), yule_coefficient.shape)):
        if i >= j:
            continue
        print(parent_names[i], parent_names[j], yule_coefficient[i, j])

    plt.imshow(images[0])
    plt.show()
    indices = np.where(np.logical_and(parents['Wearing_Lipstick'], parents['Male']))[0]
    #indices = np.where(parents['Narrow_Eyes'])[0]
    print(len(indices))
    num_images_to_display = min(8 * 8, len(indices))
    gallery = image_gallery(images[indices], ncols=8, num_images_to_display=num_images_to_display,
                            decode_fn=lambda x: x)
    plt.imshow(gallery)
    plt.show()
    return


def make_celeb_a_mask_hq_dataset(raw_data_dir: Path, size: Tuple[int, int]) -> Tuple[NDArray, Dict[str, NDArray]]:
    data = pd.read_csv(raw_data_dir / 'CelebAMask-HQ-attribute-anno.txt', delim_whitespace=True, header=1)
    _parents: List[NDArray] = []
    _images: List[NDArray] = []
    for image_name, attr in tqdm(data.iterrows()):
        image_path = raw_data_dir / 'CelebA-HQ-img' / image_name
        attr[attr == -1] = 0
        with Image.open(image_path) as image:
            _images.append(image.resize(size).__array__())
        _parents.append(attr.values)
    images = np.stack(_images)
    parents = np.stack(_parents)
    parents[parents == -1] = 0
    parents_as_dict = dict(zip(data.keys(), parents.T))
    return images, parents_as_dict


def get_celeb_a_mask_hq_dataset(dataset_dir: Path, raw_data_dir: Optional[Path] = None) -> tf.data.Dataset:
    images_path = str(dataset_dir / 'images.npy')
    parents_path = str(dataset_dir / 'parents.npy')
    try:
        images = np.load(images_path)
        parents = np.load(parents_path, allow_pickle=True).item()
    except FileNotFoundError:
        assert raw_data_dir is not None
        images, parents = make_celeb_a_mask_hq_dataset(raw_data_dir, size=(128, 128))
        dataset_dir.mkdir(exist_ok=True, parents=True)
        np.save(images_path, images)
        np.save(parents_path, parents)
    analyse_dataset(images, parents)
    return tf.data.Dataset.from_tensor_slices((images, parents))


def preprocess(feat_dict):
    image = feat_dict['image']
    attributes = {key: tf.cast(value, tf.float32) for key, value in feat_dict['attributes'].items()}
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5
    return image, attributes


if __name__ == "__main__":
    dataset_dir = Path('./data/celeb_a_mask_hq')
    raw_data_dir = Path('/vol/biodata/data/CelebAMask-HQ')
    ds_train = get_celeb_a_mask_hq_dataset(dataset_dir, raw_data_dir)
    # gcs_base_dir = "gs://celeb_a_dataset/"
    # celeb_a_builder = tfds.builder("celeb_a", data_dir=gcs_base_dir, version='2.0.0')
    # celeb_a_builder.download_and_prepare()
    # batch_size = 512

    # ds_train = celeb_a_builder.as_dataset(split='train').shuffle(1024).batch(batch_size).map(preprocess)
    # analyse(ds_train)
    # ds_train = celeb_a_builder.as_dataset(split='train').shuffle(1024).batch(batch_size).map(preprocess)
