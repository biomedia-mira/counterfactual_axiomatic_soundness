# https://www.tensorflow.org/responsible_ai/fairness_indicators/tutorials/Fairness_Indicators_TFCO_CelebA_Case_Study

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.tree_util import tree_map
from numpy.typing import NDArray
from tqdm import tqdm

from datasets.utils import image_gallery
from matplotlib import colors


def analyse(images: NDArray, parents: Dict[str, NDArray]) -> None:
    parent_names = list(parents.keys())
    corrcoef = np.corrcoef(np.array(list(parents.values())))

    # plot correlation coefficients
    plt.figure(figsize=(8, 6))
    plt.imshow(np.abs(corrcoef), cmap='seismic', norm=colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1))
    plt.colorbar()
    tick_marks = np.arange(len(parent_names))
    plt.xticks(tick_marks, parent_names, rotation=45)
    plt.yticks(tick_marks, parent_names)
    plt.tight_layout()
    plt.show()

    for i, j in zip(*np.unravel_index(np.argsort(-np.abs(corrcoef), axis=None), corrcoef.shape)):
        if i >= j:
            continue
        print(parent_names[i], parent_names[j], corrcoef[i, j])

    indices = np.where(np.logical_and(np.logical_not(parents['No_Beard']), np.logical_not(parents['Male'])))[0]
    print(len(indices))
    gallery = image_gallery(images[indices], ncols=8, num_images_to_display=8 * 8, decode_fn=lambda x: x)
    plt.imshow(gallery)
    return


def get_celeb_a_dataset(dataset_dir: Path, train: bool = True) -> Tuple[tf.data.Dataset, Dict[str, NDArray]]:
    images_path = str(dataset_dir / 'images.npy')
    parents_path = str(dataset_dir / 'parents.npy')
    try:
        images = np.load(images_path)
        parents = np.load(parents_path, allow_pickle=True).item()
    except FileNotFoundError:
        gcs_base_dir = "gs://celeb_a_dataset/"
        celeb_a_builder = tfds.builder("celeb_a", data_dir=gcs_base_dir, version='2.0.0')
        celeb_a_builder.download_and_prepare()
        dataset = celeb_a_builder.as_dataset(split='train' if train else 'test')
        parents = {}
        _images = []
        for data in tqdm(dataset.batch(1000)):
            _image, _parents = data['image'].numpy(), {key: value.numpy() for key, value in data['attributes'].items()}
            parents = tree_map(lambda x, y: np.concatenate((x, y)), parents, _parents) if parents else _parents
            _images.append(_image)
        images = np.concatenate(_images)
        dataset_dir.mkdir(exist_ok=True, parents=True)
        np.save(images_path, images)
        np.save(parents_path, parents)
    analyse(images, parents)
    dataset = tf.data.Dataset.from_tensor_slices((images, parents))
    return dataset, parents


def preprocess(feat_dict):
    # Separate out the image and target variable from the feature dictionary.
    image = feat_dict['image']
    attributes = {key: tf.cast(value, tf.float32) for key, value in feat_dict['attributes'].items()}
    # Resize and normalize image.
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [28, 28])
    image = (image - 127.5) / 127.5
    return image, attributes


# # Train data returning either 2 or 3 elements (the third element being the group)
# def celeb_a_train_data_wo_group(batch_size):
#     celeb_a_train_data = celeb_a_builder.as_dataset(split='train').shuffle(1024).repeat().batch(batch_size).map(
#         preprocess_input_dict)
#     return celeb_a_train_data.map(get_image_and_label)
#
# def celeb_a_train_data_w_group(batch_size):
#     celeb_a_train_data = celeb_a_builder.as_dataset(split='train').shuffle(1024).repeat().batch(batch_size).map(
#         preprocess_input_dict)
#     return celeb_a_train_data.map(get_image_label_and_group)
#
# # Test data for the overall evaluation
# celeb_a_test_data = celeb_a_builder.as_dataset(split='test').batch(1).map(preprocess_input_dict).map(
#     get_image_label_and_group)

if __name__ == "__main__":
    ds_train = get_celeb_a_dataset(Path('./data/celeb_a'))
    # gcs_base_dir = "gs://celeb_a_dataset/"
    # celeb_a_builder = tfds.builder("celeb_a", data_dir=gcs_base_dir, version='2.0.0')
    # celeb_a_builder.download_and_prepare()
    # batch_size = 512

    # ds_train = celeb_a_builder.as_dataset(split='train').shuffle(1024).batch(batch_size).map(preprocess)
    # analyse(ds_train)
    # ds_train = celeb_a_builder.as_dataset(split='train').shuffle(1024).batch(batch_size).map(preprocess)
