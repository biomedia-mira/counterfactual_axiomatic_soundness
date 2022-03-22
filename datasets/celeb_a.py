# https://www.tensorflow.org/responsible_ai/fairness_indicators/tutorials/Fairness_Indicators_TFCO_CelebA_Case_Study
import tensorflow_datasets as tfds
import tensorflow as tf

ATTR_KEY = "attributes"
IMAGE_KEY = "image"
LABEL_KEY = "Smiling"
GROUP_KEY = "Young"
IMAGE_SIZE = 28


def preprocess_input_dict(feat_dict):
    # Separate out the image and target variable from the feature dictionary.
    image = feat_dict[IMAGE_KEY]
    label = feat_dict[ATTR_KEY][LABEL_KEY]
    group = feat_dict[ATTR_KEY][GROUP_KEY]

    # Resize and normalize image.
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image /= 255.0

    # Cast label and group to float32.
    label = tf.cast(label, tf.float32)
    group = tf.cast(group, tf.float32)

    feat_dict[IMAGE_KEY] = image
    feat_dict[ATTR_KEY][LABEL_KEY] = label
    feat_dict[ATTR_KEY][GROUP_KEY] = group

    return feat_dict


get_image_and_label = lambda feat_dict: (feat_dict[IMAGE_KEY], feat_dict[ATTR_KEY][LABEL_KEY])
get_image_label_and_group = lambda feat_dict: (
feat_dict[IMAGE_KEY], feat_dict[ATTR_KEY][LABEL_KEY], feat_dict[ATTR_KEY][GROUP_KEY])

get_image_and_label = lambda feat_dict: (feat_dict[IMAGE_KEY], feat_dict[ATTR_KEY][LABEL_KEY])
get_image_label_and_group = lambda feat_dict: (
feat_dict[IMAGE_KEY], feat_dict[ATTR_KEY][LABEL_KEY], feat_dict[ATTR_KEY][GROUP_KEY])


def get():
    gcs_base_dir = "gs://celeb_a_dataset/"
    celeb_a_builder = tfds.builder("celeb_a", data_dir=gcs_base_dir, version='2.0.0')
    celeb_a_builder.download_and_prepare()

    num_test_shards_dict = {'0.3.0': 4, '2.0.0': 2}  # Used because we download the test dataset separately
    version = str(celeb_a_builder.info.version)
    print('Celeb_A dataset version: %s' % version)

    # Train data returning either 2 or 3 elements (the third element being the group)
    def celeb_a_train_data_wo_group(batch_size):
        celeb_a_train_data = celeb_a_builder.as_dataset(split='train').shuffle(1024).repeat().batch(batch_size).map(
            preprocess_input_dict)
        return celeb_a_train_data.map(get_image_and_label)

    def celeb_a_train_data_w_group(batch_size):
        celeb_a_train_data = celeb_a_builder.as_dataset(split='train').shuffle(1024).repeat().batch(batch_size).map(
            preprocess_input_dict)
        return celeb_a_train_data.map(get_image_label_and_group)

    # Test data for the overall evaluation
    celeb_a_test_data = celeb_a_builder.as_dataset(split='test').batch(1).map(preprocess_input_dict).map(
        get_image_label_and_group)
    return celeb_a_test_data

if __name__ == "__main__":
    b = get()
    data = iter(b).__next__()
    print('here')
