"""
The file that stores all of the preprocessing process)))
"""
# pylint:disable = ['import-error', 'unnecessary-lambda', 'unspecified-encoding']

########################################
# RUN
########################################

import json
import pickle
import random
import collections
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from hyperparams import (
    VOCAB_SIZE,
    MAX_LENGTH,
    FP,
    LOAD,
    EXTRACT_FEATURES,
    BATCH_SIZE,
    BUFFER_SIZE,
    SAVE
)

########################################
# TECH
########################################

def load_image(image_path, size=299):
    """
	inline stores image_file loader
	"""
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(size, size)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def map_func(img_name, cap):
    """
	decoder for numpy-saved image features
	"""
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap

def standardize(inputs):
    """
	text filter
	"""
    inputs = tf.strings.lower(inputs)
    return tf.strings.regex_replace(inputs, r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")

########################################
# IMAGE PREPROCESSOR
########################################

def load_feature_extractor() -> tf.keras.Model:
    """
	created imagenet `tf.keras.applications.InceptionV3()` based feature extractor.
	"""
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    return tf.keras.Model(new_input, hidden_layer)

def save_image_features(
	encode_train:list,
	image_features_extract_model: tf.keras.Model,
	batch_size=32) -> None:
    """
	saves image features in numpy files
	"""

    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
		load_image,
		num_parallel_calls=tf.data.AUTOTUNE
	).batch(batch_size)

    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3])
                                )

        for feature, place in zip(batch_features, path):
            path_of_feature = place.numpy().decode("utf-8")
            np.save(path_of_feature, feature.numpy())

########################################
# TEXT PREPROCESSOR
########################################

def prepare_text_features(train_captions:list):
    """
	RNN model base text preprocessor.
	"""
    caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)

    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        standardize=standardize,
        output_sequence_length=MAX_LENGTH
        )
    tokenizer.adapt(caption_dataset)

    cap_vector = caption_dataset.map(lambda x: tokenizer(x))
    return cap_vector, tokenizer


########################################
# RUN
########################################

def preprocess():
    """
	executable

	:return: dataset, tokenizer, image_features_extract_model, num_steps
	:rtype: list[tf.data.Dataset, tf.keras.layers.TextVectorization, tf.keras.Model, int]
	"""
    if LOAD:
        print('Загрузка сохраненного датасета...')
        dataset = tf.data.Dataset.load(path=f'{FP}data/datasets', compression='GZIP')
        print('Загрузка параметров...')
        with open(f'{FP}data/datasets/info.pkl', 'rb') as inp:
            info_data = pickle.load(inp)
        print('Успешно загружено')
        return dataset, info_data[0], info_data[1], info_data[2]

    # print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    annotation_file = f'{FP}data/annotations/captions.json'
    main_path = f'{FP}data/train/'

    with open(annotation_file, 'r') as file:
        annotations = json.load(file)

    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = main_path + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
        image_path_to_caption[image_path].append(caption)

    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)


    train_image_paths = image_paths[:6000]
    print(f'Загружено: {len(train_image_paths)} объектов')

    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

    encode_train = sorted(set(img_name_vector))
    print(f'Уникальных изображений: {len(encode_train)}')

    image_features_extract_model = load_feature_extractor()
    print(f'Запущена модель извлечения эмбендингов: {image_features_extract_model}')

    if EXTRACT_FEATURES:
        print('Обработка свойств изображений и сохранение...')
        save_image_features(encode_train, image_features_extract_model)

    print('Обработка текстов описания...')
    cap_vector, tokenizer = prepare_text_features(train_captions=train_captions)

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    img_name_train = []
    cap_train = []

    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    img_name_train_keys = img_keys

    for imgt in tqdm(img_name_train_keys):
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    num_steps = len(img_name_train) // BATCH_SIZE

    print('Обработка датасета...')
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    dataset = dataset.map(
        lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int64]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    if SAVE:
        print('Сохранение свойств...')
        with open(f'{FP}data/datasets/info.pkl', 'wb') as outp:  # Overwrites any existing file.
            pickle.dump([tokenizer, image_features_extract_model, num_steps],
				outp,
				pickle.HIGHEST_PROTOCOL
			)
        print('Сохранение датасета...')
        dataset.save(
            path=f'{FP}data/datasets',
            compression='GZIP',
            shard_func=None,
            checkpoint_args=None
        )
        print('Dataset saved successfully')

    print('Подготовка данных заверешена')

    return dataset, tokenizer, image_features_extract_model, num_steps
