# pylint:disable = [all]

import tensorflow as tf
import collections
import random
import numpy as np
import json
from tqdm import tqdm

# HYPERPARAMS

FP = 'E:/Projects/PostNeuronia/'
MAX_LENGTH = 50
VOCAB_SIZE = 5000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EXTRACT_FEATURES = True
embedding_dim = 256
units = 512
features_shape = 2048
attention_features_shape = 64
SAVE = False

####################################

def load_image(image_path, size=299):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(size, size)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

def load_feature_extractor() -> tf.keras.Model:
	image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
	new_input = image_model.input
	hidden_layer = image_model.layers[-1].output

	return tf.keras.Model(new_input, hidden_layer)

def standardize(inputs):
	inputs = tf.strings.lower(inputs)
	return tf.strings.regex_replace(inputs, r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")

def save_image_features(encode_train:list, image_features_extract_model: tf.keras.Model, batch_size=32) -> None:

	image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
	image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

	for img, path in tqdm(image_dataset):
		batch_features = image_features_extract_model(img)
		batch_features = tf.reshape(batch_features,
									(batch_features.shape[0], -1, batch_features.shape[3])
								)

		for bf, p in zip(batch_features, path):
			path_of_feature = p.numpy().decode("utf-8")
			np.save(path_of_feature, bf.numpy())

def prepare_text_features(train_captions:list):
	caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)

	tokenizer = tf.keras.layers.TextVectorization(
		max_tokens=VOCAB_SIZE,
		standardize=standardize,
		output_sequence_length=MAX_LENGTH
		)
	tokenizer.adapt(caption_dataset)

	cap_vector = caption_dataset.map(lambda x: tokenizer(x))
	return cap_vector, tokenizer

def map_func(img_name, cap):
	img_tensor = np.load(img_name.decode('utf-8')+'.npy')
	return img_tensor, cap

def preprocess():
	# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	annotation_file = f'{FP}data/annotations/captions_train2014.json'
	PATH = f'{FP}data/train/'

	with open(annotation_file, 'r') as f:
		annotations = json.load(f)

	# Group all captions together having the same image ID.
	image_path_to_caption = collections.defaultdict(list)
	for val in annotations['annotations']:
		caption = f"<start> {val['caption']} <end>"
		image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
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

	print('Обработка датасета...')
	dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
	dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
	if SAVE:
		print('Сохранение датасета...')
		dataset.save(
			path=f'{FP}data/datasets', 
			compression='GZIP', 
			shard_func=None, 
			checkpoint_args=None
		)
		print('Dataset saved successfully')
	
	return dataset, tokenizer, image_features_extract_model, num_steps