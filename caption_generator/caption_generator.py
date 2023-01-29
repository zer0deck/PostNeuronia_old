# pylint:disable=[all]
import tensorflow as tf
import collections
import random
import numpy as np
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from prepare_dataset import load_image
from PIL import Image

# HYPERPARAMS

FP = 'E:/Projects/PostNeuronia/'
MAX_LENGTH = 50
VOCAB_SIZE = 5000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
features_shape = 2048
attention_features_shape = 64

####################################

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                            self.W2(hidden_with_time_axis)))

        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

class PostNeuronia():
    def __init__(self, tokenizer: tf.keras.layers.TextVectorization, image_features_extract_model, num_steps: int) -> None:
        self.encoder = CNN_Encoder(embedding_dim)
        self.decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())
        self.optimizer = tf.keras.optimizers.Adam()
        self.image_features_extract_model = image_features_extract_model
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        checkpoint_path = f"{FP}models/checkpoints/train"
        self.ckpt = tf.train.Checkpoint(encoder=self.encoder,
                                decoder=self.decoder,
                                optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, checkpoint_path, max_to_keep=5)
        self.start_epoch = 0
        if self.ckpt_manager.latest_checkpoint:
            self.start_epoch = int(self.ckpt_manager.latest_checkpoint.split('-')[-1])
        
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        self.loss_plot = []
        self.word_to_index = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary()
        )
        self.index_to_word = tf.keras.layers.StringLookup(
            mask_token="",
            vocabulary=tokenizer.get_vocabulary(),
            invert=True
        )
        self.num_steps = num_steps
    
    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, img_tensor, target):
        loss = 0

        # initializing the hidden state for each batch
        # because the captions are not related from image to image
        hidden = self.decoder.reset_state(batch_size=target.shape[0])

        dec_input = tf.expand_dims([self.word_to_index('<start>')] * target.shape[0], 1)

        with tf.GradientTape() as tape:
            features = self.encoder(img_tensor)

            for i in range(1, target.shape[1]):
                # passing the features through the decoder
                predictions, hidden, _ = self.decoder(dec_input, features, hidden)

                loss += self.loss_function(target[:, i], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(target[:, i], 1)

        total_loss = (loss / int(target.shape[1]))

        trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        return loss, total_loss
    
    def train(self, dataset:tf.data.Dataset, num_epoch = 20):
        for epoch in range(self.start_epoch, num_epoch):
            start = time.time()
            total_loss = 0

            for (batch, (img_tensor, target)) in enumerate(dataset):
                batch_loss, t_loss = self.train_step(img_tensor, target)
                total_loss += t_loss

                if batch % 100 == 0:
                    average_batch_loss = batch_loss.numpy()/int(target.shape[1])
                    print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')

            self.loss_plot.append(total_loss / self.num_steps)

            if epoch % 5 == 0:
                self.ckpt_manager.save()

            print(f'Epoch {epoch+1} Loss {total_loss/self.num_steps:.6f}')
            print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')
    
    def print_plot(self):
        plt.plot(self.loss_plot)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Plot')
        plt.show()

    def evaluate(self, image):
        attention_plot = np.zeros((MAX_LENGTH, attention_features_shape))

        hidden = self.decoder.reset_state(batch_size=1)

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = self.image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                    -1,
                                                    img_tensor_val.shape[3]))

        features = self.encoder(img_tensor_val)

        dec_input = tf.expand_dims([self.word_to_index('<start>')], 0)
        result = []

        for i in range(MAX_LENGTH):
            predictions, hidden, attention_weights = self.decoder(dec_input,
                                                            features,
                                                            hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            predicted_word = tf.compat.as_text(self.index_to_word(predicted_id).numpy())
            result.append(predicted_word)

            if predicted_word == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot

    def plot_attention(self, image, result, attention_plot):
        temp_image = np.array(Image.open(image))

        fig = plt.figure(figsize=(10, 10))

        len_result = len(result)
        for i in range(len_result):
            temp_att = np.resize(attention_plot[i], (8, 8))
            grid_size = max(int(np.ceil(len_result/2)), 2)
            ax = fig.add_subplot(grid_size, grid_size, i+1)
            ax.set_title(result[i])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

        plt.tight_layout()
        plt.show()

    def test(self):
        image_url = 'https://tensorflow.org/images/surf.jpg'
        image_extension = image_url[-4:]
        image_path = tf.keras.utils.get_file('image'+image_extension, origin=image_url)

        result, attention_plot = self.evaluate(image_path)
        print('Prediction Caption:', ' '.join(result))
        self.plot_attention(image_path, result, attention_plot)
        # opening the image
        Image.open(image_path)