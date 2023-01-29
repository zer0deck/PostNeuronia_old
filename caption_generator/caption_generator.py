# pylint:disable=['import-error','unexpected-keyword-arg','no-value-for-parameter','invalid-name','arguments-differ','signature-differs']

"""
The file that stores main executable classes.
"""

__all__ = [
    "PostNeuronia"
]

########################################
# IMPORTS
########################################

import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from prepare_dataset import load_image
from hyperparams import FP, MAX_LENGTH, EMBEDDING_DIM, UNITS, ATTENTION_FEATURES_SHAPE

########################################
# LAYERS
########################################

class BahdanauAttention(tf.keras.Model):
    """
    Main attention layer (as like Transformer model)
    found: https://www.tensorflow.org/tutorials/text/image_captioning
    """
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        """
        Overriten `tf.keras.Model.call()` method with using attension layer.
        """
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                            self.W2(hidden_with_time_axis)))

        score = self.V(attention_hidden_layer)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    """
    Encoder layer. Adds recursive `tf.keras.layers.Dense()` layer.
    """
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        """
        Overriten `tf.keras.Model.call()` method adds
        `tf.keras.layers.Dense()` layer with relu activation.
        """
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
    """
    Decoder layer. Simplified RNN `tf.keras.Model()` implementation.
    """
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

    def call(self, inputs, training, mask):
        context_vector, attention_weights = self.attention(training, mask)
        inputs = self.embedding(inputs)
        inputs = tf.concat([tf.expand_dims(context_vector, 1), inputs], axis=-1)
        output, state = self.gru(inputs)
        inputs = self.fc1(output)
        inputs = tf.reshape(inputs, (-1, inputs.shape[2]))
        inputs = self.fc2(inputs)

        return inputs, state, attention_weights

    def reset_state(self, batch_size):
        """
        small inline function, that creates empty tensor with `batch_size`
        """
        return tf.zeros((batch_size, self.units))


########################################
# RUN
########################################

class PostNeuronia():
    # pylint:disable=['line-too-long']
    """
    Combined model for creating memes.
    Uses imagenet-based CNN and RNN architectures to generate a text description
    for an image.

    :param tokenizer: tokenizer, build with specific keras layer
    :type tokenizer: tf.keras.layers.TextVectorization
    :param image_features_extract_model: imagenet model. `tf.keras.applications.InceptionV3()` expected by default
    :type image_features_extract_model: tf.keras.Model
    :param num_steps: num_steps
    :type num_steps: int
    """
    # pylint:enable=['line-too-long']
    def __init__(self,
            tokenizer: tf.keras.layers.TextVectorization,
            image_features_extract_model: tf.keras.Model,
            num_steps: int) -> None:
        self.encoder = CNN_Encoder(EMBEDDING_DIM)
        self.decoder = RNN_Decoder(EMBEDDING_DIM, UNITS, tokenizer.vocabulary_size())
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
        """
        Gradient tape based loss_function.
        docs_url: https://www.tensorflow.org/api_docs/python/tf/GradientTape
        """
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, img_tensor, target):
        """
        Gradient-based main training function
        """
        loss = 0
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
        """
        The function to be used for training.

        :param dataset: tensorflow prebatched dataset to train
        :type dataset: tf.data.Dataset
        :param num_epoch: number of epoch to train
        :type num_epoch: int
        """
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
        """
        inline to plot training process
        """
        plt.plot(self.loss_plot)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Plot')
        plt.show()

    def predict(self, image):
        """
        Main function to predict result for pretrained model

        :param image: path to the image
        :type image: str
        """
        attention_plot = np.zeros((MAX_LENGTH, ATTENTION_FEATURES_SHAPE))

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

    def debug_plot_attention(self, image, result, attention_plot):
        """
        debug function to show image features.
        """
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
        """
        testing function
        """
        image_url = 'https://tensorflow.org/images/surf.jpg'
        image_extension = image_url[-4:]
        image_path = tf.keras.utils.get_file('image'+image_extension, origin=image_url)

        result, attention_plot = self.predict(image_path)
        print('Prediction Caption:', ' '.join(result))
        self.debug_plot_attention(image_path, result, attention_plot)
        # opening the image
        Image.open(image_path)
