# pylint:disable=[all]None
from vgg16 import VGG16
import numpy as np
import pandas as pd
import tensorflow as tf
import _pickle as pickle

EMBEDDING_DIM = 128


class CaptionGenerator():

    def __init__(self):
        self.max_cap_len = None
        self.vocab_size = None
        self.index_word = None
        self.word_index = None
        self.total_samples = None
        self.encoded_images = pickle.load( open( "encoded_images.p", "rb" ) )
        self.variable_initializer()

    def variable_initializer(self):
        df = pd.read_csv('Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        nb_samples = df.shape[0]
        iterator = df.iterrows()
        caps = []
        for i in range(nb_samples):
            x = next(iterator)
            caps.append(x[1][1])

        self.total_samples=0
        for text in caps:
            self.total_samples+=len(text.split())-1
        print ("Total samples : "+str(self.total_samples))
        
        words = [txt.split() for txt in caps]
        unique = []
        for word in words:
            unique.extend(word)

        unique = list(set(unique))
        self.vocab_size = len(unique)
        self.word_index = {}
        self.index_word = {}
        for i, word in enumerate(unique):
            self.word_index[word]=i
            self.index_word[i]=word

        max_len = 0
        for caption in caps:
            if(len(caption.split()) > max_len):
                max_len = len(caption.split())
        self.max_cap_len = max_len
        print ("Vocabulary size: "+str(self.vocab_size))
        print ("Maximum caption length: "+str(self.max_cap_len))
        print ("Variables initialization done!")


    def data_generator(self, batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        print ("Generating data...")
        gen_count = 0
        df = pd.read_csv('Flickr8k_text/flickr_8k_train_dataset.txt', delimiter='\t')
        nb_samples = df.shape[0]
        iterator = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = next(iterator)
            caps.append(x[1][1])
            imgs.append(x[1][0])


        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter+=1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split())-1):
                    total_count+=1
                    partial = [self.word_index[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    next_val = np.zeros(self.vocab_size)
                    next_val[self.word_index[text.split()[i+1]]] = 1
                    next_words.append(next_val)
                    images.append(current_image)

                    if total_count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = tf.keras.preprocessing.sequence.pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count+=1
                        print ("yielding count: "+str(gen_count))
                        print(len(images[0]), len(partial_caps[0]))
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
        
    def load_image(self, path):
        img = tf.keras.preprocessing.image.load_img(path, target_size=(224,224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        return np.asarray(x)


    def create_model(self, ret_model = False):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape = (224, 224, 3))
        base_model.trainable=False
        image_model = tf.keras.Sequential(name='Convolutional')
        image_model.add(base_model)
        image_model.add(tf.keras.layers.Flatten())
        image_model.add(tf.keras.layers.Dense(EMBEDDING_DIM, input_dim = 4096, activation='relu'))
        image_model.add(tf.keras.layers.RepeatVector(self.max_cap_len))

        lang_model = tf.keras.Sequential(name='RNN')
        lang_model.add(tf.keras.layers.Embedding(self.vocab_size, 256, input_length=self.max_cap_len))
        lang_model.add(tf.keras.layers.LSTM(256,return_sequences=True))
        lang_model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(EMBEDDING_DIM)))

        comb_out = tf.keras.layers.add([image_model.output, lang_model.output])
        # THANKS FOR https://stackoverflow.com/a/54104873
        # print(f'{comb_out}')
        input1 = tf.keras.Input(shape=(None, 4096), name='e1_inp')
        input2 = tf.keras.Input(shape=(None, self.vocab_size), name='dec1_inp')
        model = tf.keras.Sequential(name='Combined')
        model.add(tf.keras.layers.LSTM(1000,return_sequences=False))
        model.add(tf.keras.layers.Dense(self.vocab_size))
        model.add(tf.keras.layers.Activation('softmax'))
        f_model = tf.keras.Model(inputs = [base_model.input, lang_model.input], outputs = model(comb_out), name='final')
        # f_model = model
        print ("Model created!")

        if(ret_model==True):
            return f_model

        f_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        return f_model

    def get_word(self,index):
        return self.index_word[index]