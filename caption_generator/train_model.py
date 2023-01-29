# pylint:disable=[all]
from prepare_dataset import preprocess
from caption_generator import PostNeuronia

def train_model(epochs = 10):
    dataset, tokenizer, image_features_extract_model, num_steps = preprocess()
    model = PostNeuronia(tokenizer, image_features_extract_model, num_steps)
    model.train(dataset, num_epoch=epochs)
    model.test()

# if __name__ == '__main__':
#     train_model(epochs=10)