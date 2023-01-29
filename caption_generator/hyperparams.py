"""
The file that stores all the settings for the model.
"""

__all__ = [
  "FP", "MAX_LENGTH", 'VOCAB_SIZE', 'BATCH_SIZE', 'BUFFER_SIZE', 'EXTRACT_FEATURES', 'LOAD', 'SAVE',
  'EMBEDDING_DIM', 'UNITS', 'FEATURES_SHAPE', 'ATTENTION_FEATURES_SHAPE', 'TRAIN'
]

########################################
# DATASET
########################################

FP = 'E:/Projects/PostNeuronia/'
MAX_LENGTH = 50
VOCAB_SIZE = 5000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EXTRACT_FEATURES = True
LOAD = True
SAVE = False
TRAIN = False

########################################
# MODEL
########################################

EMBEDDING_DIM = 256
UNITS = 512
FEATURES_SHAPE = 2048
ATTENTION_FEATURES_SHAPE = 64
