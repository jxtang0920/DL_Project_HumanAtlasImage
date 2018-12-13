import pandas as pd
import numpy as np
from os import listdir
import keras
from skimage.io import imread
from skimage.transform import resize
import keras.backend as K


train_path = "train/"
test_path = "test/"
label_path = './train.csv'
train_files = listdir(train_path)
test_files = listdir(test_path)
train_labels = pd.read_csv(label_path)

names_dict = {
    0:  "Nucleoplasm",  
    1:  "Nuclear membrane",   
    2:  "Nucleoli",   
    3:  "Nucleoli fibrillar center",   
    4:  "Nuclear speckles",
    5:  "Nuclear bodies",   
    6:  "Endoplasmic reticulum",   
    7:  "Golgi apparatus",   
    8:  "Peroxisomes",   
    9:  "Endosomes",   
    10:  "Lysosomes",   
    11:  "Intermediate filaments",   
    12:  "Actin filaments",   
    13:  "Focal adhesion sites",   
    14:  "Microtubules",   
    15:  "Microtubule ends",   
    16:  "Cytokinetic bridge",   
    17:  "Mitotic spindle",   
    18:  "Microtubule organizing center",   
    19:  "Centrosome",   
    20:  "Lipid droplets",   
    21:  "Plasma membrane",   
    22:  "Cell junctions",   
    23:  "Mitochondria",   
    24:  "Aggresome",   
    25:  "Cytosol",   
    26:  "Cytoplasmic bodies",   
    27:  "Rods & rings"
}

class ModelParameters(object):
    path = train_path
    num_classes=28
    image_rows=512
    image_cols=512
    batch_size=100
    n_channels=3
    shuffle=False
    scaled_row_dim = 139
    scaled_col_dim = 139 
    n_epochs=100

parameter = ModelParameters()

def fill_targets(row):
    row.Target = np.array(row.Target.split(" ")).astype(np.int)
    for num in row.Target:
        name = names_dict[int(num)]
        row.loc[name] = 1
    return row

for key in names_dict.keys():
    train_labels[names_dict[key]] = 0
train_labels = train_labels.apply(fill_targets, axis=1)
train_labels["number_of_targets"] = train_labels.drop(["Id", "Target"],axis=1).sum(axis=1)

from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
for train_idx, test_idx in kf.split(train_labels.index.values):
    partition = {}
    partition["train"] = train_labels.Id.values[train_idx]
    partition["validation"] = train_labels.Id.values[test_idx]

class ImagePreprocessor:
    
    def __init__(self, modelparameter):
        self.parameter = modelparameter
        self.path = self.parameter.path
        self.scaled_row_dim = self.parameter.scaled_row_dim
        self.scaled_col_dim = self.parameter.scaled_col_dim
        self.n_channels = self.parameter.n_channels
    
    def preprocess(self, image):
        image = self.resize(image)
        image = self.reshape(image)
        image = self.normalize(image)
        return image
    
    def resize(self, image):
        return resize(image, (self.scaled_row_dim, self.scaled_col_dim))
    
    def reshape(self, image):
        return np.reshape(image, (image.shape[0], image.shape[1], self.n_channels))
    
    def normalize(self, image):
        return (image / 255.0 - 0.5) / 0.5
            
    def load_image(self, image_id):
        image = np.zeros(shape=(512,512,4))
        image[:,:,0] = imread(self.basepath + image_id + "_green" + ".png")
        image[:,:,1] = imread(self.basepath + image_id + "_blue" + ".png")
        image[:,:,2] = imread(self.basepath + image_id + "_red" + ".png")
        image[:,:,3] = imread(self.basepath + image_id + "_yellow" + ".png")
        return image[:,:,0:self.parameter.n_channels]

preprocessor = ImagePreprocessor(parameter)

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, list_IDs, labels, modelparameter, imagepreprocessor):
        self.params = modelparameter
        self.labels = labels
        self.list_IDs = list_IDs
        self.dim = (self.params.scaled_row_dim, self.params.scaled_col_dim)
        self.batch_size = self.params.batch_size
        self.n_channels = self.params.n_channels
        self.num_classes = self.params.num_classes
        self.preprocessor = imagepreprocessor
        self.shuffle = self.params.shuffle
        self.on_epoch_end()
    
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def get_targets_per_image(self, identifier):
        return self.labels.loc[self.labels.Id==identifier].drop(
                ["Id", "Target", "number_of_targets"], axis=1).values
            
    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.num_classes), dtype=int)
        for i, identifier in enumerate(list_IDs_temp):
            image = self.preprocessor.load_image(identifier)
            image = self.preprocessor.preprocess(image)
            X[i] = image
            y[i] = self.get_targets_per_image(identifier)
        return X, y
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))
 
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

class PredictGenerator:
    
    def __init__(self, predict_Ids, imagepreprocessor, predict_path):
        self.preprocessor = imagepreprocessor
        self.preprocessor.basepath = predict_path
        self.identifiers = predict_Ids
    
    def predict(self, model):
        y = np.empty(shape=(len(self.identifiers), self.preprocessor.parameter.num_classes))
        for n in range(len(self.identifiers)):
            image = self.preprocessor.load_image(self.identifiers[n])
            image = self.preprocessor.preprocess(image)
            image = image.reshape((1, *image.shape))
            y[n] = model.predict(image)
        return y

def base_f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return f1

def f1_min(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.min(f1)

def f1_max(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.max(f1)

def f1_mean(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.mean(f1)

def f1_std(y_true, y_pred):
    f1 = base_f1(y_true, y_pred)
    return K.std(f1)

class TrackHistory(keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

df = pd.read_csv('sample_submission.csv')
partition2 = np.array(df['Id'])
parameter = ModelParameters()
preprocessor = ImagePreprocessor(parameter)
labels = train_labels
training_generator = DataGenerator(partition['train'], labels,
                                           parameter, preprocessor)
validation_generator = DataGenerator(partition['validation'], labels,
                                             parameter, preprocessor)
predict_generator = PredictGenerator(partition2, preprocessor, test_path)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.models import Model
from inception_resnet_v2 import InceptionResNetV2
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LambdaCallback
from keras.callbacks import Callback
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import tensorflow as tf
import keras

def create_model(input_shape, n_out):
    
    prev_model = load_model('InceptionResNetv2_model.h5',custom_objects={"f1_mean": f1_mean,"f1_std":f1_std,\
                                                                        "f1_min":f1_std,"f1_max":f1_std})
    prev_model.layers.pop()
    prev_model.layers.pop()
    x = Dense(512,activation = 'relu', input_dim=2)(prev_model.output)
    x = Dropout(0.25)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)
    predictions = Dense(n_out, activation = 'sigmoid')(x)
    model = Model(input = prev_model.input, output = predictions)
    
    return model

model = create_model(
    input_shape=(ModelParameters.scaled_row_dim,ModelParameters.scaled_col_dim,\
                 ModelParameters.n_channels), 
    n_out=ModelParameters.num_classes)

import warnings
warnings.filterwarnings("ignore")

model.compile(
    loss='binary_crossentropy',  
    optimizer=Adam(1e-4),
    metrics=['acc', f1_mean])

history = model.fit_generator(
    training_generator,
    steps_per_epoch=100,
    validation_data=validation_generator,
    epochs=ModelParameters.n_epochs, 
    verbose=1,
    validation_steps = 100
    )

model.save("InceptionResNetv2_model_2.h5")
proba_predictions = model.predict(predict_generator)
improved_proba_predictions = pd.DataFrame(proba_predictions)
improved_proba_predictions.to_csv("InceptionResNetV2_predictions_2.csv")




