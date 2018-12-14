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