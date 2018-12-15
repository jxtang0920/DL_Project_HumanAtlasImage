# DL_Project_HumanAtlasImage

For InceptionResNet, we implemented 2 versions, Inception_resnet_v2_base_model.ipynb and Inception_resnet_v2_base_model2.ipynb, one uses random weights initialization and another use pretrained weights from imagenet. Inception_retrain is used for model retraining and improve accuracy.

## ResNet50:
:link:[ResNet50](https://github.com/jxtang0920/DL_Project_HumanAtlasImage/blob/master/resnet50.ipynb)
:link:[Reset50V2](https://github.com/jxtang0920/DL_Project_HumanAtlasImage/blob/master/resnet50-v2.ipynb)

We use a light ResNet50 model and low resolution images to have a baseline that can be used later to select higher end models and explore the effect of image resolution on the prediction accuracy. 
The most common way to convert RGBY(Yellow) to RGB is just dropping Y(Yellow) channel while keeping RGB without modification. Therefore, we replace the first convolution layer from 7x7 3->64 to 7x7 4->64 while keeping weights from 3->64 and setting new initial weights for Y channel to be zero. It allows using the original weights to initialize the network while giving the opportunity to the model to incorporate Y channel into prediction during training.
### Our ResNet50 Structure:
![alt text](https://github.com/jxtang0920/DL_Project_HumanAtlasImage/blob/master/ResNet50.png)

## InceptionResNetv2:
:link:[ InceptionResNetv2_base](https://github.com/jxtang0920/DL_Project_HumanAtlasImage/blob/master/Inception_resnet_v2_base_model.ipynb)
:link:[InceptionResNetv2_retrain](https://github.com/jxtang0920/DL_Project_HumanAtlasImage/blob/master/Inception_resnet_retrain.py)

The minimum required size of Inception-ResNet-V2 is 139X139, so, in order to save the computational resources,  we set the image size to this value. 
Keras provides the applicable version of this net, implemented in keras.applications. As the version on the SCC is out-dated, we downloaded the github version of this file.
The base parameters we used for our training batch_size is 100. Loss function we use is binary cross-entropy,  and optimizer we choose is Adadelta. We saved the initial model to use for future retraining.

### Our ResNet50 Structure:
![alt text](https://github.com/jxtang0920/DL_Project_HumanAtlasImage/blob/master/Inception.png)

## Result:

|| Baseline Model 8 layers(100 epoch)| ResNet50 with sigmoid tense layer (300 epoch)|InceptionResNetV2 with sigmoid tense layer    (300 epoch)|
|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|
| Validation F1|0.053|0.1415|0.4382|
| Predicted F1|0.048|0.162|0.451|
| Rank on Kaggle|1588/1686|1411/1686|609/1686|


