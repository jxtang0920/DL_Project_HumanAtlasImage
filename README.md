# DL_Project_HumanAtlasImage

For InceptionResNet, we implemented 2 versions, Inception_resnet_v2_base_model.ipynb and Inception_resnet_v2_base_model2.ipynb, one uses random weights initialization and another use pretrained weights from imagenet. Inception_retrain is used for model retraining and improve accuracy.

## ResNet50:

We use a light ResNet50 model and low resolution images to have a baseline that can be used later to select higher end models and explore the effect of image resolution on the prediction accuracy. 
The most common way to convert RGBY(Yellow) to RGB is just dropping Y(Yellow) channel while keeping RGB without modification. Therefore, we replace the first convolution layer from 7x7 3->64 to 7x7 4->64 while keeping weights from 3->64 and setting new initial weights for Y channel to be zero. It allows using the original weights to initialize the network while giving the opportunity to the model to incorporate Y channel into prediction during training.
### Our ResNet50 Structure:
![alt text](https://github.com/jxtang0920/DL_Project_HumanAtlasImage/blob/master/ResNet50.png)


## Result:

|| Baseline Model 8 layers(100 epoch)| ResNet50 with sigmoid tense layer (300 epoch)|InceptionResNetV2 with sigmoid tense layer    (300 epoch)|
|---------------------------------------|---------------------------------------|---------------------------------------|---------------------------------------|
| Validation F1|0.053|0.1415|0.4382|
| Predicted F1|0.048|0.162|0.451|
| Rank on Kaggle|1588/1686|1411/1686|609/1686|


