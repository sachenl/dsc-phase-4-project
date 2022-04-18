# Project 4 final
## Overview
For this project, we trained a pretrained cnn model to predict the Classification of the x-ray images.

## Business understanding
The medical dataset comes from Kermany et al. contains a set of x-ray images of pediatric patients. The images will show whether the patients have pneumonia or not. our task is to build a model that can classify whether a given patient has pneumonia, given a chest x-ray image. Since this is an Image Classification problem, we are going to solve it with Deep Learning.

## The Dataset
The dataset that we are going to use the image classification is the chese_xray which contains two categories: Pneumonia and Normal.The data was downloaded from https://data.mendeley.com/datasets/rscbjbr9sj/3 to the local drive and unzip. The data set is assigned into two folders (train and test) and contains subfoler for each of the category Pneumonia and Normal. In each of the folders, there are a lot of xray images. To check how many samples in each of the categories, we used the OS.listdir methods.


In train folder, there are normal folder which contains 1349 images and PNEUMONIA folder which contains 3884 images. In test folder, there are normal folder which contains 235 images and PNEUMONIA folder which contains 390 images. The images in each folder is too large for the modeling since our local computer is not very powerful for the mulitple testing. We need to downsampling the dataset first to find the optimal model and parameter first. Then using the full dataset to train and test our model. Base on our earlier expience, we will use 20% of the total dataset to modeling our model. We also need to make 10% of the traning data to validation dataset.


## Plan

1. Downsampleing the data set by randomly choosing 20% of the original trainning and testing images to the new data_org_subset folder. Make a new folder of validation and random select 5% of the images from trainning folder.
2. Define the train generator, validation generator and test generator.
3. Build the deep learning model base on the pretrained CNN (VGG19) by adding a few fully connected layers. Train the model with selected images.
4. Retrain the model with full training data. 
5. Evaluate the model with the test images.


### 1. Rebuild the data subset folder with 20% of the original images

Define the old and new direction of dataset and a new method to creat and transfer images to copy 20% of the training and testing images from the orignal folder. We also made a new folder for validation and randomly seleted 5% of the images from training folder.


![copy20](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/copy%2020.png)





## 2. Define the train generator, validation generator and test generator.


We plot some of the images in the training dataset. However, I can not tell which one is a case of pneumonia and which one is a normal case just by looking at the pictures. Now we are going to train the computer with a pretrainned cnn model to predict whether the picture belong to pneumonia or normal case.

![fig 2 samples](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/fig%202%20sample%20images.png)



##  3. Build the model base on pretrain network VGG19 and fit the model to the trainning images.



![model summary ](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/fig%202%20extra%20model%20summary.png)     
                                                                 



![results_1](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/results_1_partial.png)

Now, we plot the accuracy and loss curve of the model to traning dataset.


![fit3 acc](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/fig%203%20acc_partial.png)

The acc and loss curve of training gave us pretty good score and the validation scores are going to the similar range in each steps. Thus we can use the same model on the full traning dataset for better training.

Then we save the current model.

## 4. Retrain the model with full  dataset.

Now is the time to use our model for the full dataset. We  remade the folder of train, val, test folder for full dataset. 
Transfer 90% of train images to new train and 10% of train images to new validation folder. 
Transfer 100% of test to new test folder


![copy_full](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/copy%20all.png)




![result_2](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/results_2_full.png)


Plot the accuracy of the model again.


![acc_2](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/fig%204%20acc_full.png)


In this fitting, both of the training accuracy and validation accuracy are very high. Even though the fluctuation of validation accurancy are larger than trainning, they had the same trend in general.

## 5. Evaluate the model with the test images.
We first generate the test labels as the real class of the images.

We then calculated the accuracy of the model on the testing images.

10/10 [=====] - 42s 4s/step - loss: 0.0685 - acc: 0.9750

The test accuracy of the model on test dataset are 95% which is very high also.

### Then we calculate the predictions with the model and then make the confusion box


![confusion_box](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/fig%205%20confusionbox.png)


![scores](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/scores.png)

The confusion box showes that the TP and TN prediction are much higher compare to the FN and FP results. The f1-score for both normal and pneumonia data are 0.79 and 0.9 which are very reasonble too.

### Finally, we  plot few of the examples of images with  percentage of predictions


![fig_6_plot_final](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/fig%206%20samples%20final.png)

We randomly plot future of the pictures from the test folder and give the prediction and actual case of the picture. The prediction and actual results are same with each other in our samples.

## Conclusion
Based on 20% of the whole dataset, we created a CNN model based on a pretrained model (VGG19) which can classify X-ray images as a Pneumonia case or a Normal case. The model was then retrained with whole dataset and tested with the seperated test images. The accuracy of the predicion is around 95%. 





