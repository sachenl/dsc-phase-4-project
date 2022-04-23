# Project 4 final
## Overview
For this project, we trained a pretrained cnn model to predict the Classification of the x-ray images.

## Business understanding
The medical dataset comes from Kermany et al. contains a set of x-ray images of pediatric patients. The images will show whether the patients have pneumonia or not. Our task is to build a model that can classify whether a given patient has pneumonia given a chest x-ray image. Since this is an Image Classification problem, we will solve it with Deep Learning.

## The Dataset
The dataset that we will use for image classification is the chese_xray which contains two categories: Pneumonia and Normal. The data was downloaded from https://data.mendeley.com/datasets/rscbjbr9sj/3 to the local drive and unzipped. The data set is assigned into two folders (train and test) and contains a subfolder for each Pneumonia and Normal category.
In each of the folders, there are a lot of x-ray images. To check how many samples were in each category, we used the OS.listdir methods.

In the train folder, there is a regular folder that contains 1349 images and the PNEUMONIA folder, which includes 3884 images. The NORMAL folder in the test folder contains 235 pictures, and the PNEUMONIA folder contains 390 images. The images in each folder are too large for the modeling since our local computer is not very powerful for multiple testing. Therefore, we need to downsample the dataset to find the optimal model and parameter first. We are then using the entire dataset to train and test our model. Based on our earlier experience, we will use 20% of the total dataset to model our model. We also need to make 10% of the training data to the validation dataset.


## Plan

1. Downsampleing the data set by randomly choosing 20% of the initial training and testing images to the new data_org_subset folder. Make a new validation folder and randomly select 5% of the pictures from the training folder.
2. Define the trained generator, validation generator, and test generator.
3. Build the deep learning model base on the Pretrained CNN (VGG19) by adding a few fully connected layers. Then, train the model with selected images.
4. Retrain the model with complete training data. 
5. Evaluate the model with the test images.


### 1. Rebuild the data subset folder with 20% of the original images

Define the old and new direction of dataset and a new method to creat and transfer images to copy 20% of the training and testing images from the orignal folder. We also made a new folder for validation and randomly seleted 5% of the images from training folder.


![copy20](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/copy%2020.png)





## 2. Define the train generator, validation generator and test generator.


We plot some of the images in the training dataset. However, I can not tell which one is a case of pneumonia and which one is a normal case just by looking at the pictures. So now we will train the computer with a Pretrainned CNN model to predict whether the picture belongs to pneumonia or normal case.

![fig 2 samples](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/fig%202%20sample%20images.png)

## 3. Build a baseline model

![sequencial](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/sequencial.png)
![result](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/own-results.png)
![](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/own-acc.png)

##  4. Build the model base on pretrain network VGG19 and fit the model to the trainning images.




![model summary ](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/fig%202%20extra%20model%20summary.png)     
                                                                 



![results_1](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/results_1_partial.png)

Now, we plot the accuracy and loss curve of the model to traning dataset.


![fit3 acc](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/fig%203%20acc_partial.png)

The acc and loss curve of training gave us pretty good score and the validation scores are going to the similar range in each steps. Thus we can use the same model on the full traning dataset for better training.

Then we save the current model.

## 5. Retrain the model with full  dataset.

Now is the time to use our model for the full dataset. We  remade the folder of train, val, test folder for full dataset. 
Transfer 90% of train images to new train and 10% of train images to new validation folder. 
Transfer 100% of test to new test folder


![copy_full](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/copy%20all.png)




![result_2](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/results_2_full.png)


Plot the accuracy of the model again.


![acc_2](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/fig%204%20acc_full.png)


In this fitting, both training accuracy and validation accuracy are very high. Even though the fluctuation of validation accuracy is bigger than training accuracy, both accuracies generally had the same trend. 

## 6. Evaluate the model with the test images.
We first generate the test labels as the real class of the images.

We then calculated the accuracy of the model on the testing images.

10/10 [=====] - 42s 4s/step - loss: 0.0685 - acc: 0.9750

The test accuracy of the model on test dataset are 95% which is very high also.

### Then we calculate the predictions with the model and then make the confusion box


![confusion_box](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/fig%205%20confusionbox.png)


![scores](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/scores.png)

The confusion box shows that the TP and TN predictions are much higher than the FN and FP results. The f1-score for both normal and pneumonia data are 0.79 and 0.9, which are very reasonable too. 

### Finally, we  plot few of the examples of images with  percentage of predictions


![fig_6_plot_final](https://raw.githubusercontent.com/sachenl/dsc-phase-4-project/main/image/fig%206%20samples%20final.png)

We randomly plot some of the pictures from the test folder and give the prediction and actual case of the picture. The prediction and actual results are identical to each other in our samples. 

## Conclusion
Based on 20% of the whole dataset, we created a CNN model based on a Pretrained model (VGG19), which can classify X-ray images as Pneumonia cases or Normal cases. The model was then retrained with the whole dataset and tested with the separated test images. The accuracy of the prediction is around 95%. 




