# Covid-19-Infection-Percentage-Estimation

## Abstract
Motivated by [codalab](https://competitions.codalab.org/competitions/35575#learn_the_details-evaluation), we focused on simulating machine learning model to predict the COVID-19 infection rate by scanning ~746 CT Scan images.


## Dataset information
The challenge has three sets: Train, Val, and Test. The Train set is obtained from 132 CT-scans, from which 128 CT-scans has confirmed to have Covid-19 based on positive reverse transcription polymerase chain reaction (RT-PCR) and CT scan manifestations identified by two experienced thoracic radiologists. The rest four CT-scans have not any infection type (Healthy). The Val set is obtained from 57 CT-scans, from which 55 CT-scans has confirmed to have Covid-19 based on positive reverse transcription polymerase chain reaction (RT-PCR) and CT scan manifestations identified by two experienced thoracic radiologists. The rest two CT-scans have not any infection type (Healthy).

The evaluation criteria is **Entropy Loss**. The two thing that we are not classifying the disease and healthy images and evaluation metric is not accuracy/f1score make us think that this is not the classification problem. There for we have decided to take it as regression problem becuase mae is usually considered in regerssion problem.

Dataset is available at [github](https://github.com/faresbougourzi/Covid-19-Infection-Percentage-Estimation-Challenge) and we have downloaded it. 

## Model
- pretrained VGG-19 with batch normalization as our model
- Objective Function: cross entropy loss
- Architecture: <br>
  ![VGG Architecture](https://user-images.githubusercontent.com/35699839/201311679-69d85aed-026d-4225-b962-a9776c550318.png)
- Parameters: Optimizer running with Stochastic Gradient descent algorithm with Momentum value of .9

## Testing
- Area under ROC
- Confusion Matrix
