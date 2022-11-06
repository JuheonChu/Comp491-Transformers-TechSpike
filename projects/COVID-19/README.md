# Covid-19-Infection-Percentage-Estimation-Challenge

This is a competetion organized at [codalab](https://competitions.codalab.org/competitions/35575#learn_the_details-evaluation). The problem is to find percentage of infection caused by COVID in Chest CT scan. This problem is different from classification where we figure out whether the person is infected by COVID or not. In this competetion we have found out the percentage of infection caused. 

## Dataset information
The challenge has three sets: Train, Val, and Test. The Train set is obtained from 132 CT-scans, from which 128 CT-scans has confirmed to have Covid-19 based on positive reverse transcription polymerase chain reaction (RT-PCR) and CT scan manifestations identified by two experienced thoracic radiologists. The rest four CT-scans have not any infection type (Healthy). The Val set is obtained from 57 CT-scans, from which 55 CT-scans has confirmed to have Covid-19 based on positive reverse transcription polymerase chain reaction (RT-PCR) and CT scan manifestations identified by two experienced thoracic radiologists. The rest two CT-scans have not any infection type (Healthy).

The evaluation cirtieria for the competetion is **mean absolute error (mae)**. The two thing that we are not classifying the disease and healthy images and evaluation metric is not accuracy/f1score make us think that this is not the classification problem. There for we have decided to take it as regression problem becuase mae is usually considered in regerssion problem.

Dataset is available at [github](https://github.com/faresbougourzi/Covid-19-Infection-Percentage-Estimation-Challenge) and we have downloaded it. 

## Our approach
We need following things
* a deep learning model
* a loss function
* 5 fold cross validation approach
* an optimizer
* few augmentation function
