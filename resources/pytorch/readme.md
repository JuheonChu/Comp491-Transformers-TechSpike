This document outlines the setup process of PyTorch.

## Prerequisite

1) **Python**: 

Python allows you to use one of the most popular programing language--Python. 

* Below is the output to check if your local machine has Python.

>>> python-V  

![image](https://user-images.githubusercontent.com/60185619/197435718-f11ec452-8fd4-4010-b98d-9ebb7a28199d.png)

2) **pip3**: 

Pip3 is the official package manager and pip command for Python 3. It enables the installation and management of third party software packages with features and functionality not found in the Python standard library. Pip3 installs packages from PyPI (Python Package Index), but won’t resolve dependencies or help you solve dependency conflicts.


* Below is the output of pip3 installation.

>>> pip -V.

![image](https://user-images.githubusercontent.com/60185619/197435625-4363b099-f6eb-4f7e-b046-a688ae2fe97f.png)
<br/>

## PyTorch Installation (Window)
<br/><br/>
**STEP 1**: Visit <a href="https://pytorch.org/"> *PyTorch* </a> 
and scroll down the page untill you see this part:  <br/>
![image](https://user-images.githubusercontent.com/60185619/197436167-33508e07-cff1-4b9b-8c48-c022ad3590ad.png)
<br/>

**STEP 2**: Type following command in your local machine terminal and execute it.

>>> pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 

The terminal will then start installing PyTorch on your local machine as follows: 

![image](https://user-images.githubusercontent.com/60185619/197436619-0cf836d6-1cc4-4d6c-9108-f8828bc1db0c.png)

<br/>
**STEP 3**: When PyTorch installation is completed, verify PyTorch version by running following Python code below:

![image](https://user-images.githubusercontent.com/60185619/197438057-3b585476-633a-4f49-8572-79409173328f.png)

<br/>
**Output**: <br/>
![image](https://user-images.githubusercontent.com/60185619/197438226-48108d79-e337-41d5-bac4-c6fcbabf97b1.png)
(Note that the "Test1" is my python file name) 

<br/>
** STEP 4**: Try ***torch_test1*** file, using ***torch.rand()*** function. This will return a tensor filled with uniformly distributed random numbers on [0,1).

![image](https://user-images.githubusercontent.com/60185619/197672279-2dde6f15-fd2e-4fab-895d-a192dc56badf.png)

**Output**:  <br/>
![image](https://user-images.githubusercontent.com/60185619/197672532-c2eaee33-f7c9-4d97-bc8f-bbe6fa1c7752.png)


























