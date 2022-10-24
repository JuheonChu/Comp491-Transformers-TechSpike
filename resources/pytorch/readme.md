**Here will be the method of installing PyTorch into your windows local machine.**

**Prerequisite:**

Python: 

Python allows you to use one of the most popular programing language--Python. 

pip3: 

Pip3 is the official package manager and pip command for Python 3. It enables the installation and management of third party software packages with features and functionality not found in the Python standard library. Pip3 installs packages from PyPI (Python Package Index), but won’t resolve dependencies or help you solve dependency conflicts.

Since there are too many ways to install Python and Pip3, the installation methods for these two will not be demonstrate here. 

But I will provide ways for you to verify if pip3 and python3 is successfully installed or already installed into your local machine. 

To see if your local machine has Pip3: type "pip -V" in your terminal. 
You should see something like this as the output: 
![image](https://user-images.githubusercontent.com/60185619/197435625-4363b099-f6eb-4f7e-b046-a688ae2fe97f.png)


To see if your local machine has Python: type "python -V" in your local machine. 
![image](https://user-images.githubusercontent.com/60185619/197435718-f11ec452-8fd4-4010-b98d-9ebb7a28199d.png)

If there above process has been done, congratulation, you can now move forward to install PyTorch into your local machine. 

To install PyTorch, go to PyTorch page first at: https://pytorch.org/ 
Scroll down the page untill you see this part: 
![image](https://user-images.githubusercontent.com/60185619/197436167-33508e07-cff1-4b9b-8c48-c022ad3590ad.png)

Note that this README file only provides method for installing in windows. So please select all the above options as demonstrated in the image. 

Then copy the Command given from above which is :"pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116" into you terminal, the terminal will then start installing PyTorch on your local machine. 
![image](https://user-images.githubusercontent.com/60185619/197436619-0cf836d6-1cc4-4d6c-9108-f8828bc1db0c.png)

Because I already installed PyTorch, so i am not giving the installation process output here. 

Once the installation for PyTorch is done, you can verify your PyTorch version by running this Python code below: 
![image](https://user-images.githubusercontent.com/60185619/197438057-3b585476-633a-4f49-8572-79409173328f.png)
(Note that you can also find this snip of code from this github under the same path as this README file, the name of this snip of code is "PyTorch_Version")

You should see your output like this: 
![image](https://user-images.githubusercontent.com/60185619/197438226-48108d79-e337-41d5-bac4-c6fcbabf97b1.png)
(Note that the "Test1" is my python file name) 

If you have everything the same as mine, congratulation. You have now successfully installed PyTorch into your local machine. 


