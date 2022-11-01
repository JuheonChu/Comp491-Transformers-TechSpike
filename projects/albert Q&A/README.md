<!---
Albert Question-Answering Deep Learning Model (John Chu, Adia Wu)
-->

# Albert Question-Answering Deep Learning Model

This folder contains several scripts that showcase how to fine-tune a 🤗 Transformers model on a question answering dataset,
like SQuAD v2.0.

## Abstract

The [`run.py`](https://github.com/JuheonChu/Natural-Language-Processing/tree/main/projects/albert%20Q%26A) leverage the 🤗 Trainer for fine-tuning. Follow the steps below to train ALBERT Question-Answering model.

**Note:** This script only works with Question-Answering models that have a fast tokenizer (backed by the 🤗 Tokenizers library) as it
uses special features of those tokenizers. You can check if your favorite model has a fast tokenizer in
[this table](https://huggingface.co/transformers/index.html#supported-frameworks). In this project, we are using ALBERT model in order to get full PyTorch, TensorFlow, and Flax(Jax) support. Credits to [hugging's Transformers](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering)


### Check GPU Settings:

We implemented ALBERT Question-Answering model with Google Colab. So, it is necessary to change the runtime type to GPU settings.
You can choose one of the ways to check your GPA settings.

  - #### NVIDIA 
  ```bash
  !nvidia-smi
  ```
  
  - #### TensorFlow (run following python code)
  ``` python
  >>> import tensorflow as tf
  >>> device_name = tf.test.gpu_device_name()
  # Check GPU setting of the local machine
  >>> if device_name != '/device:GPU:0':
  >>>   raise SystemError('GPU Not Found')
  >>> print('Found GPU at: {}'.format(device_name))
 ``` 
  
  

### Clone Hugging Face Library from the Github and switch the branch to the working tree

```bash
!git clone https://github.com/huggingface/transformers \
&& cd transformers \
&& git checkout a3085020ed0d81d4903c50967687192e3101e770 
```

### Install Transformers & TensorBoardX

```bash
!pip install ./transformers/
!pip install tensorboardX
```



### Train ALBERT on [SQuAD v2.0](https://rajpurkar.github.io/SQuAD-explorer/)

The [`run.py`]( https://github.com/JuheonChu/Natural-Language-Processing/tree/main/projects/albert%20Q%26A) script
allows to fine-tune ALBERT model which has a (`ForQuestionAnswering` version in the library) on a question-answering SQuAD dataset in a structured JSON format. 


#### Command for SQuAD2.0:

First, Create *dataset* directory and install Train & Dev SQuAD v2.0 dataset which is structured in JSON format in *dataset* directory.

```bash
!mkdir dataset \
&& cd dataset \
&& wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json \
&& wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```
Then, train ALBERT Model. This takes 12-15 minutes for Window 21H2 16.0 GB RAM with i7 Intel Processor. [ALBERT Tokenizer](https://huggingface.co/docs/transformers/model_doc/albert) has ALBERT Parameters. ALBERT Model uses SentencePiece file in AlbertTokenizer class.

```bash
export SQUAD_DIR=/path/to/SQUAD

!export SQUAD_DIR=/content/dataset \
&& python transformers/examples/run_squad.py \
  --model_type albert \
  --model_name_or_path albert-base-v2 \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v2.0.json \
  --predict_file $SQUAD_DIR/dev-v2.0.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 1.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /content/model_output \
  --save_steps 1000 \
  --threads 4 \
  --version_2_with_negative 
```










# ALBERT Deep Learning Q&A System (John Chu, Adia Wu)

## STEP 1: Get Training data
- Get the SQUAD dataset.
* Install SQuAD v2.0 to train model with 100,000 answerable questions.

## STEP 2: Train Model with Albert
- Train Albert model with training data which takes approximately 2 hours.

## STEP 3: Hugging Face Library
- Design *prediction* functions.

## STEP 4: Evaluate Q & A
  - Run prediction on the answer regarding the question.

## Libraries
  - PyTorch
  - TensorFlow
  - Hugging's Transformers
