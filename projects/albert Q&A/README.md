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


## Installation


### Check GPU Settings:

We implemented ALBERT Question-Answering model with Google Colab. Click [Here](https://colab.research.google.com/) how to launch Google Colab and manage its settings. So, it is necessary to change the runtime type to GPU settings. Here is the manual how to change the [Google Colab Runtime setting](https://research.google.com/colaboratory/local-runtimes.html)

You can choose one of the ways to check your GPU settings.

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
  
 **Desired Output**: 
 
 ![Output](https://user-images.githubusercontent.com/35699839/199302514-2d576d94-0fb1-463d-9273-3d5fe20e89c6.png)

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

### Establish ALBERT Question-Answering Model and Configuration

Albert Model name: "ktrapeznikov/albert-xlarge-v2-squad-v2"

#### Establish ALBERT

``` python
  >>> import os
  >>> import torch
  >>> import time 

  >>> from torch.utils.data import DataLoader 

  >>> from transformers import (
  >>>     AlbertConfig, 
  >>>     AlbertForQuestionAnswering,
  >>>     AlbertTokenizer,
  >>>     squad_convert_examples_to_features
  >>> )

  >>> from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample
  >>> from transformers.data.metrics.squad_metrics import compute_predictions_logits

  >>> use_own_model = False

  >>> if use_own_model:
  >>>   model_name_or_path = "/content/model_output"
  >>> else:
  >>>   model_name_or_path = "ktrapeznikov/albert-xlarge-v2-squad-v2"

  >>> output_dir = ""
  
 ``` 

 #### Configure ALBERT Q&A Model
 
``` python
>>> n_best_size = 1
>>> max_answer_length = 30
>>> do_lower_case = True 
>>> null_score_diff_threshold = 0.0 
```


### Set up ALBERT Model & Tensor Attributes

```python
>>> def to_list(tensor):
>>>  return tensor.detach().cpu().tolist()
  
>>> config_class, model_class, tokenizer_class = (
>>>    AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer
>>> )
>>> config = config_class.from_pretrained(model_name_or_path)

>>> tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)

>>> model = model_class.from_pretrained(model_name_or_path, config=config)


>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>> model.to(device)
>>> processor = SquadV2Processor()
```

## Run prediction

run [`run.py`](https://github.com/JuheonChu/Natural-Language-Processing/tree/main/projects/albert%20Q%26A) to get desired answer from the trained model.
