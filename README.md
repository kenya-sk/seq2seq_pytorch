# seq2seq_pytorch
This repository reimplement Seq2Seq model by PyTorch.  
paper: Sequence to Sequence Learning with Neural Networks  
(https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)

<img src="./demo/model.png" margin="50" width="500" title="model"> 

## Getting Started

### Install Required Packages
First ensure that you have installed the following required packages.  
```
pip install -r requirements.txt
```

- python 3.6.8
- numPy 1.15.1
- pandas 0.23.4
- PyTorch 1.0.1
- tqdm 4.26.0
- janome (if you analyze Japanese)

### Prepare the Dataset
Prepare a dataset under the directory of "../data/datasets/".  
The dataset has two columns "src" and "trg".  
The "src" colum is a source sentence. The "trg" colum is a target sentece.  
The seq2seq model learns the conversion from "src" to "trg".

### Prepare the Vocaburary
Based on the frequency of words appearing in training data, build vocaburary to use encoder and decoder. default vacaburary size is 10,000, minimum frequency is 2.
The vocaburary is saved in "../data/vocab/{en|de}_vocab.pth".

```
# build vocaburary

python3 ./src/build_vocab.py
```

### Training Seq2Seq Model
In training seq2seq model is learned at the save time based on negative log likelihood loss of each word.

```
python3 ./src/train.py [-h] [-train_path] [-val_path] [-en_vocab_path] [-de_vocab_path]
                       [-pre_trained_path] [-save_model_path] [-en_n_layers] [-de_n_layers]
                       [-en_dropout] [-de_dropout] [-epochs] [batch_size] [-lr] [-grad_clip]
                       [-hidden_dim] [-embed_dim] [-shuffle] [-num_workers]
``` 

Hyper parameters and learning conditions can be set using arguments. Details are as follows.
```
[-train_path]        : The path of training dataset
[-val_path]          : The path of validation dataset
[-en_vocab_path]     : The path of encoder vocabuary
[-de_vocab_path]     : The path of decoder vocabuary
[-pre_trained_path]  : The path of pretrained model
[-save_model_path]   : The path of save the weight of trained seq2seq model
[-en_n_layers]       : The number of endoer GRU layer
[-de_n_layers]       : The number of decoder GRU layer
[-en_dropout]        : The dropout rate of decoder
[-de_dropout]        : The dropout rate of encoder
[-epochs]            : The number of epoch
[batch_size]         : Batch size
[-lr]                : Learning rate
[-grad_clip]         : The threshold of the gradient
[-hidden_dim]        : The dimension of hidden layer
[-embed_dim]         : The dimension of emnedding
[-shuffle]           : Whether to shuffle the dataset every epoch
[-num_workers]       : The number of threds
```