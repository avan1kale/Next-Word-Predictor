# Next Word Prediction using LSTM

A deep learning model that predicts the next word in a sequence of text using an LSTM (Long Short-Term Memory) neural network built with TensorFlow/Keras.

## Overview

This project trains a language model capable of predicting the next word given an input sequence of text. It uses word embeddings and an LSTM layer to learn patterns and context from a training corpus, then generates predictions based on that learned context.

## Model Architecture

The model is a sequential neural network with the following layers:

| Layer | Details |
|-------|---------|
| Embedding | Vocabulary size: 8932, Embedding dimension: 100, Input length: 105 |
| LSTM | 150 units |
| Dense | 8932 units, Softmax activation |

```python
model = Sequential()
model.add(Embedding(8932, 100, input_length=105, input_shape=(105,)))
model.add(LSTM(150))
model.add(Dense(8932, activation='softmax'))
```

- **Embedding Layer**: Converts word indices into dense 100-dimensional vectors, capturing semantic relationships between words.
- **LSTM Layer**: Learns sequential dependencies and context from the input word sequences.
- **Dense Output Layer**: Produces a probability distribution over the vocabulary (8932 words) to predict the most likely next word.

## Project Structure
├── README.md           # Project documentation

├── data.txt            # Raw text converted into a training dataset

├── lstm-model.ipynb    # Notebook containing model training and predictions

└── requirements.txt    # Python dependencies

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Usage

1. Clone the repository:
```bash
   git clone <repo-url>
   cd <repo-name>
```

2. Open and run `lstm-model.ipynb` in Jupyter Notebook or JupyterLab:
```bash
   jupyter notebook lstm-model.ipynb
```
3. The notebook covers:
   - Loading and preprocessing `data.txt`
   - Building and training the LSTM model
   - Testing predictions with sample input sequences
