# Text Classification with RNN and LSTM Models on Customer Complaints Data

## Business Overview

Text classification, a crucial application of Natural Language Processing (NLP), finds its relevance in various industries. This project focuses on the application of Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models for text classification. RNNs and LSTMs excel in handling sequential data, making them ideal choices for NLP tasks. The project specifically targets customer complaints about consumer financial products.

---

## Aim

The primary objective is to leverage RNN and LSTM models for text classification on a dataset containing over two million customer complaints about consumer financial products.

---

## Data Description

The dataset includes customer complaints, each associated with a product category. The text of the complaint and the corresponding product category are provided. To enhance text representation, pre-trained word vectors from the GloVe dataset (glove.6B) are employed.

---

## Tech Stack

- Language: `Python`
- Libraries: `pandas`, `torch`, `nltk`, `numpy`, `pickle`, `re`, `tqdm`, `sklearn`

---

## Approach

### 1. Installation and Imports

Install necessary packages using the `pip` command. Import the required libraries for the project.

### 2. Configuration

Define configuration file paths for managing data and model-related parameters.

### 3. Process GloVe Embeddings

- Read the GloVe text file.
- Convert embeddings to a float array.
- Add embeddings for padding and unknown items.
- Save embeddings and vocabulary.

### 4. Process Text Data

- Read the CSV file and handle null values.
- Address duplicate labels.
- Encode the label column and save the encoder and encoded labels.

### 5. Data Preprocessing

- Convert text to lowercase.
- Remove punctuation, digits, and additional spaces.
- Tokenize the text.

### 6. Build Data Loader

Construct a data loader for efficient model training.

### 7. Model Building

- Define RNN architecture.
- Define LSTM architecture.
- Create functions for training and testing the models.

### 8. Model Training

- Train the RNN model.
- Train the LSTM model.

### 9. Prediction on Test Data

Make predictions using the trained models on the test data.

---

## Modular Code Overview

1. **Input**: Contains data required for analysis, including:
   - `complaints.csv`
   - `glove.6B.50d.txt` (download from [here](https://nlp.stanford.edu/projects/glove/))

2. **Source**: Contains modularized code for various project steps, including:
   - `model.py`
   - `data.py`
   - `utils.py`

   These Python files contain functions used in the `Engine.py` file.

3. **Output**: Contains files required for model training, including:
   - `embeddings.pkl`
   - `label_encoder.pkl`
   - `labels.pkl`
   - `model_lstm.pkl`
   - `model_rnn.pkl`
   - `vocabulary.pkl`
   - `tokens.pkl`

   (The `model_lstm.pkl` and `model_rnn.pkl` files are our saved models after training)

4. **config.py**: Contains project configurations.

5. **Engine.py**: The main file to run the entire project, which trains the models and saves them in the output folder.

---
