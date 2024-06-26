import re
import torch
import config
import argparse
from Source.utils import load_file
from nltk.tokenize import word_tokenize
from Source.model import RNNNetwork, LSTMNetwork

def main(args_):
    # Process input text
    input_text = args_.test_complaint

    # Preprocess input text
    input_text = input_text.lower()  # Convert to lowercase
    input_text = re.sub(r"[^\w\d'\s]+", " ", input_text)  # Remove special characters
    input_text = re.sub("\d+", "", input_text)  # Remove digits
    input_text = re.sub(r'[x]{2,}', "", input_text)  # Remove consecutive 'x' characters
    input_text = re.sub(' +', ' ', input_text)  # Remove extra spaces

    # Tokenize the input text using NLTK's word_tokenize
    tokens = word_tokenize(input_text)

    # Add padding if the number of tokens is less than 20
    tokens = ['<pad>'] * (20 - len(tokens)) + tokens

    # Load label encoder to map class labels to integers
    label_encoder = load_file(config.label_encoder_path)
    num_classes = len(label_encoder.classes_)

    # Load the selected model (RNN or LSTM)
    if args_.model_type == "lstm":
        model = LSTMNetwork(config.input_size, config.hidden_size, num_classes)
        model_path = config.lstm_model_path
    else:
        model = RNNNetwork(config.input_size, config.hidden_size, num_classes)
        model_path = config.rnn_model_path

    # Load the model's pre-trained weights
    model.load_state_dict(torch.load(model_path))

    # Move the model to the GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Load vocabulary and word embeddings
    vocabulary = load_file(config.vocabulary_path)
    embeddings = load_file(config.embeddings_path)

    # Determine the device (GPU or CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Tokenize the input text and convert it into embeddings
    idx_token = []
    for token in tokens:
        if token in vocabulary:
            idx_token.append(vocabulary.index(token))
        else:
            idx_token.append(vocabulary.index('<unk>'))

    # Pick the word embeddings for the tokens
    token_emb = embeddings[idx_token, :]

    # Convert token embeddings into a torch tensor
    inp = torch.from_numpy(token_emb)

    # Move the tensor to the specified device (GPU or CPU)
    inp = inp.to(device)

    # Create a batch of 1 data point
    inp = torch.unsqueeze(inp, 0)

    # Forward pass through the model
    out = torch.squeeze(model(inp))

    # Find the predicted class based on the model's output
    prediction = label_encoder.classes_[torch.argmax(out)]
    print(f"Predicted Class: {prediction}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_complaint", type=str, help="Test complaint")
    parser.add_argument("--model_type", type=str, default="rnn", help="Model type: lstm or rnn")
    args = parser.parse_args()
    main(args)
