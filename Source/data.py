import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, embeddings, labels):
        """
        Initialize the TextDataset.

        :param tokens: List of word tokens (integer indices)
        :param embeddings: Word embeddings (from GloVe)
        :param labels: List of labels
        """
        self.tokens = tokens
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        """
        Get the total number of data samples in the dataset.

        :return: The length of the dataset (number of samples)
        """
        return len(self.tokens)

    def __getitem__(self, idx):
        """
        Get a single data sample from the dataset.

        :param idx: Index of the data sample to retrieve
        :return: A tuple containing the label and word embeddings for the sample
        """
        return self.labels[idx], self.embeddings[self.tokens[idx], :]
