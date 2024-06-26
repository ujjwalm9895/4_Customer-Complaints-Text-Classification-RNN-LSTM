# Hyperparameters
lr = 0.0001                 # Learning rate for training
input_size = 50             # Size of input features
num_epochs = 50             # Number of training epochs
hidden_size = 50            # Size of the hidden layer in the RNN or LSTM

# File paths for data and models
label_col = "Product"       # Name of the label column in the dataset
tokens_path = "Output/tokens.pkl"       # Path to save tokenized data
labels_path = "Output/labels.pkl"       # Path to save labels
data_path = "Input/complaints.csv"      # Path to the CSV data file
rnn_model_path = "Output/model_rnn.pth" # Path to save the RNN model
lstm_model_path = "Output/model_lstm.pth"   # Path to save the LSTM model
vocabulary_path = "Output/vocabulary.pkl"   # Path to save vocabulary
embeddings_path = "Output/embeddings.pkl"   # Path to save word embeddings
glove_vector_path = "Input/glove.6B.50d.txt" # Path to the pre-trained GloVe word vectors
text_col_name = "Consumer complaint narrative" # Name of the text column in the dataset
label_encoder_path = "Output/label_encoder.pkl" # Path to save label encoder

# Mapping of product names to shorter codes
product_map = {
    'Vehicle loan or lease': 'vehicle_loan',
    'Credit reporting, credit repair services, or other personal consumer reports': 'credit_report',
    'Credit card or prepaid card': 'card',
    'Money transfer, virtual currency, or money service': 'money_transfer',
    'virtual currency': 'money_transfer',
    'Mortgage': 'mortgage',
    'Payday loan, title loan, or personal loan': 'loan',
    'Debt collection': 'debt_collection',
    'Checking or savings account': 'savings_account',
    'Credit card': 'card',
    'Bank account or service': 'savings_account',
    'Credit reporting': 'credit_report',
    'Prepaid card': 'card',
    'Payday loan': 'loan',
    'Other financial service': 'others',
    'Virtual currency': 'money_transfer',
    'Student loan': 'loan',
    'Consumer Loan': 'loan',
    'Money transfers': 'money_transfer'
}
