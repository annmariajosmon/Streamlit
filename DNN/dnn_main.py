import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
import pickle

print("---------------------- Downloading Dataset -------------------------\n")

dataset = pd.read_csv(r'C:\Users\jeena\Desktop\Deep learning\Streamlit App\MODELS\RNN\SMSSpamCollection.csv', sep='\t', names=['label', 'message'])

print("----------------------  -------------------------\n")
print(dataset.head())
print("----------------------  -------------------------")
print(dataset.groupby('label').describe())
print("----------------------  -------------------------")
dataset['label'] = dataset['label'].map({'spam': 1, 'ham': 0})
X = dataset['message'].values
y = dataset['label'].values

print("---------------------- Train Test Split -------------------------\n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tokeniser = tf.keras.preprocessing.text.Tokenizer()
tokeniser.fit_on_texts(X_train)

# Save the tokenizer using pickle
with open('dnn_tokeniser.pkl', 'wb') as file:
    pickle.dump(tokeniser, file)

encoded_train = tokeniser.texts_to_sequences(X_train)
encoded_test = tokeniser.texts_to_sequences(X_test)
print(encoded_train[0:2])
print("----------------------  Padding  -------------------------\n")
max_length = 10
padded_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_train, maxlen=max_length, padding='post')
padded_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
print(padded_train[0:2])

print("----------------------  -------------------------\n")

vocab_size = len(tokeniser.word_index) + 1

# define the model

print("---------------------- Modelling -------------------------\n")

dnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=24, input_length=max_length),
    tf.keras.layers.SimpleRNN(24, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile the model
dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("----------------------  -------------------------\n")

# summarize the model
print(dnn_model.summary())

print("----------------------  -------------------------\n")

early_stop = tf.keras.callbacks.EarlyStopping(monitor='accuracy', mode='min', patience=10)

print("----------------------  Training -------------------------\n")

# fit the model
dnn_model.fit(x=padded_train,
              y=y_train,
              epochs=50,
              validation_data=(padded_test, y_test),
              callbacks=[early_stop]
              )
print("----------------------  -------------------------\n")


def c_report(y_true, y_pred):
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    acc_sc = accuracy_score(y_true, y_pred)
    print(f"Accuracy : {str(round(acc_sc, 2) * 100)}")
    return acc_sc


def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5, cmap="Blues", cbar=False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(r'C:\Users\jeena\Desktop\Deep learning\Streamlit App\MODELS\DNN\images\test.jpg')


preds = (dnn_model.predict(padded_test) > 0.5).astype("int32")
c_report(y_test, preds)

plot_confusion_matrix(y_test, preds)

# Save the model using pickle
with open('dnn_model.pkl', 'wb') as file:
    pickle.dump(dnn_model, file)

# Load the model back using pickle
with open('dnn_model.pkl', 'rb') as file:
    dnn_model = pickle.load(file)

# Save the tokenizer using pickle
with open('dnn_tokeniser.pkl', 'wb') as file:
    pickle.dump(tokeniser, file)

# Load the tokenizer back using pickle
with open('dnn_tokeniser.pkl', 'rb') as file:
    tokeniser = pickle.load(file)
