import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from numpy import array
import numpy as np

# Load the processed data that is saved in csv format
df = pd.read_csv('../data_preprocessing/all_df_new_features.csv')
df.info()

# Drop the first columns which contained the label of each row e.g. a1_s1_t1
df = df.drop(df.columns[0], axis=1)
df.info()

# We assigned the target column to the variable target for generating the train and test target data
target = df['target']

# we convert the df to numpy array for ease of slicing
df = array(df)

# Identify the dimension of our data 27 categories, 78 features and 41 frames
data_dim = 78
time_steps = 41
num_classes = 27

# We need the train data and test data to be of specific shape,
# hence we initialized these variables first with zeros or random integers
# split data into 30% test and 70% train data out of 861 rows
x_train = np.zeros((604, data_dim, time_steps))
x_val = np.zeros((257, data_dim, time_steps))
y_train = np.random.randint(10, size=(604, 1))
y_val = np.random.randint(10, size=(257, 1))

# when we save our processed data in to csv, the 3rd dimension was recognised as string type
# Hence, we got to reprocess it to make it into numpy array
for i in range(df.shape[0]):
    for j in range(df.shape[1]-1):
        frame_string = []
        frame_float = []
        df[i][j] = df[i][j].replace('[', '')
        df[i][j] = df[i][j].replace(']', '')
        df[i][j] = df[i][j].replace(',', '')
        frame = df[i][j].split()
        for k in frame:
            frame_float.append(float(k))
        frame_float = array(frame_float)
        df[i][j] = frame_float

# Refer to 'https://keras.io/getting-started/sequential-model-guide/' for more information
# expected input data shape: (batch_size, data_dim, time_steps)
model = Sequential()
model.add(LSTM(256, return_sequences=True,
               input_shape=(data_dim, time_steps)))  # returns a sequence of vectors of dimension 128
model.add(LSTM(512, return_sequences=True))  # returns a sequence of vectors of dimension 128
model.add(LSTM(512))  # return a single vector of dimension 128
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

# Generate training data
for i in range(x_train.shape[0]):
    for j in range(x_train.shape[1]):
        for k in range(x_train.shape[2]):
            x_train[i][j][k] = df[i][j][k]
print(x_train.shape)

# Generate training target data
y_train_array = array(target[:604])
for i in range(len(y_train_array)):
    y_train[i][0] = y_train_array[i]
print(y_train.shape)

# performed one hot encoding
# class vector to be converted into a matrix (integers from 0 to num_classes)
# Hence, we got to minus 1 from our y_train
y_train = keras.utils.to_categorical(y_train-1, num_classes=num_classes)
print(y_train.shape)

# Generate validation data
h = 0
for i in range(x_train.shape[0], x_train.shape[0] + x_val.shape[0]):
    for j in range(x_val.shape[1]):
        for k in range(x_val.shape[2]):
            x_val[h][j][k] = df[i][j][k]
    h = h + 1
print(x_val.shape)

# Generate validation target data
y_val_array = array(target[604:])
for i in range(len(y_val_array)):
    y_val[i][0] = y_val_array[i]
print(y_val.shape)

# performed one hot encoding
# class vector to be converted into a matrix (integers from 0 to num_classes)
# Hence, we got to minus 1 from our y_train
y_val = keras.utils.to_categorical(y_val-1, num_classes=num_classes)
print(y_val.shape)

# train model
model.fit(x_train, y_train, batch_size=32, epochs=50, validation_data=(x_val, y_val))

# Return the predicted result
results = model.predict(x_val, batch_size=32, verbose=1)

# Reformat the predicted result as a column of predicted class
classes = []
for i in range(1, 28):
    classes.append(i)
prediction = [int(classes[int(np.argmax(i))]) for i in results]

# Save predictions to csv
prediction_output = pd.DataFrame()
prediction_output['Predictions'] = prediction
prediction_output['Actual'] = y_val_array
prediction_output.to_csv('../results/predictions.csv')
