import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

from sklearn.model_selection import train_test_split

i_data_path = 'E:\\Python\\Hand_Gesture\\datasets\\HandData\\'

# Read Data
hand_ok_df = pd.read_csv(i_data_path + 'HandOK.txt')
hand_bye_df = pd.read_csv(i_data_path + 'HandBye.txt')
hand_fist_df = pd.read_csv(i_data_path + 'HandFist.txt')
hand_open_df = pd.read_csv(i_data_path + 'HandOpen.txt')

X = []
y = []
no_of_timesteps = 10


################## Gesture 1 #####################
dataset = hand_ok_df.iloc[:, 1:].values
n_samples = len(dataset)

for i in range(no_of_timesteps, n_samples):
    X.append(dataset[i-no_of_timesteps:i, :])
    y.append((1, 0, 0, 0))

################## Gesture 2 #####################
dataset = hand_bye_df.iloc[:, 1:].values
n_samples = len(dataset)

for i in range(no_of_timesteps, n_samples):
    X.append(dataset[i-no_of_timesteps:i, :])
    y.append((0, 1, 0, 0))

################## Gesture 3 #####################
dataset = hand_fist_df.iloc[:, 1:].values
n_samples = len(dataset)

for i in range(no_of_timesteps, n_samples):
    X.append(dataset[i-no_of_timesteps:i, :])
    y.append((0, 0, 1, 0))

################## Gesture 4 #####################
dataset = hand_open_df.iloc[:, 1:].values
n_samples = len(dataset)

for i in range(no_of_timesteps, n_samples):
    X.append(dataset[i-no_of_timesteps:i, :])
    y.append((0, 0, 0, 1))

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, y_train.shape)
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))
model.compile(optimizer="adam",
              metrics=['accuracy'],
              loss='categorical_crossentropy')
model.fit(X_train,
          # np.array([x for _, x in enumerate(X_train)]),
          y_train,
          epochs=16, batch_size=32,
          validation_data=(X_test,
                           y_test)
          )
model.save('HandModel.h5')