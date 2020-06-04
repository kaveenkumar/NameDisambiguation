#libs & paths
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras.optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.models import load_model
from matplotlib import pyplot



inputData = pd.read_csv("training_track1/result.txt", sep=',', header=None)

x = inputData.iloc[:,0:16]
y = inputData.iloc[:,16].astype('int')


# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=100)

x_train = x
y_train = y

dropout = 0.4

#define model
model = Sequential()
model.add(Dense(100, input_dim=16, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(100, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))
model.summary()

#compile model

adam = keras.optimizers.Nadam(learning_rate = 0.0001)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])


#set early stop to avoid overfitting and checkpoint to revert to best accuracy iteration
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10, min_delta=0.00001)
mc = ModelCheckpoint('bestmodel.h5', monitor='acc', verbose=1, save_best_only=True, mode='max')

history=model.fit(x_train, y_train, epochs=100, callbacks=[es,mc], batch_size=128)

#save model
model.save("bestmodel.h5")