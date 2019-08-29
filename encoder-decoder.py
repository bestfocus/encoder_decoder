# This is an example of encoder decoder
# The encoder decoder borrows from an optional model in the NLP Machine Translation project on Udacity
import numpy as np
import random

from keras.models import Model
from keras.layers import (Dense, Input, TimeDistributed, RepeatVector, GRU)
from keras.losses import sparse_categorical_crossentropy
from keras.optimizers import Adam

x_size=22
y_size=27
x=np.array([random.randint(0,100) for _ in range(3*10)]).reshape((3,10,1))
y=np.array([random.randint(0,100) for _ in range(3*14)]).reshape((3,14,1))

learning_rate = 1e-3
input_shape=x.shape
output_sequence_length=14
print(x.shape, output_sequence_length, x_size, y_size)
input_seq = Input(input_shape[1:])
rnn1=GRU(128, dropout=0.1)(input_seq)
rnn2=Dense(128,activation="relu")(rnn1)
enc=RepeatVector(output_sequence_length)(rnn2)
rnn=GRU(256,return_sequences=True, dropout=0.1)(enc)
logits = TimeDistributed(Dense(y_size,activation="softmax"))(rnn)
model = Model(input_seq, logits)
model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(learning_rate),
              metrics=['accuracy'])

model.fit(x, y, batch_size=1024, epochs=30, validation_split=0.2)
