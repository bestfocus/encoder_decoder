# This is an example of encoder decoder
import numpy as np

learning_rate = 1e-3
print(input_shape, output_sequence_length, english_vocab_size, french_vocab_size)
input_seq = Input(input_shape[1:])
rnn1=GRU(128, dropout=0.1)(input_seq)
rnn2=Dense(128,activation="relu")(rnn1)
enc=RepeatVector(output_sequence_length)(rnn2)
rnn=GRU(256,return_sequences=True, dropout=0.1)(enc)
logits = TimeDistributed(Dense(french_vocab_size,activation="softmax"))(rnn)
model = Model(input_seq, logits)
model.compile(loss=sparse_categorical_crossentropy,
              optimizer=Adam(learning_rate),
              metrics=['accuracy'])
