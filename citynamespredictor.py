from __future__ import print_function, division, absolute_import

import tflearn
from tflearn.data_utils import *

path = "indiancitynames.txt"

#Fixing maximum length of city names to be generated
maxlen = 20

X, Y, char_idx = \
    textfile_to_semi_redundant_sequences(path, seq_maxlen=maxlen, redun_step=3)

#LSTM RNN 
net = tflearn.input_data(shape=[None, maxlen, len(char_idx)])
net = tflearn.lstm(net, 512, return_seq=True, dropout=0.5)
net = tflearn.lstm(net, 512, dropout=0.5)
net = tflearn.fully_connected(net, len(char_idx), activation='softmax')
net = tflearn.regression(g, optimizer='adam', loss='categorical_crossentropy',
                       learning_rate=0.001)

#A deep neural network model for generating sequences.
model = tflearn.SequenceGenerator(net, dictionary=char_idx,
                              seq_maxlen=maxlen,
                              clip_gradients=5.0,
                              checkpoint_path='model_indian_cities')

#training
for i in range(40):
    #Seed helps us start from the same place everytime
    seed = random_sequence_from_textfile(path, maxlen)
    #Fitting data to the sequence
    model.fit(X, Y, validation_set=0.1, batch_size=128,
          n_epoch=1, run_id='indian_cities')
    print("\n...TESTING...")
    print("-- Test with temperature of 1.2 --")
    """Generate a sequence. Temperature is controlling the novelty of the created sequence, 
       a temperature near 0 will looks like samples used for training, 
       while the higher the temperature, the more novelty."""
    print(model.generate(30, temperature=1.2, seq_seed=seed))
    print("-- Test with temperature of 1.0 --")
    print(model.generate(30, temperature=1.0, seq_seed=seed))
    print("-- Test with temperature of 0.5 --")
    print(model.generate(30, temperature=0.5, seq_seed=seed))
