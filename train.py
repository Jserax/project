import tensorflow as tf
from tensorflow import keras
from process_data import get_dataframe

max_features = 40000
embedding_dim = 128
sequence_length = 400


x_train, label_train, score_train = get_dataframe('data/aclimdb/train',
                                                  stopword=True,
                                                  lemma=True)
x_test, label_test, score_test = get_dataframe('data/aclimdb/test',
                                               stopword=True,
                                               lemma=True)

vectorize_layer = keras.layers.TextVectorization(
    standardize='lower_and_strip_punctuation',
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
    vocabulary='data/aclimdb/imdb.vocab')


text_input = keras.Input(shape=(1,), dtype=tf.string, name='text')
x = vectorize_layer(text_input)
x = keras.layers.Embedding(max_features + 1, embedding_dim)(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Bidirectional(keras.layers.LSTM(128,
                                                 return_sequences=True))(x)
x = keras.layers.Bidirectional(keras.layers.LSTM(64))(x)
x = keras.layers.Dropout(0.5)(x)
x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.Dropout(0.5)(x)
label = keras.layers.Dense(1, activation='sigmoid', name='label')(x)
score = keras.layers.Dense(1,
                           activation=keras.layers.ReLU(max_value=10),
                           name='score')(x)
model = keras.Model(inputs=text_input, outputs=[label, score])
lr = keras.optimizers.schedules.ExponentialDecay(0.005, 782,
                                                 0.85, staircase=True)
model.compile(loss={'label': 'binary_crossentropy',
                    'score': 'mse'},
              loss_weights={'label': 1, 'score': 1},
              optimizer=keras.optimizers.Nadam(learning_rate=lr),
              metrics={'label': 'accuracy',
                       'score': 'mae'})
model.fit(x=x_train, y={'label': label_train, 'score': score_train},
          epochs=20, batch_size=32,  shuffle=True)
model.save('webapp/model/model')
print(model.evaluate(x_test, {'label': label_test, 'score': score_test}))
