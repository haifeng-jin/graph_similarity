from keras.models import Sequential
from keras.layers import Dense, Activation

# for a single-input model with 2 classes (binary):
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(3, input_dim=2, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
sgd = SGD(lr=0.5)
model.compile(optimizer=sgd,
              loss='mean_squared_error',
              metrics=['accuracy'])

# generate dummy data
# import numpy as np
# data = np.random.randint(2, size=(4, 2))
# labels = np.random.randint(2, size=(4, 1))
data = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [[0], [1.0], [1.0], [0]]

# train the model, iterating on the data in batches
# of 32 samples
model.fit(data, labels, nb_epoch=2000, batch_size=4)
scores = model.evaluate(data, labels)
print(model.predict_proba(data))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


print(data)
print(labels)

