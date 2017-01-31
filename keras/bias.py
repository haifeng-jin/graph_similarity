from keras.models import Sequential
from keras.layers import Dense, Activation

# for a single-input model with 2 classes (binary):
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(1, input_dim=1))
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

# generate dummy data
data = list(map(lambda x: [x], range(0, 100)))
labels = list(map(lambda x: [2 * x + 3], range(0, 100)))

# train the model, iterating on the data in batches
# of 32 samples
model.fit(data, labels, nb_epoch=5000)
scores = model.evaluate(data, labels)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.evaluate(data, labels)
for layer in model.layers:
    print(layer.get_weights())

