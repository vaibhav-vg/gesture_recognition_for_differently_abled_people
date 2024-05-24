
# import pickle
#
# data_dict = pickle.load(open('./data.pickle','rb'))
#
# # i = 0
# # while data_dict["labels"][i] != '10':
# #     i += 1
# # print(data_dict["data"][i])
#
# #
# print(data_dict["data"])
# print(data_dict["labels"])

# import pickle
#
# data_dict = pickle.load(open('./data.pickle', 'rb'))
#
# print(data_dict.keys())
# print(data_dict)






import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()







# import os

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import pickle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras  # Import for Keras functionalities

# # Load data (assuming data.pickle remains the same)
# data_dict = pickle.load(open('./data.pickle', 'rb'))
# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# # Train-test split
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# # Train the RandomForestClassifier model
# model = RandomForestClassifier()
# model.fit(x_train, y_train)

# # Evaluate the model (optional)
# y_predict = model.predict(x_test)
# score = accuracy_score(y_predict, y_test)
# print('{}% of samples were classified correctly !'.format(score * 100))

# # Create a simple Keras model mimicking the RandomForestClassifier
# inputs = keras.Input(shape=x_train.shape[1:])
# x = keras.layers.Flatten()(inputs)  # Adjust layers based on your model
# outputs = keras.layers.Dense(len(np.unique(y_train)))(x)  # Output layer
# model_keras = keras.Model(inputs=inputs, outputs=outputs)

# # Compile the Keras model (optional, loss and optimizer based on your problem)
# model_keras.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Optionally train the Keras model further (for fine-tuning)
# # model_keras.fit(x_train, y_train, epochs=...)

# # Save the trained Keras model in TensorFlow SavedModel format
# saved_model_path = 'saved_model'
# tf.saved_model.save(model_keras, saved_model_path)

# # Print instructions for converting the SavedModel to TensorFlow.js format
# # print("** Conversion to TensorFlow.js (requires separate execution) **")
# # print("Install TensorFlow.js converter: npm install @tensorflow/tfjs-converter")
# # print("Run the following command to convert the model:")
# # print(f"tensorflowjs_converter \\\n"
# #       f"\t--input_format=saved_model \\\n"
# #       f"\t--output_format=tfjs_layers_model \\\n"
# #       f"\t--saved_model_tags=serve \\\n"
# #       f"\t{saved_model_path}/ \\\n"a
# #       f"\t{saved_model_path}/tfjs_model")
