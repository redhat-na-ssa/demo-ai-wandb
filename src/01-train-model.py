#!/usr/bin/env python
# coding: utf-8
#
# This python script was converted from the fraud detection
# notebook at https://github.com/rh-aiservices-bu/fraud-detection/blob/main/1_experiment_train.ipynb
#
# Work in progress.

import numpy as np
import pandas as pd
import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
import tf2onnx
import onnx
import pickle
from pathlib import Path
import wandb



# ## Load the CSV data
# 
# The CSV data that you use to train the model contains the following fields:
# 
# * **distancefromhome** - The distance from home where the transaction happened.
# * **distancefromlast_transaction** - The distance from the last transaction that happened.
# * **ratiotomedianpurchaseprice** - The ratio of purchased price compared to median purchase price.
# * **repeat_retailer** - If it's from a retailer that already has been purchased from before.
# * **used_chip** - If the credit card chip was used.
# * **usedpinnumber** - If the PIN number was used.
# * **online_order** - If it was an online order.
# * **fraud** - If the transaction is fraudulent.


Data = pd.read_csv('data/card_transdata.csv')
Data.head()

# Set the input (X) and output (Y) data. 
# The only output data is whether it's fraudulent. All other fields are inputs to the model.

X = Data.drop(columns = ['repeat_retailer','distance_from_home', 'fraud'])
y = Data['fraud']

# Split the data into training and testing sets so you have something to test the trained model with.

# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify = y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, shuffle = False)

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.2, stratify = y_train)

# Scale the data to remove mean and have unit variance. The data will be between -1 and 1, which makes it a lot easier for the model to learn than random (and potentially large) values.
# It is important to only fit the scaler to the training data, otherwise you are leaking information about the global distribution of variables (which is influenced by the test set) into the training set.

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train.values)

Path("artifact").mkdir(parents=True, exist_ok=True)
with open("artifact/test_data.pkl", "wb") as handle:
    pickle.dump((X_test, y_test), handle)
with open("artifact/scaler.pkl", "wb") as handle:
    pickle.dump(scaler, handle)

# Since the dataset is unbalanced (it has many more non-fraud transactions than fraudulent ones), set a class weight to weight the few fraudulent transactions higher than the many non-fraud transactions.

class_weights = class_weight.compute_class_weight('balanced',classes = np.unique(y_train),y = y_train)
class_weights = {i : class_weights[i] for i in range(len(class_weights))}


# ## Build the model
# 
# The model is a simple, fully-connected, deep neural network, containing three hidden layers and one output layer.


model = Sequential()
model.add(Dense(32, activation = 'relu', input_dim = len(X.columns)))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()


# ## Train the model
# 
# Training a model is often the most time-consuming part of the machine learning process.  Large models can take multiple GPUs for days.  Expect the training on CPU for this very simple model to take a minute or more.


# Train the model and get performance
import os

epochs = 2
validation_data=(scaler.transform(X_val.values),y_val)
history = model.fit(X_train, y_train,
    epochs=epochs,
    validation_data=validation_data,
    verbose = True,
    class_weight = class_weights)

print("Training of model is complete")


# Save the model file

# Save the model as ONNX for easy use of ModelMesh
model_proto, _ = tf2onnx.convert.from_keras(model)
os.makedirs("models/fraud/1", exist_ok=True)
onnx.save(model_proto, "models/fraud/1/model.onnx")


# ## Confirm the model file was created successfully
# 
# The output should include the model name, size, and date. 



os.system(' ls -alRh ./models/')


# ## Test the model



from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import onnxruntime as rt


# Load the test data and scaler:


with open('artifact/scaler.pkl', 'rb') as handle:
    scaler = pickle.load(handle)
with open('artifact/test_data.pkl', 'rb') as handle:
    (X_test, y_test) = pickle.load(handle)


# Create an ONNX inference runtime session and predict values for all test inputs:


sess = rt.InferenceSession("models/fraud/1/model.onnx", providers=rt.get_available_providers())
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name
y_pred_temp = sess.run([output_name], {input_name: scaler.transform(X_test.values).astype(np.float32)}) 
y_pred_temp = np.asarray(np.squeeze(y_pred_temp[0]))
threshold = 0.95
y_pred = np.where(y_pred_temp > threshold, 1,0)


# Show the results:


accuracy = np.sum(np.asarray(y_test) == y_pred) / len(y_pred)
print("Accuracy: " + str(accuracy))

c_matrix = confusion_matrix(np.asarray(y_test),y_pred)
ax = sns.heatmap(c_matrix, annot=True,fmt='d', cbar=False, cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
ax.set_title('Confusion Matrix')
plt.show()
plot_file = "fraud-confusion-matrix.png"
plt.savefig(plot_file)


# ## Example: Is Sally's transaction likely to be fraudulent?
# 
# Here is the order of the fields from Sally's transaction details:
# * distance_from_last_transaction
# * ratio_to_median_price
# * used_chip 
# * used_pin_number
# * online_order 


sally_transaction_details = [
    [0.3111400080477545,
    1.9459399775518593, 
    1.0, 
    0.0, 
    0.0]
    ]

prediction = sess.run([output_name], {input_name: scaler.transform(sally_transaction_details).astype(np.float32)})

print("Is Sally's transaction predicted to be fraudulent? (true = YES, false = NO) ")
print(np.squeeze(prediction) > threshold)

print("How likely was Sally's transaction to be fraudulent? ")
print("{:.5f}".format(np.squeeze(prediction)) + "%")

#
# Begin Weights and Biases
#

wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key == None:
    print("WANDB_API_KEY environment variable not set.  Skipping W&B logging.")
else:


    wandb.login(
        host="http://local.wandb:8080",
        relogin=False,
        key=wandb_api_key,)

    #  1. Initialize a new run
    wandb.init(
        project="fraud-detection",
        notes="The Fraud Detection Demo from Openshift AI",
        tags=["fraud-detection", "demo"],
    )

    #
    # This code should be in a loop
    # that runs through epochs, hyper parms
    # and logs the results to W&B.
    #
    #  2. Capture a dictionary of hyperparameters
    wandb.config = {"epochs": epochs}

    #  3. Log metrics and artifacts
    wandb.log({"epochs": epochs})
    wandb.log({"accuracy": accuracy})
    # wandb.log({"loss": loss})
    wandb.log({"confusion_matrix": wandb.Image("fraud-confusion-matrix.png")})
    wandb.log({"sally_transaction_details": sally_transaction_details})
    wandb.log({"sally_transaction_prediction": np.squeeze(prediction)})
    wandb.log({"sally_transaction_likelihood": np.squeeze(prediction) * 100})
    wandb.log({"threshold": threshold})

    #  4. Save the model
    model_artifact = wandb.Artifact(
        name = "fraud-model",
        type = "model")
    model_artifact.add_dir("models")
    wandb.log_artifact(model_artifact)

    wandb.finish()
