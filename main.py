import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential, Model, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from matplotlib.backends.backend_pdf import PdfPages  # Import for saving to PDF

# Check TensorFlow installation
print("TensorFlow version:", tf.__version__)

# Load the Fashion MNIST dataset
(images_train, labels_train), (images_test, labels_test) = fashion_mnist.load_data()

# Reshape labels to fit OneHotEncoder
labels_train = labels_train.reshape(-1, 1)
labels_test = labels_test.reshape(-1, 1)

# One-hot encode the labels
onehot = OneHotEncoder(sparse_output=False)
labels_train = onehot.fit_transform(labels_train)
labels_test = onehot.transform(labels_test)

# Reshape images to add a single channel dimension and normalize
images_train = images_train.reshape(60000, 28, 28, 1).astype('float32') / 255
images_test = images_test.reshape(10000, 28, 28, 1).astype('float32') / 255

# Sequential API Model
seq_model = Sequential()
seq_model.add(Conv2D(32, (2, 2), input_shape=(28, 28, 1), activation='relu', padding='same'))
seq_model.add(MaxPooling2D(pool_size=(2, 2)))
seq_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
seq_model.add(MaxPooling2D(pool_size=(2, 2)))
seq_model.add(Conv2D(32, (2, 2), activation='relu', padding='same'))
seq_model.add(MaxPooling2D(pool_size=(2, 2)))
seq_model.add(Flatten())
seq_model.add(Dense(1024, activation='relu'))
seq_model.add(Dropout(0.2))
seq_model.add(Dense(200, activation='relu'))
seq_model.add(Dropout(0.2))
seq_model.add(Dense(10, activation='softmax'))  # 10 classes

# Compile Sequential model
seq_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Functional API Model
image = Input(shape=(28, 28, 1))
conv1 = Conv2D(32, (5, 5), activation='relu', padding='same')(image)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, (5, 5), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat_layer = Flatten()(pool2)
dense1 = Dense(1024, activation='relu')(flat_layer)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(200, activation='relu')(drop1)
drop2 = Dropout(0.2)(dense2)
output = Dense(10, activation='softmax')(drop2)

func_model = Model(inputs=image, outputs=output)

# Compile Functional model
func_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
func_model.fit(images_train, labels_train, validation_data=(images_test, labels_test), epochs=20, batch_size=5000, verbose=1)

# Save the model
model_json = func_model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
func_model.save_weights("model.weights.h5")

# Load the model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.weights.h5")

# Compile loaded model
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Make predictions
p = loaded_model.predict(images_test)
labels_test_predicted = np.argmax(p, axis=1)

# Evaluate predictions
y_true = np.argmax(labels_test, axis=1)
accuracy = (y_true == labels_test_predicted).sum() / len(y_true)
print(f'Accuracy: {accuracy:.4f}')

# Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_true, labels_test_predicted)
class_report = classification_report(y_true, labels_test_predicted)

print(conf_matrix)
print(class_report)

# ROC Curve for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(10):  # 10 classes
    fpr[i], tpr[i], _ = roc_curve(labels_test[:, i], p[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Create a PDF report
with PdfPages('model_report.pdf') as pdf:
    # Accuracy
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, f'Accuracy: {accuracy:.4f}', horizontalalignment='center', verticalalignment='center', fontsize=16)
    plt.axis('off')
    pdf.savefig()
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap='Blues', interpolation='none')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(1, 11)  # For labels 1 through 10
    plt.xticks(tick_marks - 1, tick_marks, rotation=0)  # Adjust ticks for 1-10
    plt.yticks(tick_marks - 1, tick_marks)  # Adjust ticks for 1-10
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    pdf.savefig()
    plt.show()  # Display the confusion matrix
    plt.close()

    # Classification Report
    plt.figure(figsize=(10, 8))
    plt.text(0.01, 0.05, class_report, {'fontsize': 12}, fontproperties='monospace')
    plt.axis('off')
    pdf.savefig()
    plt.close()

    # ROC Curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(10):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for each class')
    plt.legend(loc='lower right')
    pdf.savefig()
    plt.show()  # Display the ROC curves
    plt.close()

    # Misclassified image
    ind = 4369  # Change this index to view different misclassified images
    sample_image = images_test[ind, :, :, :]
    pixels = sample_image.reshape((28, 28))

    plt.figure(figsize=(8, 6))
    plt.imshow(pixels, cmap='gray')
    plt.title(f'Real label: {y_true[ind] + 1}, Predicted label: {labels_test_predicted[ind] + 1}')
    plt.axis('off')
    pdf.savefig()
    plt.show()  # Display the misclassified image
    plt.close()

    # Additional text if needed
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, 'End of Report', horizontalalignment='center', verticalalignment='center', fontsize=16)
    plt.axis('off')
    pdf.savefig()
    plt.close()

print("Report generated: 'model_report.pdf'")

# Further training (optional)
loaded_model.fit(images_train, labels_train, validation_data=(images_test, labels_test), epochs=3, batch_size=50, verbose=1)
