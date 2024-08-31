# Fashion MNIST Image Classification with TensorFlow

## Overview

This project demonstrates the process of building, training, and evaluating a Convolutional Neural Network (CNN) model using TensorFlow to classify images from the Fashion MNIST dataset. The model achieves an accuracy of 90.36% on the test set, providing a robust solution for classifying various types of clothing items.

## Project Structure

- **data_preprocessing.py**: Contains scripts for loading and preprocessing the Fashion MNIST dataset, including normalization and one-hot encoding of labels.
- **model_building.py**: Defines the architecture of the CNN using both Sequential and Functional APIs in TensorFlow.
- **training.py**: Includes code for training the CNN model on the training dataset, as well as saving the model architecture and weights.
- **evaluation.py**: Evaluates the model's performance using a confusion matrix, classification report, and ROC curves.
- **report_generation.py**: Generates a comprehensive PDF report summarizing the model's performance, including visualizations of results.

## Key Results

- **Accuracy**: The model achieved 90.36% accuracy on the test dataset.
- **Precision & Recall**: High precision and recall across most classes, with detailed metrics available in the report.
- **ROC Curves**: Demonstrates strong classification performance, with most classes achieving near-perfect AUC scores.

## Usage

1. Clone the repository.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the scripts in sequence to preprocess data, build the model, train, and evaluate.
4. View the generated `model_report.pdf` for detailed insights.

## Conclusion

The CNN model provides a solid foundation for image classification tasks with Fashion MNIST, with room for further tuning to enhance performance. The project can be extended to explore different architectures, data augmentation techniques, or more complex datasets.
