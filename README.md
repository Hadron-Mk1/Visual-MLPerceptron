# Neural Network From Scratch

---

![Screenshot](images/SLPerceptron.png)

## Why we chose to make a Perceptron
We both had limited previous experience in Python so we decided to create and develop a single and multi-layer 
perceptron.
This would allow us to push our boundaries as well as build upon knowledge that we had gained in 
our Artificial Intelligence module where we were introduced to the concept of machine learning and its origins.


## Single-Layer to Multi-Layer

A single-layer perceptron was what we attempted first as this would be able to provide the fundamentals of the
perceptron such as weights and biases. This gave us a better understanding of how the learning process worked and how 
this could be applied to a multi-layer network where additional layers are used to abstract and learn more complex 
relationships between data

The multi-layer perceptron did create more challenges as we had to adapt weights and biases so that they were
suitable to be passed through multiple layers. This required more understanding of the equations that perceptrons use
to learn and added complexity through the addition of forward and backward propagation and error correction.

---

This Python project explores the fundamentals of neural networks by implementing them from scratch using NumPy.

The project progresses from a basic single-layer perceptron into a multi-layer perceptron capable of classifying images 
from the Fashion-MNIST dataset.

---

## The Single-Layer Perceptron
## Explain challenges and how we overcame them as well as what it does
The first stage implements a single-layer perceptron designed to classify linearly separable data.

### Features

- Weight and bias updates
- Binary threshold activation
- Perceptron learning algorithm
- Training and testing data
- Accuracy calculation
- Plotting of the decision boundary

The model is trained on generated data, with the learned decision boundary plotted to demonstrate how the perceptron
separates the two classes.

---

## The Multi-Layer Perceptron
## Explain challenges and how we overcame them as well as what it does
The network is trained to classify images from Fashion-MNIST, which contains 10 different categories of clothing.

### Features

- Multiple hidden layers
- ReLU activation
- Softmax + Cross-entropy learning method
- One-hot encoded labels
- Forward propagation
- Backpropagation
- Gradient-based weight and bias updates
- Fashion-MNIST image classification
- Training accuracy visualisation

---

## Libraries used

- Python
- NumPy
- Matplotlib
- sklearn

---

## Explain what we've learnt

## Explain how we can improve

## Explain req to run code i.e. how to import numpy