# Jupyter Notebook referenced from [Mr. Cristian Leo](https://towardsdatascience.com/the-math-behind-neural-networks-a34a51b93873)



This project demonstrates how to implement a basic neural network from scratch using Python and NumPy. The neural network is designed, trained, and evaluated on the digit recognition task using the popular MNIST dataset. The project covers several key aspects of machine learning, including data preprocessing, model architecture design, training and testing the model, and evaluating its performance with different metrics.

## Overview

The neural network implemented in this project includes a single hidden layer. Users can easily customize the model's architecture, including the number of neurons in the hidden layer, the learning rate, and the number of training epochs. The project uses mean squared error, logistic loss, and categorical cross-entropy as loss functions, demonstrating how to apply them in a training scenario.

## Key Features

- **Model Implementation**: Step-by-step implementation of a neural network, including forward and backward propagation.
- **Loss Functions**: Implementation of multiple loss functions to understand their impact on training.
- **Training and Evaluation**: Detailed training process with evaluations on both training and test datasets.
- **Visualization**: Visualization of the training process, model's predictions, and a comparison of true vs. predicted labels.
- **Hyperparameter Tuning**: Demonstration of hyperparameter tuning using Optuna to find the optimal model configuration.

## How to Use

To use this project:

1. **Set Up Your Environment**: Ensure you have Python installed and create a virtual environment. Install the required packages using `pip install numpy matplotlib sklearn optuna`.

2. **Load the Data**: The project uses the MNIST digits dataset, which is loaded and preprocessed for neural network training.

3. **Configure the Neural Network**: Adjust the neural network's parameters, such as the number of neurons in the hidden layer, learning rate, and epochs, as needed.

4. **Train the Model**: Run the training process and observe the output metrics, including loss and accuracy.

5. **Evaluate and Visualize**: Use the provided visualization tools to analyze the model's performance, understand the loss over epochs, and inspect misclassified examples.

6. **Hyperparameter Tuning**: Optionally, use Optuna for hyperparameter tuning to enhance model performance.

## Further Visualization

The project includes several visualization techniques to better understand and evaluate the model's performance:

- **Confusion Matrix**: Understand the model's prediction accuracy across different classes.
- **Learning Curves**: Observe how the loss and accuracy evolve during training.
- **Misclassified Examples**: Inspect which images were misclassified to gain insights into potential improvements.

## Future Work

Future enhancements could include experimenting with different model architectures, implementing more sophisticated optimization algorithms, or applying the model to other datasets or tasks.


