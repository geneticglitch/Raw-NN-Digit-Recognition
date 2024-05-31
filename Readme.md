# Neural Network Implementation for Handwritten Digit Recognition without Using Any Libraries in Java
This Java project implements a neural network from scratch to recognize handwritten digits from 0 to 9, trained on the MNIST dataset. The network architecture, training process, and future enhancements are detailed below.

## Project Overview

### Network Architecture

- **Input Layer:** 784 neurons (28x28 pixel images)
- **Hidden Layer:** 10 neurons, activated by the Rectified Linear Unit (ReLU) function
- **Output Layer:** 10 neurons, activated by the softmax function

### Training Algorithm

The neural network is trained using the Stochastic Gradient Descent (SGD) Backpropagation algorithm.

![Training Output](/images/swappy-20240531_104243.png)

### Future Enhancements

- **Graphical User Interface (GUI):**
  - Implemented using JavaFX.
  - Allows users to draw a digit on the screen, which the neural network will then predict.
- **Utils_Multi Class Update:**
  - The current implementation runs out of memory.
  - An updated version will be released soon.


### Additional Features

- **Test Class:**
    - Evaluates the neural network on the MNIST test dataset.
    - Calculates and reports the accuracy of the neural network.

![Testing Output](/images/swappy-20240531_104942.png)

## Technical Details

- **Programming Language:** Java 22
- **Build Tool:** Maven
- **OS:** Arch Linux x86_64
