# Reviewer 5 - point 9.1
# 9.1 A shallow static network
# Here is a complete implementation of a shallow 
# static neural network (one input layer, one hidden layer, 
# and one output layer) built entirely in NumPy.
# It uses the Sigmoid activation function, Mean Squared 
# Error (MSE) as the cost function, and implements 
# full backpropagation and gradient descent 
# to solve a binary classification problem 
# (the classic XOR problem).
from __future__ import annotations

import numpy as np
import random
import time
import sys
import glob
import math
import os
import random
import struct
from dataclasses import dataclass
from typing import List, Sequence

LOWEST = -500000.0
HIGHEST = 500000.0

@dataclass
class LearningSet:
    total_patterns: int = 0
    inputs: int = 0
    outputs: int = 0
    rate: float = 0.0
    smoothing: float = 0.0
    tolerance: float = 0.0
    iteration_counter: int = 0
    convergence_check: int = 0


def normalise(value: float, minimum: float, maximum: float) -> float:
    if maximum == minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)

def pattern_normaliser(
    filename: str, smallest: Sequence[float], biggest: Sequence[float], inputs: int
) -> List[float]:
    values = read_numbers(filename)
    if len(values) != inputs:
        raise ValueError(
            f"{filename} has {len(values)} values, but the network expects {inputs}."
        )
    return [1.0] + [
        normalise(values[i - 1], smallest[i], biggest[i]) for i in range(1, inputs + 1)
    ]

def pause(prompt: str = "press enter to continue") -> None:
    input(prompt)

def ask_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please insert an integer value.")

def ask_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please insert a numeric value.")

def class_pattern(current_pattern: str, selector: str) -> str:
    directory = os.path.dirname(current_pattern) or "."
    selector = selector.strip()

    if not selector:
        return os.path.join(directory, "*")
    if "*" in selector or "?" in selector:
        return os.path.join(directory, selector)
    if selector.startswith("."):
        return os.path.join(directory, f"*{selector}")
    return os.path.join(directory, f"*.{selector}")

def files_for_pattern(path_pattern: str) -> List[str]:
    return sorted(p for p in glob.glob(path_pattern) if os.path.isfile(p))

def ordered_training_files(path_pattern: str, selectors: Sequence[str]) -> List[str]:
    result: List[str] = []
    for selector in selectors:
        result.extend(files_for_pattern(class_pattern(path_pattern, selector)))
    return result

def make_desired_output(class_counts: Sequence[int], outputs: int) -> List[List[float]]:
    desired: List[List[float]] = []
    for class_index in range(outputs):
        row = [0.0] * (outputs + 1)
        row[class_index + 1] = 1.0
        for _ in range(class_counts[class_index]):
            desired.append(row[:])
    return desired

def read_numbers(filename: str) -> List[float]:
    if os.path.splitext(filename)[1].upper() in {".A1", ".A2"}:
        with open(filename, "rb") as stream:
            data = stream.read()
        if len(data) % 8 != 0:
            raise ValueError(
                f"{filename} has {len(data)} bytes, which is not a whole number of doubles."
            )
        return list(struct.unpack(f"<{len(data) // 8}d", data))

    with open(filename, "r", encoding="utf-8") as stream:
        return [float(line.strip()) for line in stream if line.strip()]

def pattern_handler(filename: str, inputs: int) -> List[float]:
    values = read_numbers(filename)
    if len(values) != inputs:
        raise ValueError(
            f"{filename} has {len(values)} values, but the network expects {inputs}."
        )
    return [1.0] + values

def set_learning_variables(path_pattern: str) -> tuple[LearningSet, List[str], List[int]]:
    all_files = files_for_pattern(path_pattern)
    if not all_files:
        raise FileNotFoundError(f"cannot find FIRST FILE of the list: {path_pattern}")

    print(f"FIRST FILE of the list: {os.path.basename(all_files[0])}")
    inputs = len(read_numbers(all_files[0]))
    print(f"INPUT NEURONS ={inputs}")
    print(f"LEARNING PATTERNS ={len(all_files)}")

    outputs = ask_int("INSERT number of classes to discriminate (OUTPUT NEURONS)\n")
    selectors: List[str] = []
    class_counts: List[int] = []

    for class_index in range(1, outputs + 1):
        selector = input(f"INSERT file extension of {class_index} class [letter, .ext, or pattern]\n")
        matches = files_for_pattern(class_pattern(path_pattern, selector))
        selectors.append(selector)
        class_counts.append(len(matches))
        print(f"{len(matches)} FILES FOUND")
        for filename in matches:
            print(os.path.basename(filename))

    learning_set = LearningSet()
    learning_set.inputs = inputs
    learning_set.outputs = outputs
    learning_set.total_patterns = sum(class_counts)

    print("CHECK if the learning set is currently correct\n")
    print(f"T={learning_set.total_patterns}")
    print(f"INP={learning_set.inputs}")
    print("HID=1")
    print(f"OUT={learning_set.outputs}")

    return learning_set, selectors, class_counts

# 1. Activation Function & Its Derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # x is assumed to be the output of the sigmoid function: sigmoid(z)
    return x * (1 - x)

# 2. Mean Squared Error (MSE) Cost Function
def mse_cost(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 3. Neural Network Structure
class ShallowNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.lr = learning_rate
        
        # Initialize weights and biases randomly
        #self.W1 = np.random.randn(input_size + 1, hidden_size)
        self.W1 = np.random.randn(2, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        
    def forward(self, X):
        # Input to Hidden Layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        # Hidden to Output Layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
        
    def backward(self, X, y):
        #m = X.shape[0]  # Number of training samples
        m = learning_set.inputs
        # --- Output Layer Error & Gradients ---
        # Derivative of MSE w.r.t y_pred is: -(2/m) * (y - y_pred)
        # Chain rule: dZ2 = dCost/da2 * da2/dz2
        error_output = self.a2 - y 
        dZ2 = error_output * sigmoid_derivative(self.a2) * (2 / m)
        
        dW2 = np.dot(self.a1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        # --- Hidden Layer Error & Gradients ---
        # Chain rule: Pass the gradient back through W2 to the hidden layer
        dX1 = np.dot(dZ2, self.W2.T)
        dZ1 = dX1 * sigmoid_derivative(self.a1)
        
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # --- Gradient Descent Update ---
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            # Backward pass (updates weights)
            self.backward(X, y)
            
            # Print loss tracking every 1000 epochs
            if epoch % 1000 == 0:
                loss = mse_cost(y, predictions)
                print(f"Epoch {epoch:5d} | MSE Loss: {loss:.6f}")

# --- 4. Testing the Network on a Two-Class Problem ---
if __name__ == "__main__":
    random.seed()
    print()
    path_pattern = input("INSERT CURRENT LEARN DIRECTORY [drive:\\directory\\*.*]\n")
    learning_set, selectors, class_counts = set_learning_variables(path_pattern)

    answer = input("\nDIGIT Y to SET THE NETWORK or N to quit\n").strip().upper()
    if answer == "N":
        sys.exit(1)  # Stops the entire program with an error status

    print("\nLEARNING SET STORED\n")
    input("press enter to MAKE A RANGE FOR NORMALISATION")

    training_files = ordered_training_files(path_pattern, selectors)
    if not training_files:
        raise RuntimeError("No training files were found for the selected classes.")

    desired_outputs = make_desired_output(class_counts, learning_set.outputs)
    if len(training_files) != len(desired_outputs):
        raise RuntimeError("Internal class/file count mismatch.")

    for idx, filename in enumerate(training_files):
        pattern = pattern_handler(filename, learning_set.inputs)

    biggest = [0.0] * (learning_set.inputs + 1)
    smallest = [0.0] * (learning_set.inputs + 1)
    for inp in range(1, learning_set.inputs + 1):
        biggest[inp] = LOWEST
        smallest[inp] = HIGHEST

    for filename in training_files:
        pattern = pattern_handler(filename, learning_set.inputs)
        for inp in range(1, learning_set.inputs + 1):
            biggest[inp] = max(biggest[inp], pattern[inp])
            smallest[inp] = min(smallest[inp], pattern[inp])

    print("\nRANGE FOR NORMALISATION\n")
    for inp in range(1, learning_set.inputs + 1):
        print(f"biggest[{inp}]={biggest[inp]}\t smallest[{inp}]={smallest[inp]}")

    pause("press enter to NORMALISE ALL PATTERNS")
    print("please WAIT..")
    normalised_patterns: List[List[float]] = []
    for idx, filename in enumerate(training_files):
        pattern = pattern_normaliser(filename, smallest, biggest, learning_set.inputs)
        normalised_patterns.append(pattern)

    warnings = 0
    for idx, pattern in enumerate(normalised_patterns):
        for inp in range(1, learning_set.inputs + 1):
            if pattern[inp] > 1:
                print(f"WARNING IO[{idx}][{inp}]={pattern[inp]}")
                print(f"IS from file {training_files[idx]}")
                warnings += 1
    if warnings:
        pause("press enter to quit")
        sys.exit()
    print("All inputs correctly normalised")

    pause("press enter to SEARCH FOR MISMATCHING CASES")
    print("please WAIT...")
    for left in range(len(normalised_patterns)):
        for right in range(left + 1, len(normalised_patterns)):
            if normalised_patterns[left] == normalised_patterns[right]:
                if desired_outputs[left] != desired_outputs[right]:
                    print(f"WARNING!! {training_files[left]} is MISMATCHING {training_files[right]}")
                    for inp in range(learning_set.inputs + 1):
                        print(f"IO={normalised_patterns[left][inp]}   IO={normalised_patterns[right][inp]}")
                    pause("press enter to quit")
                    sys.exit()

    print("No mismatching cases found")
    pause("press enter to START THE LEARNING PROCESS")


for idx, filename in enumerate(training_files):
    pattern = pattern_handler(filename, learning_set.inputs)
    
    # Setup reproducible random states
    np.random.seed(42)
    
    matrix = np.zeros((len(pattern), 2))
   
    for i in range(1, len(pattern)):
        for j in range(1, 2):
            matrix[i][j] = pattern[i]
            
    X = matrix
   
    # Binary targets (Two-class classification)
    y = np.array([0])
    
    search_string = ".A1"
    if search_string in filename:
       y = np.array([1])
    search_string = ".A2"
    if search_string in filename:
       y = np.array([0])
    # Initialize network:  learning_set.inputs, 1 hidden neuron, 1 output
    nn = ShallowNeuralNetwork(input_size=learning_set.inputs, hidden_size=1, output_size=1, learning_rate=0.5)
    
    print("Training the Neural Network...")
    nn.train(X, y, epochs=10001)
    
    # Final Predictions
    print("\nFinal Predictions on:", filename)
    final_outputs = nn.forward(X)
    print(final_outputs)
    
print("Reviwer 5 - point 9.1 (ends here)")