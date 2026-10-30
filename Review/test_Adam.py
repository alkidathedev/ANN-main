# Reviewer 5 - point 9.2
# A modern MLP baseline using ReLU

import numpy as np
import sys
import glob
from dataclasses import dataclass
from typing import List, Sequence
import os
import random
import struct

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
    print(f"OUT={learning_set.outputs}")

    return learning_set, selectors, class_counts

# ==========================================
# 1. NETWORK COMPONENTS (BIAS TRICK STYLE)
# ==========================================

class DenseLayer:
    # Accept both arguments so your existing instantiation code doesn't crash
    def __init__(self, input_dim, output_dim):
        self.output_dim = output_dim
        self.weights = None  # We will initialize this dynamically to prevent shape errors
        self.x_with_bias = None
        self.dweights = None

    def forward(self, x):
        # 1. Create a column vector of ones matching the exact batch size (rows) of x
        ones = np.ones((x.shape[0], 1))
        
        # 2. Horizontally stack to add the bias feature column
        # If x is (2, 24697), self.x_with_bias becomes (2, 24698)
        self.x_with_bias = np.hstack((x, ones))
        
        # 3. Dynamic Initialization: Automatically matches the real feature dimension of your data
        if self.weights is None:
            num_features_with_bias = self.x_with_bias.shape[1] # Will be 24698
            actual_input_dim = num_features_with_bias - 1      # Will be 24697
            
            self.weights = np.random.randn(num_features_with_bias, self.output_dim) * np.sqrt(2.0 / actual_input_dim)

        # 4. Matrix multiplication: (2, 24698) dot (24698, output_dim)
        return np.dot(self.x_with_bias, self.weights)

    def backward(self, dout):
        self.dweights = np.dot(self.x_with_bias.T, dout)
        return np.dot(dout, self.weights[:-1, :].T)


    def backward(self, dout):
        # Gradient with respect to weights: (24698, batch_size) dot (batch_size, output_dim)
        self.dweights = np.dot(self.x_with_bias.T, dout)
        # Gradient with respect to input: drops the bias gradient row at the bottom
        return np.dot(dout, self.weights[:-1, :].T)


    def backward(self, dout):
        # Gradient with respect to weights: (24698, 2) dot (2, 2) -> (24698, 2)
        self.dweights = np.dot(self.x_with_bias.T, dout)
        # Gradient with respect to input: (2, 2) dot (2, 24697) -> (2, 24697)
        return np.dot(dout, self.weights[:-1, :].T)


class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx

class SoftmaxCrossEntropy:
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits, y_true):
        self.y_true = y_true
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        batch_size = logits.shape[0]
        loss = -np.log(self.probs[range(batch_size), y_true] + 1e-15)
        return np.mean(loss)

    def backward(self):
        batch_size = self.y_true.shape[0]
        dx = self.probs.copy()
        dx[range(batch_size), self.y_true] -= 1
        return dx / batch_size

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.loss_layer = SoftmaxCrossEntropy()

    def forward(self, x, y_true):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        loss = self.loss_layer.forward(out, y_true)
        return out, loss

    def backward(self):
        dout = self.loss_layer.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def update(self, lr):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.weights -= lr * layer.dweights


class Adam:
    def __init__(self, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0  # Time step counter for bias correction
        
        # Dictionaries to hold momentum and velocity buffers per layer
        self.m = {}
        self.v = {}

    def step(self, layers):
        self.t += 1  # Increment time step
        
        for idx, layer in enumerate(layers):
            # Only optimize layers that track dweights (e.g., DenseLayer)
            if not hasattr(layer, 'dweights') or layer.dweights is None:
                continue
                
            # Initialize buffers on the first optimization step
            if idx not in self.m:
                self.m[idx] = np.zeros_like(layer.weights)
                self.v[idx] = np.zeros_like(layer.weights)
                
            # Get current gradients
            dw = layer.dweights
            
            # 1. Update biased first moment estimate (Momentum)
            self.m[idx] = self.beta_1 * self.m[idx] + (1 - self.beta_1) * dw
            
            # 2. Update biased second raw moment estimate (RMSprop tracking)
            self.v[idx] = self.beta_2 * self.v[idx] + (1 - self.beta_2) * (dw ** 2)
            
            # 3. Compute bias-corrected first moment estimate
            m_corrected = self.m[idx] / (1 - self.beta_1 ** self.t)
            
            # 4. Compute bias-corrected second raw moment estimate
            v_corrected = self.v[idx] / (1 - self.beta_2 ** self.t)
            
            # 5. Apply updates to the layer weights using the bias trick parameters
            layer.weights -= self.lr * m_corrected / (np.sqrt(v_corrected) + self.epsilon)


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

    # ==========================================
    # 2. GENERATING DUMMY TRAINING DATA
    # ==========================================
    N = learning_set.inputs  # number of points per class

    # Setup reproducible random states
    np.random.seed(42)
    
    matrix = np.zeros((2, len(pattern)))
   
    for i in range(1, len(pattern)):
        for j in range(1, 2):
            matrix[j][i] = pattern[i]
            
    X = matrix

    # Binary targets (Two-class classification)
    y = np.array([0])
    
    search_string = ".A1"
    if search_string in filename:
       y = np.array([1])
    search_string = ".A2"
    if search_string in filename:
       y = np.array([0])

    
    # ==========================================
    # 3. CREATING AND TRAINING THE MODEL
    # ==========================================

    # Network Architecture: 
    model = MLP([
                DenseLayer(learning_set.inputs, 2),
                ReLU(),
                DenseLayer(2, output_dim=2)
               ])

    epochs = 200
    learning_rate = 0.1

    print("--- Starting Training Loop ---")
    print("Modern MLP baseline using ReLU")
    print("Pattern being processed: ", filename)

    for epoch in range(epochs):
        # 1. Forward Pass
        predictions, loss = model.forward(X, y)
    
        # 2. Backward Pass
        model.backward()
    
        # 3. Update Weights
        model.update(learning_rate)
    
        # 4. Calculate Training Accuracy
        predicted_classes = np.argmax(predictions, axis=1)
        accuracy = np.mean(predicted_classes == y)
    
        # Print status every 20 epochs
        if epoch % 20 == 0 or epoch == epochs - 1:
           print(f"Epoch {epoch:3d} | Loss: {abs(loss):.4f} | Accuracy: {accuracy * 100:.2f}%")

    
    # Adam Optimizer - Initialization
    print("Adam Optimizer")
    mlp = MLP([
    DenseLayer(input_dim=learning_set.inputs, output_dim=2),
    ReLU(),
    DenseLayer(input_dim=2, output_dim=2)  # Fixed for 2 output classes!
    ])

    optimizer = Adam(learning_rate=0.001)

    num_epochs = 200  # Set your desired total epochs
    # Inside your training loop:
    for epoch in range(num_epochs):
        # 1. Forward Pass
        logits, loss = mlp.forward(X, y)
    
        # 2. Backward Pass
        mlp.backward()
    
        # 3. Optimizer Step (Updates weights inside the layers)
        optimizer.step(mlp.layers)

        # Print status exactly every 20 epochs using your print format
        if epoch % 20 == 0:
            # Calculate running accuracy for the current batch
            predictions = np.argmax(logits, axis=-1)
            accuracy = np.mean(predictions == y)
        
            # Execute your specific metric logging statement
            print(f"Epoch {epoch:3d} | Loss: {abs(loss):.4f} | Accuracy: {accuracy * 100:.2f}%")
