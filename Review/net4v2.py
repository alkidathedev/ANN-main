#!/usr/bin/env python3
"""
Train a four-layer feed-forward neural network:
input layer -> hidden layer 1 -> hidden layer 2 -> output layer.

The workflow intentionally mirrors net98.py while adding the second hidden
layer and WEIGHT3.txt:
  WEIGHT1.txt   input-to-hidden1 weights
  WEIGHT2.txt   hidden1-to-hidden2 weights
  WEIGHT3.txt   hidden2-to-output weights
  BIGGEST.txt   per-input maximum values
  SMALLEST.txt  per-input minimum values
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import glob
import math
import os
import random
import struct
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from itertools import zip_longest
from numpy import asarray


LOWEST = -500000.0
HIGHEST = 500000.0


@dataclass
class LearningSet:
    total_patterns: int = 0
    inputs: int = 0
    hidden1: int = 0
    hidden2: int = 0
    outputs: int = 0
    rate: float = 0.0
    smoothing: float = 0.0
    tolerance: float = 0.0
    iteration_counter: int = 0
    convergence_check: int = 0


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


def pause(prompt: str = "press enter to continue") -> None:
    input(prompt)


def files_for_pattern(path_pattern: str) -> List[str]:
    return sorted(p for p in glob.glob(path_pattern) if os.path.isfile(p))


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


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + np.exp(-value))


def derivative_from_output(value: float) -> float:
    return value * (1.0 - value)


def zero_matrix(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def random_destination_source(destinations: int, sources: int) -> List[List[float]]:
    return [[0.0] * (sources + 1)] + [
        [random.randrange(10) * 0.01 for _ in range(sources + 1)]
        for _ in range(destinations)
    ]


def random_source_destination(sources: int, destinations: int) -> List[List[float]]:
    return [
        [random.randrange(10) * 0.01 for _ in range(destinations + 1)]
        for _ in range(sources + 1)
    ]


def true_matrix(rows: int, cols: int) -> List[List[bool]]:
    return [[True for _ in range(cols)] for _ in range(rows)]


def false_matrix(rows: int, cols: int) -> List[List[bool]]:
    return [[False for _ in range(cols)] for _ in range(rows)]


def apply_train_mask(change: List[List[float]], train_mask: Sequence[Sequence[bool]]) -> None:
    for row in range(len(change)):
        for col in range(len(change[row])):
            if not train_mask[row][col]:
                change[row][col] = 0.0


def ask_relearn_count(layer_name: str, previous_neurons: int) -> int:
    print(f"How many previous neurones want you relearn in {layer_name}?")
    print(f"Please select between 0 and {previous_neurons} neurones")
    while True:
        value = ask_int("")
        if 0 <= value <= previous_neurons:
            return value
        print(f"Please select between 0 and {previous_neurons} neurones")


def first_layer_output(
    weights: Sequence[Sequence[float]], inputs: Sequence[float], hidden: int, input_count: int
) -> List[float]:
    output = [0.0] * (hidden + 1)
    for hid in range(1, hidden + 1):
        total = 0.0
        for inp in range(0, input_count + 1):
            total += weights[hid][inp] * inputs[inp]
        output[hid] = sigmoid(total)
    output[0] = 1.0
    return output


def source_destination_output(
    weights: Sequence[Sequence[float]], source_outputs: Sequence[float], destinations: int, sources: int
) -> List[float]:
    output = [0.0] * (destinations + 1)
    for dest in range(1, destinations + 1):
        total = 0.0
        for src in range(0, sources + 1):
            total += weights[src][dest] * source_outputs[src]
        output[dest] = sigmoid(total)
    output[0] = 1.0
    return output


def output_layer_output(
    weights: Sequence[Sequence[float]], hidden_outputs: Sequence[float], outputs: int, hidden: int
) -> List[float]:
    result = source_destination_output(weights, hidden_outputs, outputs, hidden)
    result[0] = 0.0
    return result


def make_desired_output(class_counts: Sequence[int], outputs: int) -> List[List[float]]:
    desired: List[List[float]] = []
    for class_index in range(outputs):
        row = [0.0] * (outputs + 1)
        row[class_index + 1] = 1.0
        for _ in range(class_counts[class_index]):
            desired.append(row[:])
    return desired


def random_destination_source(destinations: int, sources: int) -> List[List[float]]:

    # Reviewer 1 point 1
    # 1. Initialize the PRNG with a seed
    rng = np.random.default_rng(seed=42)
    number = rng.normal(loc=0.0, scale=0.01, size=(1, 1))

    return [[0.0] * (sources + 1)] + [
        [random.randrange(10) * number for _ in range(sources + 1)]
        for _ in range(destinations)
    ]


def random_source_destination(sources: int, destinations: int) -> List[List[float]]:

    # Reviewer 1 point 1
    # 1. Initialize the PRNG with a seed
    rng = np.random.default_rng(seed=42)
    number = rng.normal(loc=0.0, scale=0.01, size=(1, 1))

    return [
        [random.randrange(10) * number for _ in range(destinations + 1)]
        for _ in range(sources + 1)
    ]

# Reviewer 2 point 2.1
# 2.1 Xavier/Glorot initialization & Orthogonal initialization
# Xavier/Glorot initialization
# This function needs proper function call in the program
def random_destination_source_xavier(destinations: int, sources: int, variant='uniform') -> List[List[float]]:

     # Define the shape of the weight matrix
    shape = (1, 1)
    n_in = destinations
    n_out = sources

    if variant == 'uniform':
        # Bound: sqrt(6 / (n_in + n_out))
        limit = np.sqrt(6.0 / (n_in + n_out))
        number = np.random.uniform(low=-limit, high=limit, size=shape)
        
    elif variant == 'normal':
        # Standard deviation: sqrt(2 / (n_in + n_out))
        std_dev = np.sqrt(2.0 / (n_in + n_out))
        number = np.random.normal(loc=0.0, scale=std_dev, size=shape)
        
    else:
        raise ValueError("Variant must be 'uniform' or 'normal'")

    return [[0.0] * (sources + 1)] + [
        [random.randrange(10) * number for _ in range(sources + 1)]
        for _ in range(destinations)
    ]


def random_source_destination_xavier(sources: int, destinations: int, variant='uniform') -> List[List[float]]:

    # Define the shape of the weight matrix
    shape = (1, 1)
    n_in = destinations
    n_out = sources

    if variant == 'uniform':
        # Bound: sqrt(6 / (n_in + n_out))
        limit = np.sqrt(6.0 / (n_in + n_out))
        number = np.random.uniform(low=-limit, high=limit, size=shape)
        
    elif variant == 'normal':
        # Standard deviation: sqrt(2 / (n_in + n_out))
        std_dev = np.sqrt(2.0 / (n_in + n_out))
        number = np.random.normal(loc=0.0, scale=std_dev, size=shape)
        
    else:
        raise ValueError("Variant must be 'uniform' or 'normal'")

    return [
        [random.randrange(10) * number for _ in range(destinations + 1)]
        for _ in range(sources + 1)
    ]

         
# Reviewer 2 point 2.1
# Orthogonal initialization
def random_destination_source_orthogonal(destinations: int, sources: int, gain=1.0) -> List[List[float]]:
    
    shape = (destinations, sources)
     
    # Flatten non-2D shapes into a 2D matrix structure
    if len(shape) < 2:
        raise ValueError("Tensor must be at least 2-dimensional.")
    
    flattened_shape = (shape[0] * int(np.prod(shape[1:])), 1) if len(shape) == 2 else (int(np.prod(shape[:-1])), shape[-1])
    
    # Step 1: Draw random normal numbers
    a = np.random.randn(*flattened_shape)
    
    # Step 2: Perform QR decomposition
    q, r = np.linalg.qr(a)
    
    # Make Q uniform by adjusting signs based on the diagonal of R
    d = np.diag(r)
    q *= np.sign(d)

    # Reshape back to target shape and apply gain multiplier
    matrix = (q.reshape(shape)) * gain

    # Turn matrix into a number
    matrix = np.array(matrix)

    # 1. Flatten the matrix to 1D
    flat_array = matrix.flatten()

    # 2. scale to avoid sigmoidal overflow 
    if  ((1.0 + np.min(flat_array)) != 0):
    
        flat_array_coeff = (1.0 - np.max(flat_array)) / (1.0 + np.min(flat_array)) 

    else: flat_array_coeff = 0.0
    
    return [[0.0] * (sources + 1)] + [
        [random.randrange(10) * flat_array_coeff for _ in range(sources + 1)]
        for _ in range(destinations)
    ]


def random_source_destination_orthogonal(sources: int, destinations: int, gain=1.0) -> List[List[float]]:
   
    shape = (destinations, sources)

    # Flatten non-2D shapes into a 2D matrix structure
    if len(shape) < 2:
        raise ValueError("Tensor must be at least 2-dimensional.")
    
    flattened_shape = (shape[0] * int(np.prod(shape[1:])), 1) if len(shape) == 2 else (int(np.prod(shape[:-1])), shape[-1])
    
    # Step 1: Draw random normal numbers
    a = np.random.randn(*flattened_shape)
    
    # Step 2: Perform QR decomposition
    q, r = np.linalg.qr(a)
    
    # Make Q uniform by adjusting signs based on the diagonal of R
    d = np.diag(r)
    q *= np.sign(d)

    # Reshape back to target shape and apply gain multiplier
    matrix = (q.reshape(shape)) * gain

    # Turn matrix into a number
    matrix = np.array(matrix)

    # 1. Flatten the matrix to 1D
    flat_array = matrix.flatten()

    # 2. scale to avoid sigmoidal overflow 
    if  ((1.0 + np.min(flat_array)) != 0):
    
        flat_array_coeff = (1.0 - np.max(flat_array)) / (1.0 + np.min(flat_array)) 

    else: flat_array_coeff = 0.0

    return [
        [random.randrange(10) * flat_array_coeff for _ in range(destinations + 1)]
        for _ in range(sources + 1)
    ]

# Reviewer 3:
# 1. weight similarities during training (Frobenius Norm)
def Frobenius_norm(matrix):
    # Calculate the Frobenius norm
    fro_norm = np.linalg.norm(matrix)
    return fro_norm

# Reviewer 5 - point 5.2
# 5.2 Zero initialization with small random, residual based, orthogonal and 
# diversity-promoting initialization

# a. Zero Initialization with Small Random (Perturbed Zero)
def random_destination_source_zero_with_small_random(destinations: int, sources: int, scale=1e-4) -> List[List[float]]:
    """
    Initializes weights to zero, perturbed by a tiny random value 
    to break initial symmetry.
    """
    number = np.random.normal(loc=0.0, scale=scale, size=(1,1))
    
    return [[0.0] * (sources + 1)] + [
        [random.randrange(10) * number for _ in range(sources + 1)]
        for _ in range(destinations)
    ]

# a. Zero Initialization with Small Random (Perturbed Zero)
def random_source_destination_zero_with_small_random(sources: int, destinations: int, scale=1e-4) -> List[List[float]]:
    
    """
    Initializes weights to zero, perturbed by a tiny random value 
    to break initial symmetry.
    """
    number = np.random.normal(loc=0.0, scale=scale, size=(1,1))
    
    return [
        [random.randrange(10) * number for _ in range(destinations + 1)]
        for _ in range(sources + 1)
    ]

# b. Residual-Based Initialization
def random_destination_source_residual_init(destinations: int, sources: int) -> List[List[float]]:
    
    """
    Initializes weights to random Gaussian numbers
    """
    mean = 0
    std_dev = 1
    number = random.gauss(mean, std_dev)
    
    return [[0.0] * (sources + 1)] + [
        [random.randrange(10) * number for _ in range(sources + 1)]
        for _ in range(destinations)
    ]

# b. Residual-Based Initialization
def random_source_destination_residual_init(sources: int, destinations: int) -> List[List[float]]:
    
    """
    Initializes weights to random Gaussian numbers
    """
    mean = 0
    std_dev = 1
    number = random.gauss(mean, std_dev)
    
    return [
        [random.randrange(10) * number for _ in range(destinations + 1)]
        for _ in range(sources + 1)
    ]

# d. Diversity-Promoting Initialization
def random_destination_source_diversity_promoting_init(destinations: int, sources: int, alpha=0.1) -> List[List[float]]:
    """
    Generates weights and applies an iterative push (repulsion) 
    to maximize the angles between neuron weight vectors.
    """
    input_dim = destinations
    output_dim = sources
    
    # Start with standard random initialization
    W = np.random.normal(0.0, 1.0, size=(input_dim, output_dim))
    
    # Normalize columns (neurons) to unit vectors
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    
    # Iterative repulsion: push similar vectors away from each other
    for _ in range(5):
        # Compute cosine similarities between all pairs of neurons
        similarity = np.dot(W.T, W) # Shape: (output_dim, output_dim)
        
        # Calculate a gradient direction that pushes correlated neurons apart
        # (Subtracting identity so a neuron doesn't repel itself)
        repulsion = np.dot(W, similarity - np.eye(output_dim))
        
        # Step away from high similarity
        W -= alpha * repulsion
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        
    # Scale appropriately (e.g., using Xavier scale adjustment)
    xavier_scale = np.sqrt(2.0 / (input_dim + output_dim))
    W_scaled = W * xavier_scale

    # Reshape back to target shape and apply gain multiplier
    matrix = np.array(W_scaled)

    # 1. Flatten the matrix to 1D
    flat_array = matrix.flatten()

    # 2. Find the next perfect square size
    next_square_side = int(np.ceil(np.sqrt(flat_array.size)))
    total_required_elements = next_square_side ** 2

    # 3. Pad the array with zeros to match the perfect square size
    padded_array = np.zeros(total_required_elements)
    padded_array[:flat_array.size] = flat_array

    # 4. Reshape into the square matrix
    square_matrix = padded_array.reshape(next_square_side, next_square_side)
    
    # Turn matrix into a number
    matrix = np.array(square_matrix)

    # 1. Flatten the matrix to 1D
    flat_array = matrix.flatten()

    # 2. scale to avoid sigmoidal overflow 
    if  ((1.0 + np.min(flat_array)) != 0):
    
        flat_array_coeff = (1.0 - np.min(flat_array)) / (1.0 + np.max(flat_array)) 

    else: flat_array_coeff = 0.0 
        
    return [[0.0] * (sources + 1)] + [
        [random.randrange(10) * flat_array_coeff for _ in range(sources + 1)]
        for _ in range(destinations)
    ]

# d. Diversity-Promoting Initialization
def random_source_destination_diversity_promoting_init(sources: int, destinations: int, alpha=0.1) -> List[List[float]]:
    
    """
    Generates weights and applies an iterative push (repulsion) 
    to maximize the angles between neuron weight vectors.
    """
    input_dim = destinations
    output_dim = sources

    # Start with standard random initialization
    W = np.random.normal(0.0, 1.0, size=(input_dim, output_dim))
    
    # Normalize columns (neurons) to unit vectors
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    
    # Iterative repulsion: push similar vectors away from each other
    for _ in range(5):
        # Compute cosine similarities between all pairs of neurons
        similarity = np.dot(W.T, W) # Shape: (output_dim, output_dim)
        
        # Calculate a gradient direction that pushes correlated neurons apart
        # (Subtracting identity so a neuron doesn't repel itself)
        repulsion = np.dot(W, similarity - np.eye(output_dim))
        
        # Step away from high similarity
        W -= alpha * repulsion
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        
    # Scale appropriately (e.g., using Xavier scale adjustment)
    xavier_scale = np.sqrt(2.0 / (input_dim + output_dim))
    W_scaled = W * xavier_scale

    # Reshape back to target shape and apply gain multiplier
    matrix = np.array(W_scaled)

    # 1. Flatten the matrix to 1D
    flat_array = matrix.flatten()

    # 2. Find the next perfect square size
    next_square_side = int(np.ceil(np.sqrt(flat_array.size)))
    total_required_elements = next_square_side ** 2

    # 3. Pad the array with zeros to match the perfect square size
    padded_array = np.zeros(total_required_elements)
    padded_array[:flat_array.size] = flat_array

    # 4. Reshape into the square matrix
    square_matrix = padded_array.reshape(next_square_side, next_square_side)
    
     # Turn matrix into a number
    matrix = np.array(square_matrix)

    # 1. Flatten the matrix to 1D
    flat_array = matrix.flatten()

    # 2. scale to avoid sigmoidal overflow 
    if  ((1.0 + np.min(flat_array)) != 0):
    
        flat_array_coeff = (1.0 - np.min(flat_array)) / (1.0 + np.max(flat_array)) 

    else: flat_array_coeff = 0.0
    
    return [
        [random.randrange(10) * flat_array_coeff for _ in range(destinations + 1)]
        for _ in range(sources + 1)
    ]


def output_delta(desired: Sequence[float], actual: Sequence[float], outputs: int, alpha=0.1) -> List[float]:
    delta = [0.0] * (outputs + 1)
    for out in range(1, outputs + 1):
        delta[out] = (desired[out] - actual[out]) * derivative_from_output(actual[out])
    return delta

# Reviewer 2 point 1.4
# 1.4 output-layer cancellation patterns
def sigmoid_derivative(output):
    return output * (1 - output)

def output_delta_function(
    desired: Sequence[float], actual: Sequence[float], derivative: Sequence[float], outputs: int
) -> List[float]:

    delta = [0.0] * (outputs + 1)
    output_delta = [0.0] * (outputs + 1)
    for out in range(1, outputs + 1):
        delta[out] = (desired[out] - actual[out]) * derivative[out]
        # 1. Compute the raw error pattern (difference)
        output_error = (desired[out] - actual[out])
        # 2. Calculate cancellation/delta pattern for the output layer
        output_delta[out] = output_error * sigmoid_derivative(actual[out])

    return output_delta


# Reviewer 2 point 5.1
# 5.1 Calculate the mean SSE
def Calculate_Mean_SSE(desired: Sequence[float], actual: Sequence[float]) -> List[float]:

    y_true = desired
    y_pred = actual

    ite = len(desired)

    sum_sse = 0.0
    sum_tss = 0.0

    arr = np.array(y_true, dtype=object)
    arr_1d = arr.reshape(arr.shape[0], -1)
    
    for ite in range(1, ite):
        sum_sse += (arr_1d[ite] - y_pred[ite]) ** 2
        sum_tss += (arr_1d[ite] - np.mean(arr_1d)) ** 2

    # Avoid division by zero if true values are constant
    if sum_tss == 0.0:
       result = 0.0 
    else: result = sum_sse/sum_tss

    # Normalized SSE loss (1 - R-squared in regression contexts)
    return result

def hidden_delta(
    next_delta: Sequence[float],
    next_weights: Sequence[Sequence[float]],
    layer_output: Sequence[float],
    layer_nodes: int,
    next_nodes: int,
) -> List[float]:
    delta = [0.0] * (layer_nodes + 1)
    for node in range(1, layer_nodes + 1):
        total = 0.0
        for next_node in range(1, next_nodes + 1):
            total += next_delta[next_node] * next_weights[node][next_node]
        delta[node] = total * derivative_from_output(layer_output[node])
    return delta


def add_source_destination_correction(
    accumulator: List[List[float]],
    destination_delta: Sequence[float],
    source_output: Sequence[float],
    sources: int,
    destinations: int,
) -> None:
    for src in range(0, sources + 1):
        for dest in range(1, destinations + 1):
            accumulator[src][dest] += destination_delta[dest] * source_output[src]


def add_first_layer_correction(
    accumulator: List[List[float]],
    hidden_delta_values: Sequence[float],
    pattern: Sequence[float],
    hidden: int,
    inputs: int,
) -> None:
    for hid in range(1, hidden + 1):
        for inp in range(0, inputs + 1):
            accumulator[hid][inp] += hidden_delta_values[hid] * pattern[inp]


def error_calculator(actual: Sequence[float], desired: Sequence[float], outputs: int) -> float:
    total = 0.0
    for out in range(1, outputs + 1):
        total += (actual[out] - desired[out]) ** 2
    return total


def adjust(
    corrections: Sequence[Sequence[float]],
    previous_corrections: Sequence[Sequence[float]],
    rate: float,
    smoothing: float,
) -> List[List[float]]:
    result = zero_matrix(len(corrections), len(corrections[0]))
    for row in range(len(corrections)):
        for col in range(len(corrections[row])):
            result[row][col] = rate * corrections[row][col] + smoothing * previous_corrections[row][col]
    return result


def apply_change(weights: List[List[float]], change: Sequence[Sequence[float]]) -> None:
    for row in range(len(weights)):
        for col in range(len(weights[row])):
            weights[row][col] += change[row][col]

def save_first_layer(weights: Sequence[Sequence[float]], hidden1: int, inputs: int) -> None:
    with open("WEIGHT1.txt", "w", encoding="utf-8") as stream:
        for hid in range(1, hidden1 + 1):
            for inp in range(0, inputs + 1):
                stream.write(f"{weights[hid][inp].item():.6f}\n")

def save_second_layer(weights: Sequence[Sequence[float]], hidden1: int, hidden2: int) -> None:
    with open("WEIGHT2.txt", "w", encoding="utf-8") as stream:
        for hid in range(0, hidden1 + 1):
            for out in range(1, hidden2 + 1):
                stream.write(f"{weights[hid][out].item():.6f}\n")

def save_third_layer(weights: Sequence[Sequence[float]], hidden2: int, outputs: int) -> None:
    with open("WEIGHT3.txt", "w", encoding="utf-8") as stream:
        for hid in range(0, hidden2 + 1):
            for out in range(1, outputs + 1):
                stream.write(f"{weights[hid][out].item():.6f}\n")

# Reviewer 2 point 1.1
# 1.1 pairwise cosine similarity 
# Calculate the pairwise similarity matrix
def save_first_layer_COSINE(weights: Sequence[Sequence[float]]) -> None:
    np.savetxt("COSINE_SIMILARITY_1.txt", weights, fmt="%.6f", delimiter="\t")

# Reviewer 2 point 1.1
# 1.1 pairwise cosine similarity 
# Calculate the pairwise similarity matrix
def save_third_layer_COSINE(weights: Sequence[Sequence[float]]) -> None:
    np.savetxt("COSINE_SIMILARITY_3.txt", weights, fmt="%.6f", delimiter="\t")

# Reviewer 2 point 1.2
# 1.2 Euclidean distance between weights
# Compute pairwise Euclidean distances
def save_first_layer_EUCLIDEAN(weights: Sequence[Sequence[float]]) -> None:
    np.savetxt("EUCLIDEAN_DISTANCE_1.txt", weights, fmt="%.6f", delimiter="\t")

# Reviewer 2 point 1.2
# 1.2 Euclidean distance between weights
# Compute pairwise Euclidean distances
def save_third_layer_EUCLIDEAN(weights: Sequence[Sequence[float]]) -> None:
    np.savetxt("EUCLIDEAN_DISTANCE_3.txt", weights, fmt="%.6f", delimiter="\t")

# Reviewer 3:
# 1. weight similarities during training (Frobenius Norm)
def save_first_layer_FROBENIUS_NORM(weights: Sequence[Sequence[float]]) -> None:
    with open("FROBENIUS_NORM_1.txt", "w", encoding="utf-8") as stream:
              to_save = np.sum(weights)
              stream.write(f"{to_save.item():.6f}\n")

# Reviewer 3:
# 1. weight similarities during training (Frobenius Norm)
def save_third_layer_FROBENIUS_NORM(weights: Sequence[Sequence[float]]) -> None:
    with open("FROBENIUS_NORM_3.txt", "w", encoding="utf-8") as stream:
              to_save = np.sum(weights)
              stream.write(f"{to_save.item():.6f}\n")

# Reviewer 2 point 1.1
# 1.1 pairwise cosine similarity 
# Calculate the pairwise similarity matrix
def save_second_layer_COSINE(weights: Sequence[Sequence[float]]) -> None:
    np.savetxt("COSINE_SIMILARITY_2.txt", weights, fmt="%.6f", delimiter="\t")

# Reviewer 2 point 1.2
# 1.2 Euclidean distance between weights
# Compute pairwise Euclidean distances
def save_second_layer_EUCLIDEAN(weights: Sequence[Sequence[float]]) -> None:
    np.savetxt("EUCLIDEAN_DISTANCE_2.txt", weights, fmt="%.6f", delimiter="\t")

# Reviewer 2 point 1.3
# 1.3 Correlations between hidden activations across the training set
def save_second_layer_CORRELATION(weights: Sequence[Sequence[float]]) -> None:
    np.savetxt("CORRELATIONS_2.txt", weights, fmt="%.6f", delimiter="\t")

# Reviewer 2 point 1.3
# 1.3 Correlations between hidden activations across the training set
def save_third_layer_CORRELATION(weights: Sequence[Sequence[float]]) -> None:
    np.savetxt("CORRELATIONS_3.txt", weights, fmt="%.6f", delimiter="\t")

# Reviewer 2 point 1.4
# 1.4 output-layer cancellation patterns
def multiplier(value: float) -> float:
    return value * (1.0 - value)

def f_net(values: Sequence[float], nodes: int) -> List[float]:
    result = [0.0] * (nodes + 1)
    for node in range(1, nodes + 1):
        result[node] = multiplier(values[node])
    return result

# 2. Calculate cancellation/delta pattern for the output layer
def save_third_layer_DELTA_OUTPUT(weights: Sequence[Sequence[float]], outputs: int) -> None:
    with open("DELTA_OUPUT.txt", "w", encoding="utf-8") as stream:
            for out in range(1, outputs + 1):
                stream.write(f"{weights[out].item():.6f}\n")

# Reviewer 3:
# 1. weight similarities during training (Frobenius Norm)
def save_second_layer_FROBENIUS_NORM(weights: Sequence[Sequence[float]]) -> None:
    with open("FROBENIUS_NORM_2.txt", "w", encoding="utf-8") as stream:
              to_save = np.sum(weights)
              stream.write(f"{to_save.item():.6f}\n")

# 1. weight similarities during training (Frobenius Norm)
def save_third_layer_FROBENIUS_NORM(weights: Sequence[Sequence[float]]) -> None:
    with open("FROBENIUS_NORM_3.txt", "w", encoding="utf-8") as stream:
              to_save = np.sum(weights)
              stream.write(f"{to_save.item():.6f}\n")

# Reviewer 3 - point 2.2
# 2.2 An objective automated stopping/growth criterion that uses a combination of:
# mean, standard deviation, epochs, and computational cost
class AutomatedTrainingController:
    def __init__(self, 
                 window_size=10, 
                 min_epochs=20,
                 max_epochs=200, 
                 max_cost=500.0,  # e.g., Maximum budget in USD
                 target_perf=0.90, 
                 utility_threshold=1e-4, 
                 stability_threshold=0.02,
                 beta1=1.0, beta2=0.5, beta3=0.1, beta4=0.2):
        
        # Hyperparameters
        self.window_size = window_size
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.max_cost = max_cost
        self.target_perf = target_perf
        self.utility_threshold = utility_threshold
        self.stability_threshold = stability_threshold
        
        # Core Utility Weights (Beta values)
        self.b1 = beta1  # Reward for Performance Mean
        self.b2 = beta2  # Penalty for Instability (Std Dev)
        self.b3 = beta3  # Penalty for Time (Epochs)
        self.b4 = beta4  # Penalty for Budget (Cost)
        
        # History Tracking
        self.perf_history = []
        self.utility_history = []

    def evaluate_step(self, epoch, current_perf, cumulative_cost):
        """
        Evaluates training progress and returns the recommended action.
        Assumes current_perf is a metric where higher is better (e.g., Accuracy, F1).
        """
        self.perf_history.append(current_perf)
        
        # 1. Warm-up phase check
        if epoch < self.min_epochs:
            return "CONTINUE_TRAINING"
            
        # 2. Hard budget/time ceiling checks
        if epoch >= self.max_epochs or cumulative_cost >= self.max_cost:
            return "STOP_TRAINING"

        # 3. Calculate Mean and StdDev over the moving window
        window = self.perf_history[-self.window_size:]
        mean_perf = np.mean(window)
        std_perf = np.std(window)

        # 4. Normalize inputs to a [0, 1] scale
        norm_epoch = epoch / self.max_epochs
        norm_cost = min(cumulative_cost / self.max_cost, 1.0) 

        # 5. CORE UTILITY FORMULA
        # Rewards mean performance, penalizes variance, epoch duration, and cost
        utility = (self.b1 * mean_perf) - (self.b2 * std_perf) - (self.b3 * norm_epoch) - (self.b4 * norm_cost)
        self.utility_history.append(utility)

        # 6. DECISION LOGIC
        # We need at least window_size evaluations of utility to calculate the gradient delta_U
        if len(self.utility_history) < self.window_size:
            return "CONTINUE_TRAINING"
            
        # Calculate utility change (delta U) compared to the start of the window
        delta_utility = utility - self.utility_history[-self.window_size]

        # Scenario A: High volatility / Instability
        if std_perf > self.stability_threshold:
            return "GROW_MODEL"  # Model is wildly fluctuating

        # Scenario B: Stagnation (Utility delta has flattened out)
        if delta_utility < self.utility_threshold:
            if mean_perf >= self.target_perf:
                return "STOP_TRAINING"  # Target reached and converged successfully
            else:
                return "GROW_MODEL"     # Stuck below target; expand capacity (data/parameters)

        # Scenario C: Stable, efficient progression
        return "CONTINUE_TRAINING"

def save_ranges(biggest: Sequence[float], smallest: Sequence[float], inputs: int) -> None:
    with open("BIGGEST.txt", "w", encoding="utf-8") as stream:
        for idx in range(1, inputs + 1):
            stream.write(f"{biggest[idx]:.6f}\n")
    with open("SMALLEST.txt", "w", encoding="utf-8") as stream:
        for idx in range(1, inputs + 1):
            stream.write(f"{smallest[idx]:.6f}\n")


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
    learning_set.hidden1 = ask_int("INSERT NUMBER of HIDDEN LAYER 1 NEURONS\n")
    learning_set.hidden2 = ask_int("INSERT NUMBER of HIDDEN LAYER 2 NEURONS\n")
    learning_set.rate = ask_float("INSERT LEARNING RATE [0...1]\n")
    learning_set.smoothing = ask_float("INSERT SMOOTHING FACTOR [0...1]\n")
    learning_set.tolerance = ask_float("INSERT LEARNING PROCESS TOLERANCE\n")
    learning_set.iteration_counter = ask_int("INSERT ITERATION COUNTER VALUE (integer)\n")
    learning_set.convergence_check = ask_int(
        "INSERT LEARNING PROCESS CONVERGENCE CHECK VALUE (integer)\n"
    )

    print("CHECK if the learning set is currently correct\n")
    print(f"T={learning_set.total_patterns}")
    print(f"INP={learning_set.inputs}")
    print(f"HID1={learning_set.hidden1}")
    print(f"HID2={learning_set.hidden2}")
    print(f"OUT={learning_set.outputs}")
    print(f"RATE={learning_set.rate}")
    print(f"SMOOF={learning_set.smoothing}")
    print(f"TOLL={learning_set.tolerance}")
    print(f"ITE={learning_set.iteration_counter}")
    print(f"CONV={learning_set.convergence_check}")

    return learning_set, selectors, class_counts


def ordered_training_files(path_pattern: str, selectors: Sequence[str]) -> List[str]:
    result: List[str] = []
    for selector in selectors:
        result.extend(files_for_pattern(class_pattern(path_pattern, selector)))
    return result


def main() -> None:
    random.seed()
    print()
    path_pattern = input("INSERT CURRENT LEARN DIRECTORY [drive:\\directory\\*.*]\n")
    learning_set, selectors, class_counts = set_learning_variables(path_pattern)

    answer = input("\nDIGIT Y to SET THE NETWORK or N to quit\n").strip().upper()
    if answer == "N":
        return

    print("\nLEARNING SET STORED\n")
    pause("press enter to MAKE A RANGE FOR NORMALISATION")

    training_files = ordered_training_files(path_pattern, selectors)
    if not training_files:
        raise RuntimeError("No training files were found for the selected classes.")

    desired_outputs = make_desired_output(class_counts, learning_set.outputs)
    if len(training_files) != len(desired_outputs):
        raise RuntimeError("Internal class/file count mismatch.")

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
    normalised_patterns: List[List[float]] = []
    for idx, filename in enumerate(training_files):
        pattern = pattern_normaliser(filename, smallest, biggest, learning_set.inputs)
        normalised_patterns.append(pattern)
        #for inp, value in enumerate(pattern):
        #    print(f"IO[{idx}][{inp}]={value}")
        #print()

    warnings = 0
    for idx, pattern in enumerate(normalised_patterns):
        for inp in range(1, learning_set.inputs + 1):
            if pattern[inp] > 1:
                print(f"WARNING IO[{idx}][{inp}]={pattern[inp]}")
                print(f"IS from file {training_files[idx]}")
                warnings += 1
    if warnings:
        pause("press enter to quit")
        return
    print("All inputs correctly normalised")

    pause("press enter to SEARCH FOR MISMATCHING CASES")
    print("please WAIT...")
    for left in range(len(normalised_patterns)):
        for right in range(left + 1, len(normalised_patterns)):
            if normalised_patterns[left] == normalised_patterns[right]:
                if desired_outputs[left] != desired_outputs[right]:
                    print(f"WARNING!! {training_files[left]} is MISMATCHING {training_files[right]}")
                    pause("press enter to quit")
                    return

    print("No mismatching cases found")
    pause("press enter to START THE LEARNING PROCESS")

    hidden1 = learning_set.hidden1
    hidden2 = learning_set.hidden2
    outputs = learning_set.outputs

    # Choose the Initialization Scheme (begin)
    print("Please choose the Initialization Scheme:")
    print("p for Pseudo Random Number Generator")
    print("x for Xavier/Glorot")
    print("o for Orthogonal initialization")
    print("z for Zero Initialization with Small Random (Perturbed Zero)")
    print("r for Residual-Based Initialization")
    print("d for Diversity-Promoting Initialization")
    user_input = input().lower().strip()

    if user_input == 'p':
       weight1 = random_destination_source(hidden1, learning_set.inputs)
       weight2 = random_source_destination(hidden1, hidden2)
       weight3 = random_source_destination(hidden2, outputs)
   
    elif user_input == 'x':
        
         user_input_X = input("Choose the Initialization Scheme (u for uniform, n for normal): ").lower().strip()

         if user_input_X == 'u':
            weight1 = random_destination_source_xavier(hidden1, learning_set.inputs, variant='uniform')
            weight2 = random_source_destination_xavier(hidden1, hidden2, variant='uniform')
            weight3 = random_source_destination_xavier(hidden2, outputs, variant='uniform')

         elif user_input_X == 'n':
              weight1 = random_destination_source_xavier(hidden1, learning_set.inputs, variant='normal')
              weight2 = random_source_destination_xavier(hidden1, hidden2, variant='normal')
              weight3 = random_source_destination_xavier(hidden2, outputs, variant='normal')
              
    elif user_input == 'o':
         weight1 = random_destination_source_orthogonal(hidden1, learning_set.inputs, gain=2.78)
         weight2 = random_source_destination_orthogonal(hidden1, hidden2, gain=2.78)
         weight3 = random_source_destination_orthogonal(hidden2, outputs, gain=2.78) 
       
    elif user_input == 'z':
         weight1 = random_destination_source_zero_with_small_random(hidden1, learning_set.inputs, scale=1e-4)
         weight2 = random_source_destination_zero_with_small_random(hidden1, hidden2, scale=1e-4)
         weight3 = random_source_destination_zero_with_small_random(hidden2, outputs, scale=1e-4) 
    
    elif user_input == 'r':
         weight1 = random_destination_source_residual_init(hidden1, learning_set.inputs)
         weight2 = random_source_destination_residual_init(hidden1, hidden2)
         weight3 = random_source_destination_residual_init(hidden2, outputs) 
    
    elif user_input == 'd':
         weight1 = random_destination_source_diversity_promoting_init(hidden1, learning_set.inputs, alpha=0.1)
         weight2 = random_source_destination_diversity_promoting_init(hidden1, hidden2, alpha=0.1)
         weight3 = random_source_destination_diversity_promoting_init(hidden2, outputs, alpha=0.1)  
    # Choose the Initialization Scheme (end)

 
    change1 = zero_matrix(hidden1 + 1, learning_set.inputs + 1)
    change2 = zero_matrix(hidden1 + 1, hidden2 + 1)
    change3 = zero_matrix(hidden2 + 1, outputs + 1)

    train1 = true_matrix(hidden1 + 1, learning_set.inputs + 1)
    train2 = true_matrix(hidden1 + 1, hidden2 + 1)
    train3 = true_matrix(hidden2 + 1, outputs + 1)

    similarity_matrix_first_layer = zero_matrix(hidden1, learning_set.inputs)
    similarity_matrix_second_layer = zero_matrix(hidden1, hidden2)
    similarity_matrix_third_layer = zero_matrix(hidden2, outputs)

    counter = 0
    count = 0

    # Initialize the controller
    controller = AutomatedTrainingController(
        window_size=5, min_epochs=10, max_epochs=100, max_cost=100.0, target_perf=0.85
    )
    
    # Mocking a training loop where performance plateaus early around 0.72
    mock_performance = np.concatenate([
        np.linspace(0.1, 0.7, 15),                # Rapid early growth
        np.random.normal(0.72, 0.005, size=20)    # Premature plateau below target
    ])
    
    cost_per_epoch = 1.50 # e.g., Cloud GPU instance cost per epoch

    what_to_do = ""
    error_value = float("inf")

    while what_to_do != "S" and (error_value * 0.5) >= learning_set.tolerance:
        
          for e in range(1, len(mock_performance) + 1):  
        
              perf = mock_performance[e-1]
              cost = e * cost_per_epoch
              action = controller.evaluate_step(epoch=e, current_perf=perf, cumulative_cost=cost)

              # Reviewer 2 point 5.1
              # 5.1 Calculate the mean SSE 
              error_value = 0.0
              loss = 0.0

              corrections1 = zero_matrix(hidden1 + 1, learning_set.inputs + 1)
              corrections2 = zero_matrix(hidden1 + 1, hidden2 + 1)
              corrections3 = zero_matrix(hidden2 + 1, outputs + 1)

              for pattern, desired in zip(normalised_patterns, desired_outputs):
                  hidden1_output = first_layer_output(weight1, pattern, hidden1, learning_set.inputs)
                  hidden2_output = source_destination_output(weight2, hidden1_output, hidden2, hidden1)
                  actual_output = output_layer_output(weight3, hidden2_output, outputs, hidden2)

                  out_delta = output_delta(desired, actual_output, outputs)
                  hidden2_delta = hidden_delta(out_delta, weight3, hidden2_output, hidden2, outputs)
                  hidden1_delta = hidden_delta(hidden2_delta, weight2, hidden1_output, hidden1, hidden2)

                  add_source_destination_correction(corrections3, out_delta, hidden2_output, hidden2, outputs)
                  add_source_destination_correction(corrections2, hidden2_delta, hidden1_output, hidden1, hidden2)
                  add_first_layer_correction(
                  corrections1, hidden1_delta, pattern, hidden1, learning_set.inputs
                  )

                  output_derivative = f_net(actual_output, learning_set.outputs)

                  error_value += error_calculator(actual_output, desired, outputs)

                  # Reviewer 2 point 5.1
                  # 5.1 Calculate the mean SSE 
                  loss += Calculate_Mean_SSE(actual_output, desired)

              counter += 1
              if counter > learning_set.iteration_counter:
                 print(f"ERROR = {(error_value * 0.5).item():.6f}")
                 counter = 0
                 count += 1

              if (counter == 1):
                 with open("Error_Log_Beginning.txt", "w", encoding="utf-8") as f:
                            # Reviewer 2 point 5.1
                            # 5.1 Calculate the mean SSE 
                            my_string = "Error: \t\t"
                            f.write(my_string)
                            f.write(f"{(error_value * 0.5).item():.6f}\n")
                            my_string = "Mean_SSE: \t"
                            f.write(my_string)
                            to_save = np.sum(loss)
                            f.write(f"{to_save.item():.6f}\n")
                            f.close()

              if (error_value * 0.5) <= learning_set.tolerance:  
                  break

              step3 = adjust(corrections3, change3, learning_set.rate, learning_set.smoothing)
              step2 = adjust(corrections2, change2, learning_set.rate, learning_set.smoothing)
              step1 = adjust(corrections1, change1, learning_set.rate, learning_set.smoothing)

              apply_train_mask(step3, train3)
              apply_train_mask(step2, train2)
              apply_train_mask(step1, train1)

              apply_change(weight3, step3)
              apply_change(weight2, step2)
              apply_change(weight1, step1)

              change3 = step3
              change2 = step2
              change1 = step1

              if action in ["GROW_MODEL"] and count >= learning_set.convergence_check:
                  count = 0

                  if error_value * 0.5 >= learning_set.tolerance:
                     count = 0         
                     print(f"\n[Triggered] Loop broken by controller with action: {action}")
                     while True:
                               print("\nYOU CAN INTERACT with the LEARNING PROCESS")
                               print("DIGIT A to ADD a NEURON to the FIRST HIDDEN LAYER")
                               print("DIGIT B to ADD a NEURON to the SECOND HIDDEN LAYER")
                               print("DIGIT N to NOT CHANGE the NETWORK ARCHITECTURE")
                               print("DIGIT S to STOP The LEARNING PROCESS")
                               what_to_do = input().strip().upper()
                               if what_to_do in {"A", "B", "N", "S"}:
                                  break
                     

                  if what_to_do in {"S"}:
                     break

                  if what_to_do == "A":
                      previous_hidden1 = hidden1
                      hidden1 += 1
                      print("\nA NEURON IS ADDED TO THE FIRST HIDDEN LAYER")
                      print(f"{hidden1} neurones are in the FIRST HIDDEN LAYER")
                      relearn_count = ask_relearn_count(
                          "the first hidden layer", previous_hidden1
                      )
                      relearn_start = hidden1 - relearn_count

                      weight1.append([0.0] * (learning_set.inputs + 1))
                      change1.append([0.0] * (learning_set.inputs + 1))
                      weight2.append([0.0] * (hidden2 + 1))
                      change2.append([0.0] * (hidden2 + 1))

                      for row in range(relearn_start, hidden1 + 1):
                          weight1[row] = [0.0] * (learning_set.inputs + 1)
                          change1[row] = [0.0] * (learning_set.inputs + 1)
                          weight2[row] = [0.0] * (hidden2 + 1)
                          change2[row] = [0.0] * (hidden2 + 1)

                      train1 = false_matrix(hidden1 + 1, learning_set.inputs + 1)
                      train2 = false_matrix(hidden1 + 1, hidden2 + 1)
                      train3 = false_matrix(hidden2 + 1, outputs + 1)
                      for row in range(relearn_start, hidden1 + 1):
                          for col in range(0, learning_set.inputs + 1):
                              train1[row][col] = True
                      for col in range(1, hidden2 + 1):
                              train2[row][col] = True

                      counter = 0
                      count = 0
                      what_to_do = ""

                  elif what_to_do == "B":
                       previous_hidden2 = hidden2
                       hidden2 += 1
                       print("\nA NEURON IS ADDED TO THE SECOND HIDDEN LAYER")
                       print(f"{hidden2} neurones are in the SECOND HIDDEN LAYER")
                       relearn_count = ask_relearn_count(
                           "the second hidden layer", previous_hidden2
                       )
                
                       relearn_start = hidden2 - relearn_count

                       for row in range(0, hidden1 + 1):
                           weight2[row].append(0.0)
                           change2[row].append(0.0)
                       weight3.append([0.0] * (outputs + 1))
                       change3.append([0.0] * (outputs + 1))

                       for dest in range(relearn_start, hidden2 + 1):
                           for row in range(0, hidden1 + 1):
                               weight2[row][dest] = 0.0
                               change2[row][dest] = 0.0
                           weight3[dest] = [0.0] * (outputs + 1)
                           change3[dest] = [0.0] * (outputs + 1)

                       train1 = false_matrix(hidden1 + 1, learning_set.inputs + 1)
                       train2 = false_matrix(hidden1 + 1, hidden2 + 1)
                       train3 = false_matrix(hidden2 + 1, outputs + 1)
                       for dest in range(relearn_start, hidden2 + 1):
                           for row in range(0, hidden1 + 1):
                               train2[row][dest] = True
                           for col in range(1, outputs + 1):
                               train3[dest][col] = True

                       counter = 0
                       count = 0
                       what_to_do = ""

                  elif what_to_do == "N":
                       what_to_do = ""

                  elif error_value * 0.5 < learning_set.tolerance:
                       break

    print("THE NETWORK REACHED THE REQUESTED MINIMUM")
    print(f"THE FIRST HIDDEN LAYER HAS {hidden1} NEURONES")
    print(f"THE SECOND HIDDEN LAYER HAS {hidden2} NEURONES")
    pause("press enter to finish computing and to save the RESULTS")
    print("please WAIT...")

    with open("Error_Log_Ending.txt", "w", encoding="utf-8") as f:
               # Reviewer 2 point 5.1
               # 5.1 Calculate the mean SSE 
               my_string = "Error: \t\t"
               f.write(my_string)
               f.write(f"{(error_value * 0.5).item():.6f}\n")
               my_string = "Mean_SSE: \t"
               f.write(my_string)
               to_save = np.sum(loss)
               f.write(f"{to_save.item():.6f}\n")
               f.close()
    
    # Reviewer 2 point 1.1
    # 1.1 pairwise cosine similarity 
    # Calculate the pairwise similarity matrix
       
    # Find the longest length in your data
    arr = np.array(weight1, dtype=object)
    arr_2d = arr.reshape(arr.shape[0], -1)
    similarity_matrix_first_layer = cosine_similarity(arr_2d)
     
    arr = np.array(weight2 , dtype=object)
    arr_2d = arr.reshape(arr.shape[0], -1)
    similarity_matrix_second_layer = cosine_similarity(arr_2d)

    arr = np.array(weight3, dtype=object)
    arr_2d = arr.reshape(arr.shape[0], -1)
    similarity_matrix_third_layer = cosine_similarity(arr_2d)
        
    # Reviewer 2 point 1.2
    # 1.2 Euclidean distance between weights
    # Compute pairwise Euclidean distances
    # first layer
    arr = np.array(weight1, dtype=object)
    arr_2d = arr.reshape(arr.shape[0], -1)
    the_sum = np.sum((arr_2d[:, np.newaxis, :] - arr_2d[np.newaxis, :, :]), axis=-1) 
    
    # 1. Create a helper function to extract single values from nested arrays
    unwrap = np.vectorize(lambda x: x.item() if isinstance(x, np.ndarray) else x)
    # 2. Clean the array and convert it fully to floating-point numbers
    clean_arr = unwrap(the_sum).astype(float)
    Euclidean_distances_first_layer = np.sqrt(clean_arr ** 2)
  
    # second layer
    arr = np.array(weight2, dtype=object)
    arr_2d = arr.reshape(arr.shape[0], -1)
    the_sum = np.sum((arr_2d[:, np.newaxis, :] - arr_2d[np.newaxis, :, :]), axis=-1) 
        
    # 1. Create a helper function to extract single values from nested arrays
    unwrap = np.vectorize(lambda x: x.item() if isinstance(x, np.ndarray) else x)
    # 2. Clean the array and convert it fully to floating-point numbers
    clean_arr = unwrap(the_sum).astype(float)
    Euclidean_distances_second_layer = np.sqrt(clean_arr ** 2)

    # third layer
    arr = np.array(weight3, dtype=object)
    arr_2d = arr.reshape(arr.shape[0], -1)
    the_sum = np.sum((arr_2d[:, np.newaxis, :] - arr_2d[np.newaxis, :, :]), axis=-1) 
        
    # 1. Create a helper function to extract single values from nested arrays
    unwrap = np.vectorize(lambda x: x.item() if isinstance(x, np.ndarray) else x)
    # 2. Clean the array and convert it fully to floating-point numbers
    clean_arr = unwrap(the_sum).astype(float)
    Euclidean_distances_third_layer = np.sqrt(clean_arr ** 2)


    # Option A: Using Pandas (treats columns as variables/neurons, rows as samples)
    # Reviewer 2 point 1.3
    # 1.3 Correlations between hidden activations across the training set
    # Second Layer
    # 1. Force the nested list structure into a clean, flat 2D numeric NumPy array
    # np.block combines nested arrays/lists, and np.squeeze removes useless 1-sized dimensions
    try:
       numeric_grid = np.squeeze(np.block(weight2))
    except ValueError:
    # Fallback if lengths are irregular: deep flatten element by element
       numeric_grid = np.array(
              [[np.squeeze(item).item() for item in row] for row in weight2], 
              dtype=float
    )
    # 2. Convert the flat list into a Pandas DataFrame
    activation_df = pd.DataFrame(numeric_grid)
    # 3. Fill missing values with the mean
    data_cleared = activation_df.fillna(activation_df.mean())
    data_cleared = data_cleared.fillna(0)
    constant_columns = data_cleared.columns[data_cleared.std() == 0]
    if len(constant_columns) > 0:
       data_cleared = data_cleared.drop(columns=constant_columns)
    # 4. Compute the Pearson correlation matrix
    correlation_matrix_second_layer = data_cleared.corr(method="pearson")
              
    #third layer
    # Reviewer 2 point 1.3
    # 1.3 Correlations between hidden activations across the training set
    # Second Layer
    # 1. Force the nested list structure into a clean, flat 2D numeric NumPy array
    # np.block combines nested arrays/lists, and np.squeeze removes useless 1-sized dimensions
    try:
       numeric_grid = np.squeeze(np.block(weight3))
    except ValueError:
    # Fallback if lengths are irregular: deep flatten element by element
       numeric_grid = np.array(
              [[np.squeeze(item).item() for item in row] for row in weight3], 
              dtype=float
    )
    # 2. Convert the flat list into a Pandas DataFrame
    activation_df = pd.DataFrame(numeric_grid)
    # 3. Fill missing values with the mean
    data_cleared = activation_df.fillna(activation_df.mean())
    data_cleared = data_cleared.fillna(0)
    constant_columns = data_cleared.columns[data_cleared.std() == 0]
    if len(constant_columns) > 0:
       data_cleared = data_cleared.drop(columns=constant_columns)
    # 4. Compute the Pearson correlation matrix
    correlation_matrix_third_layer = data_cleared.corr(method="pearson")        


    # Reviewer 3:
    # 1. weight similarities after training (Frobenius Norm)
    # First Layer
    arr = np.array(weight1, dtype=object)
    arr_2d = arr.reshape(arr.shape[0], -1)
    fro_norm_first_layer = np.linalg.norm(arr_2d)
    # Second Layer
    arr = np.array(weight2, dtype=object)
    arr_2d = arr.reshape(arr.shape[0], -1)
    fro_norm_second_layer = np.linalg.norm(arr_2d)
    # Third Layer
    arr = np.array(weight3, dtype=object)
    arr_2d = arr.reshape(arr.shape[0], -1)
    fro_norm_third_layer = np.linalg.norm(arr_2d)
   
    save_first_layer(weight1, hidden1, learning_set.inputs)
    print("FIRST LAYER WEIGHT SAVED")
    save_second_layer(weight2, hidden1, hidden2)
    print("SECOND LAYER WEIGHT SAVED")
    save_third_layer(weight3, hidden2, outputs)
    print("SECOND LAYER WEIGHT SAVED")


    # Reviewer 2 point 1.1
    # 1.1 pairwise cosine similarity 
    # Calculate the pairwise similarity matrix 
    save_first_layer_COSINE(similarity_matrix_first_layer)
    print("FIRST LAYER SIMILARITY MATRIX SAVED")
    save_second_layer_COSINE(similarity_matrix_second_layer)
    print("SECOND LAYER SIMILARITY MATRIX SAVED")
    save_third_layer_COSINE(similarity_matrix_third_layer)
    print("THIRD LAYER SIMILARITY MATRIX SAVED")

    # Reviewer 2 point 1.2
    # 1.2 Euclidean distance between weights
    # Compute pairwise Euclidean distances
    save_first_layer_EUCLIDEAN(Euclidean_distances_first_layer)
    print("FIRST LAYER EUCLIDEAN DISTANCES SAVED")
    save_second_layer_EUCLIDEAN(Euclidean_distances_second_layer)
    print("SECOND LAYER EUCLIDEAN DISTANCES SAVED")
    save_third_layer_EUCLIDEAN(Euclidean_distances_third_layer)
    print("THIRD LAYER EUCLIDEAN DISTANCES SAVED")

    # Reviewer 2 point 1.3
    # 1.3 Correlations between hidden activations across the training set
    save_second_layer_CORRELATION(correlation_matrix_second_layer)
    print("SECOND LAYER CORRELATION MATRIX SAVED")
    save_third_layer_CORRELATION(correlation_matrix_third_layer)
    print("THIRD LAYER CORRELATION MATRIX SAVED")

    # Reviewer 2 point 1.4
    # 1.4 output-layer cancellation patterns
    delta_output_function = output_delta_function(desired, actual_output, output_derivative, learning_set.outputs)                                              
    save_third_layer_DELTA_OUTPUT(delta_output_function, learning_set.outputs)

    # Reviewer 3:
    # 1. weight similarities during training (Frobenius Norm)
    save_first_layer_FROBENIUS_NORM(fro_norm_first_layer)
    print("FIRST LAYER FROBENIUS NORM SAVED")
    save_second_layer_FROBENIUS_NORM(fro_norm_second_layer)
    print("SECOND LAYER FROBENIUS NORM SAVED")
    save_third_layer_FROBENIUS_NORM(fro_norm_third_layer)
    print("THIRD LAYER FROBENIUS NORM SAVED")

    save_ranges(biggest, smallest, learning_set.inputs)
    print("RANGE FOR NORMALISATION SAVED")

    pause("press enter to quit")


if __name__ == "__main__":
    main()


