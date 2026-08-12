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
    return 1.0 / (1.0 + math.exp(-value))


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


def output_delta(desired: Sequence[float], actual: Sequence[float], outputs: int) -> List[float]:
    delta = [0.0] * (outputs + 1)
    for out in range(1, outputs + 1):
        delta[out] = (desired[out] - actual[out]) * derivative_from_output(actual[out])
    return delta


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


def save_destination_source(filename: str, weights: Sequence[Sequence[float]], destinations: int, sources: int) -> None:
    with open(filename, "w", encoding="utf-8") as stream:
        for dest in range(1, destinations + 1):
            for src in range(0, sources + 1):
                stream.write(f"{weights[dest][src]:.6f}\n")


def save_source_destination(filename: str, weights: Sequence[Sequence[float]], sources: int, destinations: int) -> None:
    with open(filename, "w", encoding="utf-8") as stream:
        for src in range(0, sources + 1):
            for dest in range(1, destinations + 1):
                stream.write(f"{weights[src][dest]:.6f}\n")


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
        for inp, value in enumerate(pattern):
            print(f"IO[{idx}][{inp}]={value}")
        print()

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

    weight1 = random_destination_source(hidden1, learning_set.inputs)
    weight2 = random_source_destination(hidden1, hidden2)
    weight3 = random_source_destination(hidden2, outputs)

    change1 = zero_matrix(hidden1 + 1, learning_set.inputs + 1)
    change2 = zero_matrix(hidden1 + 1, hidden2 + 1)
    change3 = zero_matrix(hidden2 + 1, outputs + 1)

    train1 = true_matrix(hidden1 + 1, learning_set.inputs + 1)
    train2 = true_matrix(hidden1 + 1, hidden2 + 1)
    train3 = true_matrix(hidden2 + 1, outputs + 1)

    counter = 0
    count = 0
    what_to_do = ""
    error_value = float("inf")

    while what_to_do != "S" and (error_value * 0.5) >= learning_set.tolerance:
        error_value = 0.0
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

            error_value += error_calculator(actual_output, desired, outputs)

        counter += 1
        if counter > learning_set.iteration_counter:
            print(f"ERROR={error_value * 0.5}")
            counter = 0
            count += 1

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

        if count >= learning_set.convergence_check:
            count = 0
            while True:
                print("\nYOU CAN INTERACT with the LEARNING PROCESS")
                print("DIGIT A to ADD a NEURON to the FIRST HIDDEN LAYER")
                print("DIGIT B to ADD a NEURON to the SECOND HIDDEN LAYER")
                print("DIGIT N to NOT CHANGE the NETWORK ARCHITECTURE")
                print("DIGIT S to STOP The LEARNING PROCESS")
                what_to_do = input().strip().upper()
                if what_to_do in {"A", "B", "N", "S"}:
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

    print("THE NETWORK REACHED THE REQUESTED MINIMUM")
    print(f"THE FIRST HIDDEN LAYER HAS {hidden1} NEURONES")
    print(f"THE SECOND HIDDEN LAYER HAS {hidden2} NEURONES")
    pause("press enter to save RESULTS")

    save_destination_source("WEIGHT1.txt", weight1, hidden1, learning_set.inputs)
    print("FIRST LAYER WEIGHT SAVED")
    save_source_destination("WEIGHT2.txt", weight2, hidden1, hidden2)
    print("SECOND LAYER WEIGHT SAVED")
    save_source_destination("WEIGHT3.txt", weight3, hidden2, outputs)
    print("THIRD LAYER WEIGHT SAVED")
    save_ranges(biggest, smallest, learning_set.inputs)
    print("RANGE FOR NORMALISATION SAVED")

    pause("press enter to quit")


if __name__ == "__main__":
    main()
