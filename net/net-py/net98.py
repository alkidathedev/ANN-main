#!/usr/bin/env python3
"""
Python translation of NET98.CPP.

This program trains an adaptive three-layer perceptron:
input layer -> hidden layer -> output layer.

It intentionally keeps the original file-based workflow:
  WEIGHT1.txt   input-to-hidden weights
  WEIGHT2.txt   hidden-to-output weights
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


def multiplier(value: float) -> float:
    return value * (1.0 - value)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def output_nodes_producer(
    weights: Sequence[Sequence[float]], inputs: Sequence[float], nodes: int, previous: int
) -> List[float]:
    output = [0.0] * (nodes + 1)
    for node in range(1, nodes + 1):
        total = 0.0
        for idx in range(0, previous + 1):
            total += weights[node][idx] * inputs[idx]
        output[node] = sigmoid(total)
    return output


def second_layer_output_producer(
    weights: Sequence[Sequence[float]], hidden_outputs: Sequence[float], outputs: int, hidden: int
) -> List[float]:
    output = [0.0] * (outputs + 1)
    for out in range(1, outputs + 1):
        total = 0.0
        for hid in range(0, hidden + 1):
            total += weights[hid][out] * hidden_outputs[hid]
        output[out] = sigmoid(total)
    return output


def f_net(values: Sequence[float], nodes: int) -> List[float]:
    result = [0.0] * (nodes + 1)
    for node in range(1, nodes + 1):
        result[node] = multiplier(values[node])
    return result


def make_desired_output(class_counts: Sequence[int], outputs: int) -> List[List[float]]:
    desired: List[List[float]] = []
    for class_index in range(outputs):
        row = [0.0] * (outputs + 1)
        row[class_index + 1] = 1.0
        for _ in range(class_counts[class_index]):
            desired.append(row[:])
    return desired


def random_first_layer(inputs: int, hidden: int) -> List[List[float]]:
    return [
        [0.0] * (inputs + 1)
    ] + [[random.randrange(10) * 0.01 for _ in range(inputs + 1)] for _ in range(hidden)]


def random_second_layer(hidden: int, outputs: int) -> List[List[float]]:
    return [
        [random.randrange(10) * 0.01 for _ in range(outputs + 1)]
        for _ in range(hidden + 1)
    ]


def zero_matrix(rows: int, cols: int) -> List[List[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def delta2(
    desired: Sequence[float], actual: Sequence[float], derivative: Sequence[float], outputs: int
) -> List[float]:
    delta = [0.0] * (outputs + 1)
    for out in range(1, outputs + 1):
        delta[out] = (desired[out] - actual[out]) * derivative[out]
    return delta


def delta1(
    output_delta: Sequence[float],
    second_layer: Sequence[Sequence[float]],
    hidden_derivative: Sequence[float],
    hidden: int,
    outputs: int,
) -> List[float]:
    delta = [0.0] * (hidden + 1)
    for hid in range(1, hidden + 1):
        total = 0.0
        for out in range(1, outputs + 1):
            total += output_delta[out] * second_layer[hid][out]
        delta[hid] = total * hidden_derivative[hid]
    return delta


def error_calculator(actual: Sequence[float], desired: Sequence[float], outputs: int) -> float:
    total = 0.0
    for out in range(1, outputs + 1):
        total += (actual[out] - desired[out]) ** 2
    return total


def add_second_layer_correction(
    accumulator: List[List[float]],
    output_delta: Sequence[float],
    hidden_output: Sequence[float],
    hidden: int,
    outputs: int,
) -> None:
    for hid in range(0, hidden + 1):
        for out in range(1, outputs + 1):
            accumulator[hid][out] += output_delta[out] * hidden_output[hid]


def add_first_layer_correction(
    accumulator: List[List[float]],
    hidden_delta: Sequence[float],
    pattern: Sequence[float],
    hidden: int,
    inputs: int,
) -> None:
    for hid in range(1, hidden + 1):
        for inp in range(0, inputs + 1):
            accumulator[hid][inp] += hidden_delta[hid] * pattern[inp]


def adjust_all(
    corrections: Sequence[Sequence[float]],
    previous_change: Sequence[Sequence[float]],
    rate: float,
    smoothing: float,
) -> List[List[float]]:
    rows = len(corrections)
    cols = len(corrections[0])
    result = zero_matrix(rows, cols)
    for row in range(rows):
        for col in range(cols):
            result[row][col] = rate * corrections[row][col] + smoothing * previous_change[row][col]
    return result


def adjust_from(
    corrections: Sequence[Sequence[float]],
    previous_change: Sequence[Sequence[float]],
    rate: float,
    smoothing: float,
    first_trainable_row: int,
) -> List[List[float]]:
    rows = len(corrections)
    cols = len(corrections[0])
    result = zero_matrix(rows, cols)
    for row in range(first_trainable_row, rows):
        for col in range(cols):
            result[row][col] = rate * corrections[row][col] + smoothing * previous_change[row][col]
    return result


def apply_change(weights: List[List[float]], change: Sequence[Sequence[float]]) -> None:
    for row in range(len(weights)):
        for col in range(len(weights[row])):
            weights[row][col] += change[row][col]


def save_first_layer(weights: Sequence[Sequence[float]], hidden: int, inputs: int) -> None:
    with open("WEIGHT1.txt", "w", encoding="utf-8") as stream:
        for hid in range(1, hidden + 1):
            for inp in range(0, inputs + 1):
                stream.write(f"{weights[hid][inp]:.6f}\n")


def save_second_layer(weights: Sequence[Sequence[float]], hidden: int, outputs: int) -> None:
    with open("WEIGHT2.txt", "w", encoding="utf-8") as stream:
        for hid in range(0, hidden + 1):
            for out in range(1, outputs + 1):
                stream.write(f"{weights[hid][out]:.6f}\n")


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
    print("HID=1")
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

    for idx, filename in enumerate(training_files):
        pattern = pattern_handler(filename, learning_set.inputs)
        for inp, value in enumerate(pattern):
            print(f"IO[{idx}][{inp}]={value}")

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
                    for inp in range(learning_set.inputs + 1):
                        print(f"IO={normalised_patterns[left][inp]}   IO={normalised_patterns[right][inp]}")
                    pause("press enter to quit")
                    return

    print("No mismatching cases found")
    pause("press enter to START THE LEARNING PROCESS")

    hidden = 1
    relearn_start = 1
    first_layer = random_first_layer(learning_set.inputs, hidden)
    second_layer = random_second_layer(hidden, learning_set.outputs)
    first_change = zero_matrix(hidden + 1, learning_set.inputs + 1)
    second_change = zero_matrix(hidden + 1, learning_set.outputs + 1)

    counter = 0
    count = 0
    what_to_do = ""
    error_value = float("inf")

    while what_to_do != "S" and (error_value * 0.5) >= learning_set.tolerance:
        error_value = 0.0
        first_corrections = zero_matrix(hidden + 1, learning_set.inputs + 1)
        second_corrections = zero_matrix(hidden + 1, learning_set.outputs + 1)

        for pattern, desired in zip(normalised_patterns, desired_outputs):
            hidden_output = output_nodes_producer(first_layer, pattern, hidden, learning_set.inputs)
            hidden_output[0] = 1.0
            hidden_derivative = f_net(hidden_output, hidden)

            actual_output = second_layer_output_producer(
                second_layer, hidden_output, learning_set.outputs, hidden
            )
            output_derivative = f_net(actual_output, learning_set.outputs)

            out_delta = delta2(
                desired, actual_output, output_derivative, learning_set.outputs
            )
            add_second_layer_correction(
                second_corrections,
                out_delta,
                hidden_output,
                hidden,
                learning_set.outputs,
            )

            hid_delta = delta1(
                out_delta,
                second_layer,
                hidden_derivative,
                hidden,
                learning_set.outputs,
            )
            add_first_layer_correction(
                first_corrections, hid_delta, pattern, hidden, learning_set.inputs
            )

            error_value += error_calculator(actual_output, desired, learning_set.outputs)

        counter += 1
        if counter > learning_set.iteration_counter:
            print(f"ERROR={error_value * 0.5}")
            counter = 0
            count += 1

        if hidden == 1:
            second_step = adjust_all(
                second_corrections, second_change, learning_set.rate, learning_set.smoothing
            )
            first_step = adjust_all(
                first_corrections, first_change, learning_set.rate, learning_set.smoothing
            )
        else:
            second_step = adjust_from(
                second_corrections,
                second_change,
                learning_set.rate,
                learning_set.smoothing,
                relearn_start,
            )
            first_step = adjust_from(
                first_corrections,
                first_change,
                learning_set.rate,
                learning_set.smoothing,
                relearn_start,
            )

        apply_change(second_layer, second_step)
        apply_change(first_layer, first_step)
        second_change = second_corrections
        first_change = first_corrections

        if count >= learning_set.convergence_check:
            count = 0
            max_hidden = int(2 * math.sqrt(learning_set.inputs + learning_set.outputs))
            if hidden < max_hidden:
                while True:
                    print("\nYOU CAN INTERACT with the LEARNING PROCESS")
                    print("DIGIT Y to ADD a NEURON to the NETWORK")
                    print("DIGIT N to NOT CHANGE the NETWORK ARCHITECTURE")
                    print("DIGIT S to STOP The LEARNING PROCESS")
                    what_to_do = input().strip().upper()
                    if what_to_do in {"Y", "N", "S"}:
                        break

                if what_to_do == "Y":
                    hidden += 1
                    print("\nA NEURON IS ADDED TO THE HIDDEN LAYER")
                    print(f"{hidden} neurones are in the HIDDEN LAYER")
                    print("How many previous neurones want you relearn?")
                    print(f"Please select between 0 and {hidden - 1} neurones")
                    previous_to_relearn = ask_int("")
                    if previous_to_relearn < 0 or previous_to_relearn > hidden - 1:
                        print("The number of neurones you have selected to re-learn is not acceptable")
                        pause("Please press enter to Exit")
                        return

                    relearn_start = hidden - previous_to_relearn

                    first_layer.append([0.0] * (learning_set.inputs + 1))
                    first_change.append([0.0] * (learning_set.inputs + 1))
                    first_corrections.append([0.0] * (learning_set.inputs + 1))
                    second_layer.append([0.0] * (learning_set.outputs + 1))
                    second_change.append([0.0] * (learning_set.outputs + 1))
                    second_corrections.append([0.0] * (learning_set.outputs + 1))

                    for row in range(relearn_start, hidden + 1):
                        first_layer[row] = [0.0] * (learning_set.inputs + 1)
                        first_change[row] = [0.0] * (learning_set.inputs + 1)
                        second_layer[row] = [0.0] * (learning_set.outputs + 1)
                        second_change[row] = [0.0] * (learning_set.outputs + 1)

            else:
                while True:
                    print("\nYOU CAN INTERACT with the LEARNING PROCESS")
                    print("DIGIT Y to continue the learning process")
                    print("DIGIT S to STOP The LEARNING PROCESS")
                    what_to_do = input().strip().upper()
                    if what_to_do in {"Y", "S"}:
                        break

    print("THE NETWORK REACHED THE REQUESTED MINIMUM")
    print(f"THE HIDDEN LAYER HAS {hidden} NEURONES")
    pause("press enter to save RESULTS")

    save_first_layer(first_layer, hidden, learning_set.inputs)
    print("FIRST LAYER WEIGHT SAVED")
    save_second_layer(second_layer, hidden, learning_set.outputs)
    print("SECOND LAYER WEIGHT SAVED")
    save_ranges(biggest, smallest, learning_set.inputs)
    print("RANGE FOR NORMALISATION SAVED")

    pause("press enter to quit")


if __name__ == "__main__":
    main()
