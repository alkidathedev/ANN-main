#!/usr/bin/env python3
"""
Python translation of TESTNET98.CPP.

This program loads a trained network produced by net98.py / NET98.CPP and
classifies files from a test directory. It writes responses to RESPONSE.txt.
"""

from __future__ import annotations

import glob
import math
import os
import struct
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class TestSet:
    total_patterns: int = 0
    inputs: int = 0
    hidden: int = 0
    outputs: int = 0


def ask_int(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please insert an integer value.")


def pause(prompt: str = "press enter to continue") -> None:
    input(prompt)


def files_for_pattern(path_pattern: str) -> List[str]:
    return sorted(p for p in glob.glob(path_pattern) if os.path.isfile(p))


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


def normalise(value: float, minimum: float, maximum: float) -> float:
    if maximum == minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def pattern_normaliser(
    filename: str, smallest: Sequence[float], biggest: Sequence[float], inputs: int
) -> List[float]:
    values = read_numbers(filename)
    if len(values) != inputs:
        raise ValueError(
            f"{filename} has {len(values)} values, but the network expects {inputs}."
        )

    pattern = [0.0] * (inputs + 1)
    for idx in range(1, inputs + 1):
        value = normalise(values[idx - 1], smallest[idx], biggest[idx])
        if value < 0.01:
            pattern[idx] = 0.0
        elif value > 0.99:
            pattern[idx] = 1.0
        else:
            pattern[idx] = value
    return pattern


def output_nodes_producer_thresholded(
    weights: Sequence[Sequence[float]], inputs: Sequence[float], nodes: int, previous: int
) -> List[float]:
    output = [0.0] * (nodes + 1)
    for node in range(1, nodes + 1):
        total = 0.0
        for idx in range(1, previous + 1):
            total += weights[node][idx] * inputs[idx]
        threshold = weights[node][0]
        output[node] = sigmoid(total - threshold)
    return output


def second_layer_output_producer_thresholded(
    weights: Sequence[Sequence[float]], hidden_outputs: Sequence[float], outputs: int, hidden: int
) -> List[float]:
    output = [0.0] * (outputs + 1)
    for out in range(1, outputs + 1):
        total = 0.0
        for hid in range(1, hidden + 1):
            total += weights[hid][out] * hidden_outputs[hid]
        threshold = weights[0][out]
        output[out] = sigmoid(total - threshold)
    return output


def load_first_layer(hidden: int, inputs: int) -> List[List[float]]:
    values = read_numbers("WEIGHT1.txt")
    expected = hidden * (inputs + 1)
    if len(values) < expected:
        raise ValueError(f"WEIGHT1.txt has {len(values)} values, expected {expected}.")

    weights = [[0.0] * (inputs + 1) for _ in range(hidden + 1)]
    cursor = 0
    for hid in range(1, hidden + 1):
        for inp in range(0, inputs + 1):
            weights[hid][inp] = values[cursor]
            cursor += 1
    return weights


def load_second_layer(hidden: int, outputs: int) -> List[List[float]]:
    values = read_numbers("WEIGHT2.txt")
    expected = (hidden + 1) * outputs
    if len(values) < expected:
        raise ValueError(f"WEIGHT2.txt has {len(values)} values, expected {expected}.")

    weights = [[0.0] * (outputs + 1) for _ in range(hidden + 1)]
    cursor = 0
    for hid in range(0, hidden + 1):
        for out in range(1, outputs + 1):
            weights[hid][out] = values[cursor]
            cursor += 1
    return weights


def load_range(filename: str, inputs: int) -> List[float]:
    values = read_numbers(filename)
    if len(values) < inputs:
        raise ValueError(f"{filename} has {len(values)} values, expected {inputs}.")
    return [0.0] + values[:inputs]


def set_learning_variables(path_pattern: str) -> tuple[TestSet, List[str], List[str]]:
    test_files = files_for_pattern(path_pattern)
    if not test_files:
        raise FileNotFoundError(f"cannot find FIRST FILE of the list: {path_pattern}")

    print(f"FIRST FILE of the list: {os.path.basename(test_files[0])}")
    inputs = len(read_numbers(test_files[0]))
    print(f"INPUT NEURONS ={inputs}")
    print(f"NUMBER OF PATTERNS TO BE TESTED ={len(test_files)}")

    outputs = ask_int("INSERT number of classes to discriminate (OUTPUT NEURONS)\n")
    class_names: List[str] = []
    for class_index in range(outputs):
        class_name = input("Please Insert Name of the Class to Identify\n")
        class_names.append(class_name)
        print(f"Class to Identify: {class_name}")

    hidden = ask_int("INSERT NUMBER of HIDDEN NEURONS\n")

    test_set = TestSet(
        total_patterns=len(test_files),
        inputs=inputs,
        hidden=hidden,
        outputs=outputs,
    )

    print("CHECK if the TEST set is currently correct\n")
    print(f"TEST PATTERNS={test_set.total_patterns}")
    print(f"INP={test_set.inputs}")
    print(f"HID={test_set.hidden}")
    print(f"OUT={test_set.outputs}\n")

    answer = input("DIGIT Y to SET THE NETWORK or N to quit\n").strip().upper()
    if answer == "N":
        raise SystemExit(0)

    return test_set, class_names, test_files


def main() -> None:
    path_pattern = input("INSERT TEST DIRECTORY [drive:\\directory\\*.*]\n")

    test_set, class_names, test_files = set_learning_variables(path_pattern)
    print("TEST SET STORED")
    print("PLEASE WAIT LOADING...")

    input_layer = load_first_layer(test_set.hidden, test_set.inputs)
    hidden_layer = load_second_layer(test_set.hidden, test_set.outputs)
    biggest = load_range("BIGGEST.txt", test_set.inputs)
    smallest = load_range("SMALLEST.txt", test_set.inputs)

    for hid in range(1, test_set.hidden + 1):
        input_layer[hid][0] = -input_layer[hid][0]

    for out in range(1, test_set.outputs + 1):
        hidden_layer[0][out] = -hidden_layer[0][out]

    pause("READY TO CLASSIFY press enter")

    with open("RESPONSE.txt", "w", encoding="utf-8") as stream:
        stream.write("NETWORK RESPONSE:\n")

        for filename in test_files:
            pattern = pattern_normaliser(filename, smallest, biggest, test_set.inputs)
            hidden_activator = output_nodes_producer_thresholded(
                input_layer, pattern, test_set.hidden, test_set.inputs
            )
            output_activator = second_layer_output_producer_thresholded(
                hidden_layer, hidden_activator, test_set.outputs, test_set.hidden
            )

            stream.write("CHARATERIZED AS:\n")
            for out in range(1, test_set.outputs + 1):
                class_name = class_names[out - 1]
                response = output_activator[out]
                print(f"{class_name}={response}")
                stream.write(f"{class_name} {response}\n")

            print()
            print(f"SOURCE FILE: {filename}")
            stream.write(f"source file: {filename}\n")

            print()
            pause("PRESS ENTER to CLASSIFY a NEW INPUT")

    print("\nCLASSIFICATIONS CORRECTLY SAVED ON FILE")
    pause("hit enter to quit")


if __name__ == "__main__":
    main()
