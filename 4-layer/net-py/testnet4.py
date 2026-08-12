#!/usr/bin/env python3
"""
Classify test patterns using a four-layer network trained by net4.py.

The program loads:
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
import struct
from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class TestSet:
    total_patterns: int = 0
    inputs: int = 0
    hidden1: int = 0
    hidden2: int = 0
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


def load_destination_source(filename: str, destinations: int, sources: int) -> List[List[float]]:
    values = read_numbers(filename)
    expected = destinations * (sources + 1)
    if len(values) < expected:
        raise ValueError(f"{filename} has {len(values)} values, expected {expected}.")

    weights = [[0.0] * (sources + 1) for _ in range(destinations + 1)]
    cursor = 0
    for dest in range(1, destinations + 1):
        for src in range(0, sources + 1):
            weights[dest][src] = values[cursor]
            cursor += 1
    return weights


def load_source_destination(filename: str, sources: int, destinations: int) -> List[List[float]]:
    values = read_numbers(filename)
    expected = (sources + 1) * destinations
    if len(values) < expected:
        raise ValueError(f"{filename} has {len(values)} values, expected {expected}.")

    weights = [[0.0] * (destinations + 1) for _ in range(sources + 1)]
    cursor = 0
    for src in range(0, sources + 1):
        for dest in range(1, destinations + 1):
            weights[src][dest] = values[cursor]
            cursor += 1
    return weights


def load_range(filename: str, inputs: int) -> List[float]:
    values = read_numbers(filename)
    if len(values) < inputs:
        raise ValueError(f"{filename} has {len(values)} values, expected {inputs}.")
    return [0.0] + values[:inputs]


def first_layer_output_thresholded(
    weights: Sequence[Sequence[float]], inputs: Sequence[float], hidden: int, input_count: int
) -> List[float]:
    output = [0.0] * (hidden + 1)
    for hid in range(1, hidden + 1):
        total = 0.0
        for inp in range(1, input_count + 1):
            total += weights[hid][inp] * inputs[inp]
        threshold = weights[hid][0]
        output[hid] = sigmoid(total - threshold)
    output[0] = 1.0
    return output


def source_destination_output_thresholded(
    weights: Sequence[Sequence[float]], source_outputs: Sequence[float], destinations: int, sources: int
) -> List[float]:
    output = [0.0] * (destinations + 1)
    for dest in range(1, destinations + 1):
        total = 0.0
        for src in range(1, sources + 1):
            total += weights[src][dest] * source_outputs[src]
        threshold = weights[0][dest]
        output[dest] = sigmoid(total - threshold)
    output[0] = 1.0
    return output


def output_layer_output_thresholded(
    weights: Sequence[Sequence[float]], hidden_outputs: Sequence[float], outputs: int, hidden: int
) -> List[float]:
    result = source_destination_output_thresholded(weights, hidden_outputs, outputs, hidden)
    result[0] = 0.0
    return result


def set_testing_variables(path_pattern: str) -> tuple[TestSet, List[str], List[str]]:
    test_files = files_for_pattern(path_pattern)
    if not test_files:
        raise FileNotFoundError(f"cannot find FIRST FILE of the list: {path_pattern}")

    print(f"FIRST FILE of the list: {os.path.basename(test_files[0])}")
    inputs = len(read_numbers(test_files[0]))
    print(f"INPUT NEURONS ={inputs}")
    print(f"NUMBER OF PATTERNS TO BE TESTED ={len(test_files)}")

    outputs = ask_int("INSERT number of classes to discriminate (OUTPUT NEURONS)\n")
    class_names: List[str] = []
    for _ in range(outputs):
        class_name = input("Please Insert Name of the Class to Identify\n")
        class_names.append(class_name)
        print(f"Class to Identify: {class_name}")

    hidden1 = ask_int("INSERT NUMBER of HIDDEN LAYER 1 NEURONS\n")
    hidden2 = ask_int("INSERT NUMBER of HIDDEN LAYER 2 NEURONS\n")

    test_set = TestSet(
        total_patterns=len(test_files),
        inputs=inputs,
        hidden1=hidden1,
        hidden2=hidden2,
        outputs=outputs,
    )

    print("CHECK if the TEST set is currently correct\n")
    print(f"TEST PATTERNS={test_set.total_patterns}")
    print(f"INP={test_set.inputs}")
    print(f"HID1={test_set.hidden1}")
    print(f"HID2={test_set.hidden2}")
    print(f"OUT={test_set.outputs}\n")

    answer = input("DIGIT Y to SET THE NETWORK or N to quit\n").strip().upper()
    if answer == "N":
        raise SystemExit(0)

    return test_set, class_names, test_files


def main() -> None:
    path_pattern = input("INSERT TEST DIRECTORY [drive:\\directory\\*.*]\n")

    test_set, class_names, test_files = set_testing_variables(path_pattern)
    print("TEST SET STORED")
    print("PLEASE WAIT LOADING...")

    weight1 = load_destination_source("WEIGHT1.txt", test_set.hidden1, test_set.inputs)
    weight2 = load_source_destination("WEIGHT2.txt", test_set.hidden1, test_set.hidden2)
    weight3 = load_source_destination("WEIGHT3.txt", test_set.hidden2, test_set.outputs)
    biggest = load_range("BIGGEST.txt", test_set.inputs)
    smallest = load_range("SMALLEST.txt", test_set.inputs)

    for hid in range(1, test_set.hidden1 + 1):
        weight1[hid][0] = -weight1[hid][0]

    for hid in range(1, test_set.hidden2 + 1):
        weight2[0][hid] = -weight2[0][hid]

    for out in range(1, test_set.outputs + 1):
        weight3[0][out] = -weight3[0][out]

    pause("READY TO CLASSIFY press enter")

    with open("RESPONSE.txt", "w", encoding="utf-8") as stream:
        stream.write("NETWORK RESPONSE:\n")

        for filename in test_files:
            pattern = pattern_normaliser(filename, smallest, biggest, test_set.inputs)
            hidden1_output = first_layer_output_thresholded(
                weight1, pattern, test_set.hidden1, test_set.inputs
            )
            hidden2_output = source_destination_output_thresholded(
                weight2, hidden1_output, test_set.hidden2, test_set.hidden1
            )
            output_activator = output_layer_output_thresholded(
                weight3, hidden2_output, test_set.outputs, test_set.hidden2
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
