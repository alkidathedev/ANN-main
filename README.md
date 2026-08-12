# Data Adaptive Growth Three-Layer Perceptron

Companion software for the paper *"Data Adaptive Growth Three-Layer Perceptron"* (Alkida Isaku, Dr. Carlo Ciulla).

It implements a perceptron whose **hidden layer is grown during training** instead of being fixed in
advance. Training starts with a single hidden neuron; whenever the error stalls above the tolerance, a
new neuron is added under the **`relearn = 0`** schedule: the new neuron starts at zero, the
previously trained neurons are kept, and the network keeps training. The final hidden-layer size is
therefore *discovered from the data*, not chosen beforehand.

The code is pure Python (standard library only), faithfully ported from the original
1990s C++ (`NET98.CPP`, by Dr. Carlo Ciulla), included for reference.

## Contents

| Path | What it is |
|------|------------|
| `net/net-py/net98.py`, `testnet98.py` | **Adaptive 3-layer** trainer + tester (the main programs) |
| `net/net-cpp/NET98.CPP`, `TESTNET98.CPP` | Original 1990s C++ implementation |
| `newnet2024v2/net-py/newnet2024.py` | **Static 3-layer** control (fixed hidden size, no growth) |
| `4-layer/net-py/net4.py`, `testnet4.py` | **4-layer** variant (two hidden layers) |
| `4-layer/math/` | Backpropagation derivations for the 4-layer network |
| `results/CROSS_DATASET_VALIDATION_RESULTS.md` | Cross-dataset validation summary reported in the paper |

## Requirements

- **Python 3.7+** — no third-party packages. Run a trainer with `python3 net98.py`.
- (Optional) a C++ compiler for the historical `.CPP` sources; the Python ports are the recommended way to run.

## How it works

- **Architecture:** `inputs → H hidden (sigmoid) → outputs`. The number of inputs is set by your data
  (the values in one pattern file), and the predicted class is the output neuron with the higher activation.
- **Training:** batch backpropagation with a momentum (smoothing) term, min–max input normalization,
  and adaptive hidden-layer growth. Weights are saved to text files and reloaded for a separate test phase.

---

## Bring your own data

The software ships without datasets (the paper's image sets are available from the author on request),
so you run it on your own patterns. The format is simple:

- **One file per pattern.** Either a text file with one numeric feature per line, or a binary file of
  little-endian doubles (used with the `.A1` / `.A2` extensions).
- **The file extension encodes the class:** e.g. `*.A1` = class 1, `*.A2` = class 2. Any two extensions
  or glob patterns work (`.a`/`.b`, `*.cat`/`*.dog`, …).
- **All files must have the same number of values** — that count becomes the number of input neurons.
- Put training files in one folder and held-out test files in another.

```text
train/                     test/
  sample_001.A1              sample_101.A1
  sample_002.A1              sample_102.A2
  sample_003.A2              ...
  ...
```

## Runtime walkthrough

### 1. Train — `python3 net98.py`

The trainer asks for your training folder and the settings. Lines marked `←` are what you type; the
reported input/pattern counts come from *your* data:

```text
INSERT CURRENT LEARN DIRECTORY [drive:\directory\*.*]
path/to/train/*.*                              ← your training folder
FIRST FILE of the list: sample_001.A1
INPUT NEURONS =<values per file>
LEARNING PATTERNS =<number of files>
INSERT number of classes to discriminate (OUTPUT NEURONS)
2                                              ← number of classes
INSERT file extension of 1 class [letter, .ext, or pattern]
A1                                             ← class-1 files (e.g. *.A1)
INSERT file extension of 2 class [letter, .ext, or pattern]
A2                                             ← class-2 files (e.g. *.A2)
INSERT LEARNING RATE [0...1]
0.005
INSERT SMOOTHING FACTOR [0...1]
0.005
INSERT LEARNING PROCESS TOLERANCE
0.1
INSERT ITERATION COUNTER VALUE (integer)
10
INSERT LEARNING PROCESS CONVERGENCE CHECK VALUE (integer)
100
```

It echoes the configuration; type `Y` to begin:

```text
DIGIT Y to SET THE NETWORK or N to quit
Y
```

> The program then prints every normalized input value (very verbose) and pauses a few times — just
> press **Enter** through the range/normalization/start steps.

Training prints the error periodically. **If a single hidden neuron cannot separate your classes, the
error stalls above the tolerance** (for `N` training patterns it settles near `N × 0.25`) and the program
pauses so you can grow the network:

```text
ERROR=9.498851325355679
ERROR=9.498838326835966
...
YOU CAN INTERACT with the LEARNING PROCESS
DIGIT Y to ADD a NEURON to the NETWORK
DIGIT N to NOT CHANGE the NETWORK ARCHITECTURE
DIGIT S to STOP The LEARNING PROCESS
Y                                              ← add a hidden neuron
A NEURON IS ADDED TO THE HIDDEN LAYER
2 neurones are in the HIDDEN LAYER
How many previous neurones want you relearn?
Please select between 0 and 1 neurones
0                                              ← relearn = 0 (keep the trained neurons)
```

Repeat the grow step as needed. When the error drops below the tolerance the run finishes:

```text
THE NETWORK REACHED THE REQUESTED MINIMUM
THE HIDDEN LAYER HAS 3 NEURONES
press enter to save RESULTS
```

This writes `WEIGHT1.txt`, `WEIGHT2.txt`, `BIGGEST.txt`, `SMALLEST.txt` to the current directory.
**Note the final hidden-neuron count** (here `3`) — you need it for testing.

### 2. Test — `python3 testnet98.py`

Run from the same directory (so it finds the saved weights), and point it at your test folder:

```text
INSERT TEST DIRECTORY [drive:\directory\*.*]
path/to/test/*.*
NUMBER OF PATTERNS TO BE TESTED =<number of files>
INSERT number of classes to discriminate (OUTPUT NEURONS)
2
Please Insert Name of the Class to Identify
class_a                                        ← a label for class 1
Please Insert Name of the Class to Identify
class_b                                        ← a label for class 2
INSERT NUMBER of HIDDEN NEURONS
3                                              ← the count reported by the trainer
DIGIT Y to SET THE NETWORK or N to quit
Y
```

It classifies each pattern and writes `RESPONSE.txt`:

```text
class_a=0.9438585182355781
class_b=0.05618490513440274
SOURCE FILE: path/to/test/sample_101.A1
```

The **predicted class is the output with the higher activation** (here `class_a`).

## Files written

| File | Contents |
|------|----------|
| `WEIGHT1.txt` / `WEIGHT2.txt` | input→hidden and hidden→output weights |
| `BIGGEST.txt` / `SMALLEST.txt` | per-input min/max used for normalization |
| `RESPONSE.txt` | test-phase output activations per pattern |

## Other programs

- **`newnet2024.py`** — static 3-layer control: same workflow, but the hidden size is fixed at the start
  and never grows (no add-a-neuron prompt).
- **`net4.py` / `testnet4.py`** — four-layer variant with a second hidden layer; also writes `WEIGHT3.txt`.

## Citation

If you use this software, please cite the paper *"Data Adaptive Growth Three-Layer Perceptron"* (Alkida Isaku, Dr. Carlo Ciulla).
