# Python Translation Usage

This repository now contains Python translations of the original C++ programs:

- `net98.py` translates `NET98.CPP` and trains the adaptive perceptron.
- `testnet98.py` translates `TESTNET98.CPP` and classifies test patterns using saved weights.

The translated programs preserve the original workflow:

1. Train with `net98.py`.
2. Save `WEIGHT1.txt`, `WEIGHT2.txt`, `BIGGEST.txt`, and `SMALLEST.txt`.
3. Test/classify with `testnet98.py`.
4. Read classification results from `RESPONSE.txt`.

## Training Data Format

Each input pattern is a text file containing one numeric feature per line.

Example:

```text
0.42
1.75
3.12
0.08
```

All training and test files must contain the same number of numeric values.

The number of values in the first file becomes the number of input neurons.

## Class Selection

The original C++ program groups training files by extension. The Python version supports the same idea.

Example training files:

```text
sample_001.a
sample_002.a
sample_003.b
sample_004.b
```

If you run the trainer with:

```text
data/train/*.*
```

Then enter:

```text
a
b
```

as the two class extensions.

You may also enter `.a`, `.b`, or wildcard patterns like `*.a`.

## Run Training

```bash
python3 net98.py
```

The trainer asks for:

- training directory pattern
- number of output classes
- file extension or pattern for each class
- learning rate
- smoothing factor
- error tolerance
- iteration counter
- convergence-check interval

The network starts with one hidden neuron. During learning, the program periodically asks whether to add a hidden neuron.

At the end, it writes:

```text
WEIGHT1.txt
WEIGHT2.txt
BIGGEST.txt
SMALLEST.txt
```

Record the final hidden-neuron count printed by the program. You need it for testing.

## Run Testing

```bash
python3 testnet98.py
```

The tester asks for:

- test directory pattern
- number of output classes
- class names
- final number of hidden neurons

It writes the output activations to:

```text
RESPONSE.txt
```

The predicted class is normally the class with the largest output value.

