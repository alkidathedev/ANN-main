# Full Four-Layer Backpropagation Derivation

This document expands the four-layer neural network backpropagation in the same style as the three-layer PDF. It uses partial derivatives and the delta rule for each group of weights.

The four-layer network is:

```text
input layer -> hidden layer 1 -> hidden layer 2 -> output layer
```

The example uses two input patterns, as in the PDF:

```text
Pattern 1: i11, i12
Pattern 2: i21, i22
```

The target values follow the PDF style:

```text
(target11, target21) = (1, 0)
(target12, target22) = (0, 1)
```

Therefore:

```text
Pattern 1 target = target11, target12 = 1, 0
Pattern 2 target = target21, target22 = 0, 1
```

## Step 1. Forward Pass

From the input layer to hidden layer 1:

```text
net_h11 = w1 * i11 + w2 * i12 + b1 * 1
out_h11 = 1 / (1 + e^(-net_h11))

net_h12 = w3 * i11 + w4 * i12 + b2 * 1
out_h12 = 1 / (1 + e^(-net_h12))

net_h21 = w1 * i21 + w2 * i22 + b1 * 1
out_h21 = 1 / (1 + e^(-net_h21))

net_h22 = w3 * i21 + w4 * i22 + b2 * 1
out_h22 = 1 / (1 + e^(-net_h22))
```

From hidden layer 1 to hidden layer 2:

```text
net_g11 = w5 * out_h11 + w6 * out_h12 + b3 * 1
out_g11 = 1 / (1 + e^(-net_g11))

net_g12 = w7 * out_h11 + w8 * out_h12 + b4 * 1
out_g12 = 1 / (1 + e^(-net_g12))

net_g21 = w5 * out_h21 + w6 * out_h22 + b3 * 1
out_g21 = 1 / (1 + e^(-net_g21))

net_g22 = w7 * out_h21 + w8 * out_h22 + b4 * 1
out_g22 = 1 / (1 + e^(-net_g22))
```

From hidden layer 2 to the output layer:

```text
net11 = w9 * out_g11 + w10 * out_g12 + b5 * 1
out11 = 1 / (1 + e^(-net11))

net12 = w11 * out_g11 + w12 * out_g12 + b6 * 1
out12 = 1 / (1 + e^(-net12))

net21 = w9 * out_g21 + w10 * out_g22 + b5 * 1
out21 = 1 / (1 + e^(-net21))

net22 = w11 * out_g21 + w12 * out_g22 + b6 * 1
out22 = 1 / (1 + e^(-net22))
```

## Step 2. Calculation of the Error of Fit

```text
Etotal = 1/2 * [(out11 - target11)^2 + (out21 - target21)^2
              + (out12 - target12)^2 + (out22 - target22)^2]
```

With the PDF target values substituted:

```text
Etotal = 1/2 * [(out11 - 1)^2 + (out21 - 0)^2
              + (out12 - 0)^2 + (out22 - 1)^2]
```

## Step 3. Backpropagation From Output Layer to Hidden Layer 2

The weights connecting hidden layer 2 to the output layer are:

```text
w9, w10, b5, w11, w12, b6
```

The delta rule is:

```text
Δw = -η * ∂Etotal / ∂w
```

### Step 3.1 Update of Weighting Connection w9

```text
∂Etotal / ∂w9 =
∂Etotal / ∂out11 * ∂out11 / ∂net11 * ∂net11 / ∂w9
+
∂Etotal / ∂out21 * ∂out21 / ∂net21 * ∂net21 / ∂w9
```

```text
∂Etotal / ∂out11 = out11 - target11
∂Etotal / ∂out21 = out21 - target21
```

```text
∂out11 / ∂net11 = out11 * (1 - out11)
∂out21 / ∂net21 = out21 * (1 - out21)
```

```text
∂net11 / ∂w9 = out_g11
∂net21 / ∂w9 = out_g21
```

Therefore:

```text
∂Etotal / ∂w9 =
(out11 - target11) * out11 * (1 - out11) * out_g11
+
(out21 - target21) * out21 * (1 - out21) * out_g21
```

```text
Δw9 = -η * ∂Etotal / ∂w9
```

### Step 3.2 Update of Weighting Connection w10

```text
∂Etotal / ∂w10 =
(out11 - target11) * out11 * (1 - out11) * out_g12
+
(out21 - target21) * out21 * (1 - out21) * out_g22
```

```text
Δw10 = -η * ∂Etotal / ∂w10
```

### Step 3.3 Update of Bias Weight b5

```text
∂Etotal / ∂b5 =
(out11 - target11) * out11 * (1 - out11) * 1
+
(out21 - target21) * out21 * (1 - out21) * 1
```

```text
Δb5 = -η * ∂Etotal / ∂b5
```

### Step 3.4 Update of Weighting Connection w11

```text
∂Etotal / ∂w11 =
(out12 - target12) * out12 * (1 - out12) * out_g11
+
(out22 - target22) * out22 * (1 - out22) * out_g21
```

```text
Δw11 = -η * ∂Etotal / ∂w11
```

### Step 3.5 Update of Weighting Connection w12

```text
∂Etotal / ∂w12 =
(out12 - target12) * out12 * (1 - out12) * out_g12
+
(out22 - target22) * out22 * (1 - out22) * out_g22
```

```text
Δw12 = -η * ∂Etotal / ∂w12
```

### Step 3.6 Update of Bias Weight b6

```text
∂Etotal / ∂b6 =
(out12 - target12) * out12 * (1 - out12) * 1
+
(out22 - target22) * out22 * (1 - out22) * 1
```

```text
Δb6 = -η * ∂Etotal / ∂b6
```

## Step 4. Backpropagation From Hidden Layer 2 to Hidden Layer 1

The weights connecting hidden layer 1 to hidden layer 2 are:

```text
w5, w6, b3, w7, w8, b4
```

For readability, define the output-layer error components:

```text
O11 = (out11 - target11) * out11 * (1 - out11)
O12 = (out12 - target12) * out12 * (1 - out12)
O21 = (out21 - target21) * out21 * (1 - out21)
O22 = (out22 - target22) * out22 * (1 - out22)
```

Then the hidden layer 2 net-error terms are:

```text
G11 = (O11 * w9  + O12 * w11) * out_g11 * (1 - out_g11)
G12 = (O11 * w10 + O12 * w12) * out_g12 * (1 - out_g12)
G21 = (O21 * w9  + O22 * w11) * out_g21 * (1 - out_g21)
G22 = (O21 * w10 + O22 * w12) * out_g22 * (1 - out_g22)
```

### Step 4.1 Update of Weighting Connection w5

```text
∂Etotal / ∂w5 =
G11 * out_h11 + G21 * out_h21
```

Expanded:

```text
∂Etotal / ∂w5 =
[(O11 * w9 + O12 * w11) * out_g11 * (1 - out_g11) * out_h11]
+
[(O21 * w9 + O22 * w11) * out_g21 * (1 - out_g21) * out_h21]
```

```text
Δw5 = -η * ∂Etotal / ∂w5
```

### Step 4.2 Update of Weighting Connection w6

```text
∂Etotal / ∂w6 =
G11 * out_h12 + G21 * out_h22
```

Expanded:

```text
∂Etotal / ∂w6 =
[(O11 * w9 + O12 * w11) * out_g11 * (1 - out_g11) * out_h12]
+
[(O21 * w9 + O22 * w11) * out_g21 * (1 - out_g21) * out_h22]
```

```text
Δw6 = -η * ∂Etotal / ∂w6
```

### Step 4.3 Update of Bias Weight b3

```text
∂Etotal / ∂b3 =
G11 * 1 + G21 * 1
```

```text
Δb3 = -η * ∂Etotal / ∂b3
```

### Step 4.4 Update of Weighting Connection w7

```text
∂Etotal / ∂w7 =
G12 * out_h11 + G22 * out_h21
```

Expanded:

```text
∂Etotal / ∂w7 =
[(O11 * w10 + O12 * w12) * out_g12 * (1 - out_g12) * out_h11]
+
[(O21 * w10 + O22 * w12) * out_g22 * (1 - out_g22) * out_h21]
```

```text
Δw7 = -η * ∂Etotal / ∂w7
```

### Step 4.5 Update of Weighting Connection w8

```text
∂Etotal / ∂w8 =
G12 * out_h12 + G22 * out_h22
```

Expanded:

```text
∂Etotal / ∂w8 =
[(O11 * w10 + O12 * w12) * out_g12 * (1 - out_g12) * out_h12]
+
[(O21 * w10 + O22 * w12) * out_g22 * (1 - out_g22) * out_h22]
```

```text
Δw8 = -η * ∂Etotal / ∂w8
```

### Step 4.6 Update of Bias Weight b4

```text
∂Etotal / ∂b4 =
G12 * 1 + G22 * 1
```

```text
Δb4 = -η * ∂Etotal / ∂b4
```

## Step 5. Backpropagation From Hidden Layer 1 to Input Layer

The weights connecting the input layer to hidden layer 1 are:

```text
w1, w2, b1, w3, w4, b2
```

The hidden layer 1 net-error terms are:

```text
H11 = (G11 * w5 + G12 * w7) * out_h11 * (1 - out_h11)
H12 = (G11 * w6 + G12 * w8) * out_h12 * (1 - out_h12)
H21 = (G21 * w5 + G22 * w7) * out_h21 * (1 - out_h21)
H22 = (G21 * w6 + G22 * w8) * out_h22 * (1 - out_h22)
```

### Step 5.1 Update of Weighting Connection w1

```text
∂Etotal / ∂w1 =
H11 * i11 + H21 * i21
```

Expanded:

```text
∂Etotal / ∂w1 =
[(G11 * w5 + G12 * w7) * out_h11 * (1 - out_h11) * i11]
+
[(G21 * w5 + G22 * w7) * out_h21 * (1 - out_h21) * i21]
```

```text
Δw1 = -η * ∂Etotal / ∂w1
```

### Step 5.2 Update of Weighting Connection w2

```text
∂Etotal / ∂w2 =
H11 * i12 + H21 * i22
```

Expanded:

```text
∂Etotal / ∂w2 =
[(G11 * w5 + G12 * w7) * out_h11 * (1 - out_h11) * i12]
+
[(G21 * w5 + G22 * w7) * out_h21 * (1 - out_h21) * i22]
```

```text
Δw2 = -η * ∂Etotal / ∂w2
```

### Step 5.3 Update of Bias Weight b1

```text
∂Etotal / ∂b1 =
H11 * 1 + H21 * 1
```

```text
Δb1 = -η * ∂Etotal / ∂b1
```

### Step 5.4 Update of Weighting Connection w3

```text
∂Etotal / ∂w3 =
H12 * i11 + H22 * i21
```

Expanded:

```text
∂Etotal / ∂w3 =
[(G11 * w6 + G12 * w8) * out_h12 * (1 - out_h12) * i11]
+
[(G21 * w6 + G22 * w8) * out_h22 * (1 - out_h22) * i21]
```

```text
Δw3 = -η * ∂Etotal / ∂w3
```

### Step 5.5 Update of Weighting Connection w4

```text
∂Etotal / ∂w4 =
H12 * i12 + H22 * i22
```

Expanded:

```text
∂Etotal / ∂w4 =
[(G11 * w6 + G12 * w8) * out_h12 * (1 - out_h12) * i12]
+
[(G21 * w6 + G22 * w8) * out_h22 * (1 - out_h22) * i22]
```

```text
Δw4 = -η * ∂Etotal / ∂w4
```

### Step 5.6 Update of Bias Weight b2

```text
∂Etotal / ∂b2 =
H12 * 1 + H22 * 1
```

```text
Δb2 = -η * ∂Etotal / ∂b2
```

## Step 6. Weight Update Rule With Smoothing

The PDF-style delta rule is:

```text
Δw = -η * ∂Etotal / ∂w
```

In the Python program, the learning rate is `RATE`, and the smoothing factor is `SMOOF`.

Without smoothing:

```text
new_weight = old_weight + Δw
```

With smoothing:

```text
change = Δw + SMOOF * previous_change
new_weight = old_weight + change
```

## Extension to the Real Application

The example uses two input neurons for clarity. In the real dataset:

```text
input neurons = 24,696
```

Therefore, equations such as:

```text
net_h11 = w1 * i11 + w2 * i12 + b1 * 1
```

become:

```text
net_h1 = w1 * i1 + w2 * i2 + ... + w24696 * i24696 + b1 * 1
```

The proposed four-layer architecture is:

```text
24696 -> 2 -> 2 -> 2
```

