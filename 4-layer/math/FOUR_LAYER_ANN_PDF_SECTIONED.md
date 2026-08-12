# Backpropagation Example of the Four-Layer Artificial Neural Network

The three-layer network is:

```text
input layer -> hidden layer -> output layer
```

The four-layer network is:

```text
input layer -> hidden layer 1 -> hidden layer 2 -> output layer
```

The PDF example uses two input patterns:

```text
Pattern 1: i11, i12
Pattern 2: i21, i22
```

This four-layer example keeps the same two-pattern style. The real application uses 24,696 input values per pattern, but the mathematical process is the same.

![Four-layer ANN diagram](/4-layer/diagrams/four_layer_ann_2x2x2x2.svg)

Diagram file: [four_layer_ann_2x2x2x2.svg](../diagrams/four_layer_ann_2x2x2x2.svg)

## Mapping to the Three-Layer PDF Sections

The PDF sections are handled as follows:

```text
PDF Step 1 Forward Pass
-> Four-layer Step 1 Forward Pass
   The new hidden layer 2 forward equations are inserted between hidden layer 1 and output.

PDF Step 2 Calculation of the Error of Fit
-> Four-layer Step 2 Error Calculation
   The same two-pattern total error is used.

PDF Step 3 Backpropagation from output layer to hidden layer
-> Four-layer Step 3 and Step 4.1
   The output-to-hidden calculation is now output layer to hidden layer 2.

PDF Step 4 Backpropagation from hidden layer to input layer
-> Four-layer Step 3.2, Step 3.3, Step 4.2, and Step 4.3
   The error must pass through hidden layer 2 before reaching hidden layer 1.
```

The important change is that the four-layer network has one extra backpropagation stage:

```text
output -> hidden layer 2 -> hidden layer 1 -> input
```

## Network Structure

Input patterns:

```text
Pattern 1 inputs: i11, i12
Pattern 2 inputs: i21, i22
```

Hidden layer 1 outputs:

```text
Pattern 1: out_h11, out_h12
Pattern 2: out_h21, out_h22
```

Hidden layer 2 outputs:

```text
Pattern 1: out_g11, out_g12
Pattern 2: out_g21, out_g22
```

Output layer outputs:

```text
Pattern 1: out11, out12
Pattern 2: out21, out22
```

Bias inputs are constant:

```text
1
```

Bias weights:

```text
b1, b2 = hidden layer 1 bias weights
b3, b4 = hidden layer 2 bias weights
b5, b6 = output layer bias weights
```

For the circle/square application:

```text
first output neuron  = circle response
second output neuron = square response
```

# Step 1. Forward Pass

The forward pass calculates the outputs from left to right.

The activation formula is:

```text
out = 1 / (1 + e^(-net))
```

## From the Input Layer to Hidden Layer 1

For pattern 1:

```text
net_h11 = w1 * i11 + w2 * i12 + b1 * 1
```

```text
out_h11 = 1 / (1 + e^(-net_h11))
```

```text
net_h12 = w3 * i11 + w4 * i12 + b2 * 1
```

```text
out_h12 = 1 / (1 + e^(-net_h12))
```

For pattern 2:

```text
net_h21 = w1 * i21 + w2 * i22 + b1 * 1
```

```text
out_h21 = 1 / (1 + e^(-net_h21))
```

```text
net_h22 = w3 * i21 + w4 * i22 + b2 * 1
```

```text
out_h22 = 1 / (1 + e^(-net_h22))
```

## From Hidden Layer 1 to Hidden Layer 2

For pattern 1:

```text
net_g11 = w5 * out_h11 + w6 * out_h12 + b3 * 1
```

```text
out_g11 = 1 / (1 + e^(-net_g11))
```

```text
net_g12 = w7 * out_h11 + w8 * out_h12 + b4 * 1
```

```text
out_g12 = 1 / (1 + e^(-net_g12))
```

For pattern 2:

```text
net_g21 = w5 * out_h21 + w6 * out_h22 + b3 * 1
```

```text
out_g21 = 1 / (1 + e^(-net_g21))
```

```text
net_g22 = w7 * out_h21 + w8 * out_h22 + b4 * 1
```

```text
out_g22 = 1 / (1 + e^(-net_g22))
```

## From Hidden Layer 2 to the Output Layer

For pattern 1:

```text
net11 = w9 * out_g11 + w10 * out_g12 + b5 * 1
```

```text
out11 = 1 / (1 + e^(-net11))
```

```text
net12 = w11 * out_g11 + w12 * out_g12 + b6 * 1
```

```text
out12 = 1 / (1 + e^(-net12))
```

For pattern 2:

```text
net21 = w9 * out_g21 + w10 * out_g22 + b5 * 1
```

```text
out21 = 1 / (1 + e^(-net21))
```

```text
net22 = w11 * out_g21 + w12 * out_g22 + b6 * 1
```

```text
out22 = 1 / (1 + e^(-net22))
```

# Step 2. Error Calculation

The PDF writes the error by substituting the expected target values directly into the formula.

In the PDF example, the target patterns are:

```text
(target11, target21) = (1, 0)
(target12, target22) = (0, 1)
```

This means:

```text
Pattern 1 target = (1, 0)
Pattern 2 target = (0, 1)
```

Therefore, the total error is:

```text
Etotal = 1/2 * ((out11 - 1)^2 + (out21 - 0)^2
              + (out12 - 0)^2 + (out22 - 1)^2)
```

More generally, if the desired outputs are:

```text
target11 = desired first output for pattern 1
target21 = desired first output for pattern 2
target12 = desired second output for pattern 1
target22 = desired second output for pattern 2
```

then:

```text
Etotal = 1/2 * ((out11 - target11)^2 + (out21 - target21)^2
              + (out12 - target12)^2 + (out22 - target22)^2)
```

For the circle/square application:

```text
circle target = 1, 0
square target = 0, 1
```

# Step 3. Backward Pass

The backward pass calculates error terms, also called deltas. The error moves from the output layer back through hidden layer 2 and then hidden layer 1.

The derivative term for the activation formula is:

```text
out * (1 - out)
```

## 3.1 Output Layer Error Terms

For pattern 1:

```text
delta11 = (target11 - out11) * out11 * (1 - out11)
```

```text
delta12 = (target12 - out12) * out12 * (1 - out12)
```

For pattern 2:

```text
delta21 = (target21 - out21) * out21 * (1 - out21)
```

```text
delta22 = (target22 - out22) * out22 * (1 - out22)
```

If the target values are already substituted, for example `1, 0`, then:

```text
delta11 = (1 - out11) * out11 * (1 - out11)
```

```text
delta12 = (0 - out12) * out12 * (1 - out12)
```

The PDF writes the partial derivative in the opposite sign:

```text
∂Etotal / ∂w = (out - target) * out * (1 - out) * source
```

Then it applies the delta rule:

```text
Δw = -η * ∂Etotal / ∂w
```

This gives the same update direction as using:

```text
(target - out) * out * (1 - out) * source
```

## 3.2 Hidden Layer 2 Error Terms

Hidden layer 2 receives its error from the output layer.

For pattern 1:

```text
delta_g11 = out_g11 * (1 - out_g11) *
            (delta11 * w9 + delta12 * w11)
```

```text
delta_g12 = out_g12 * (1 - out_g12) *
            (delta11 * w10 + delta12 * w12)
```

For pattern 2:

```text
delta_g21 = out_g21 * (1 - out_g21) *
            (delta21 * w9 + delta22 * w11)
```

```text
delta_g22 = out_g22 * (1 - out_g22) *
            (delta21 * w10 + delta22 * w12)
```

## 3.3 Hidden Layer 1 Error Terms

Hidden layer 1 receives its error from hidden layer 2.

For pattern 1:

```text
delta_h11 = out_h11 * (1 - out_h11) *
            (delta_g11 * w5 + delta_g12 * w7)
```

```text
delta_h12 = out_h12 * (1 - out_h12) *
            (delta_g11 * w6 + delta_g12 * w8)
```

For pattern 2:

```text
delta_h21 = out_h21 * (1 - out_h21) *
            (delta_g21 * w5 + delta_g22 * w7)
```

```text
delta_h22 = out_h22 * (1 - out_h22) *
            (delta_g21 * w6 + delta_g22 * w8)
```

# Step 4. Weight Correction

Each weight correction is calculated as:

```text
correction = delta of destination neuron * output of source neuron
```

Because the example has two patterns, the corrections from both patterns are added together.

In the PDF, each weight is derived using partial derivatives. For example, for the four-layer connection `w9`:

```text
∂Etotal / ∂w9 =
∂Etotal / ∂out11 * ∂out11 / ∂net11 * ∂net11 / ∂w9
+
∂Etotal / ∂out21 * ∂out21 / ∂net21 * ∂net21 / ∂w9
```

Since:

```text
∂Etotal / ∂out11 = out11 - target11
∂out11 / ∂net11 = out11 * (1 - out11)
∂net11 / ∂w9 = out_g11
```

and:

```text
∂Etotal / ∂out21 = out21 - target21
∂out21 / ∂net21 = out21 * (1 - out21)
∂net21 / ∂w9 = out_g21
```

then:

```text
∂Etotal / ∂w9 =
(out11 - target11) * out11 * (1 - out11) * out_g11
+
(out21 - target21) * out21 * (1 - out21) * out_g21
```

The delta rule is:

```text
Δw9 = -η * ∂Etotal / ∂w9
```

The correction equations below use the equivalent update direction:

```text
correction = (target - out) * out * (1 - out) * source
```

## 4.1 Corrections from Hidden Layer 2 to Output Layer

```text
correction_w9  = delta11 * out_g11 + delta21 * out_g21
correction_w10 = delta11 * out_g12 + delta21 * out_g22
```

```text
correction_w11 = delta12 * out_g11 + delta22 * out_g21
correction_w12 = delta12 * out_g12 + delta22 * out_g22
```

```text
correction_b5 = delta11 * 1 + delta21 * 1
correction_b6 = delta12 * 1 + delta22 * 1
```

## 4.2 Corrections from Hidden Layer 1 to Hidden Layer 2

```text
correction_w5 = delta_g11 * out_h11 + delta_g21 * out_h21
correction_w6 = delta_g11 * out_h12 + delta_g21 * out_h22
```

```text
correction_w7 = delta_g12 * out_h11 + delta_g22 * out_h21
correction_w8 = delta_g12 * out_h12 + delta_g22 * out_h22
```

```text
correction_b3 = delta_g11 * 1 + delta_g21 * 1
correction_b4 = delta_g12 * 1 + delta_g22 * 1
```

## 4.3 Corrections from Input Layer to Hidden Layer 1

For the first layer, the chain rule is longer because the error must pass through both hidden layer 2 and hidden layer 1 before reaching the input-to-hidden weights.

For example, for `w1`, the relevant paths for pattern 1 are:

```text
w1 -> net_h11 -> out_h11 -> net_g11 -> out_g11 -> net11 -> out11 -> Etotal
w1 -> net_h11 -> out_h11 -> net_g12 -> out_g12 -> net12 -> out12 -> Etotal
```

For pattern 2, the equivalent paths are:

```text
w1 -> net_h21 -> out_h21 -> net_g21 -> out_g21 -> net21 -> out21 -> Etotal
w1 -> net_h21 -> out_h21 -> net_g22 -> out_g22 -> net22 -> out22 -> Etotal
```

Using the compact delta notation, this becomes:

```text
correction_w1 = delta_h11 * i11 + delta_h21 * i21
```

where `delta_h11` and `delta_h21` already contain all the downstream chain-rule terms from hidden layer 2 and the output layer.

```text
correction_w1 = delta_h11 * i11 + delta_h21 * i21
correction_w2 = delta_h11 * i12 + delta_h21 * i22
```

```text
correction_w3 = delta_h12 * i11 + delta_h22 * i21
correction_w4 = delta_h12 * i12 + delta_h22 * i22
```

```text
correction_b1 = delta_h11 * 1 + delta_h21 * 1
correction_b2 = delta_h12 * 1 + delta_h22 * 1
```

# Step 5. Weight Update

After calculating corrections, the weights are updated.

Without smoothing:

```text
new_weight = old_weight + RATE * correction
```

With smoothing:

```text
change = RATE * correction + SMOOF * previous_change
new_weight = old_weight + change
```

## 5.1 Updating Input to Hidden Layer 1 Weights

```text
w1 = w1 + RATE * correction_w1
w2 = w2 + RATE * correction_w2
w3 = w3 + RATE * correction_w3
w4 = w4 + RATE * correction_w4
```

```text
b1 = b1 + RATE * correction_b1
b2 = b2 + RATE * correction_b2
```

## 5.2 Updating Hidden Layer 1 to Hidden Layer 2 Weights

```text
w5 = w5 + RATE * correction_w5
w6 = w6 + RATE * correction_w6
w7 = w7 + RATE * correction_w7
w8 = w8 + RATE * correction_w8
```

```text
b3 = b3 + RATE * correction_b3
b4 = b4 + RATE * correction_b4
```

## 5.3 Updating Hidden Layer 2 to Output Weights

```text
w9  = w9  + RATE * correction_w9
w10 = w10 + RATE * correction_w10
w11 = w11 + RATE * correction_w11
w12 = w12 + RATE * correction_w12
```

```text
b5 = b5 + RATE * correction_b5
b6 = b6 + RATE * correction_b6
```

# Extension to the Real Application

The PDF uses two inputs only to make the example readable. In the real application, each image pattern has:

```text
24,696 inputs
```

For the first hidden neuron, the equation becomes:

```text
net_h1 = w1 * i1 + w2 * i2 + ... + w24696 * i24696 + b1 * 1
```

The proposed four-layer application architecture is:

```text
24696 -> 2 -> 2 -> 2
```

This means:

```text
24,696 input neurons
2 neurons in hidden layer 1
2 neurons in hidden layer 2
2 output neurons
```

# Files Needed for the Four-Layer Program

The three-layer program saves:

```text
WEIGHT1.txt = input -> hidden
WEIGHT2.txt = hidden -> output
```

The four-layer program should save:

```text
WEIGHT1.txt = input -> hidden layer 1
WEIGHT2.txt = hidden layer 1 -> hidden layer 2
WEIGHT3.txt = hidden layer 2 -> output
```

The normalization files remain:

```text
BIGGEST.txt
SMALLEST.txt
```
