
# Perceptron Implementation from Scratch

def activation_function(value):
    return 1 if value >= 0 else 0


def perceptron(inputs, weights, bias):
    weighted_sum = sum(
        x * w for x, w in zip(inputs, weights)
    ) + bias

    return activation_function(weighted_sum)


# Example: AND gate
weights = [1, 1]
bias = -1.5

inputs_list = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

for inputs in inputs_list:
    prediction = perceptron(inputs, weights, bias)
    print(f"Input: {inputs} -> Output: {prediction}")
