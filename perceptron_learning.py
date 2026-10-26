# Day 2: Perceptron Learning Algorithm

def activation_function(value):
    return 1 if value >= 0 else 0


def train_perceptron(X, y, learning_rate=0.1, epochs=10):
    weights = [0.0, 0.0]
    bias = 0.0

    for epoch in range(epochs):
        total_errors = 0

        for inputs, target in zip(X, y):

            # Step 1: Calculate weighted sum
            weighted_sum = (
                inputs[0] * weights[0]
                + inputs[1] * weights[1]
                + bias
            )

            # Step 2: Make prediction
            prediction = activation_function(weighted_sum)

            # Step 3: Calculate error
            error = target - prediction

            # Step 4: Update weights and bias
            weights[0] += learning_rate * error * inputs[0]
            weights[1] += learning_rate * error * inputs[1]
            bias += learning_rate * error

            if error != 0:
                total_errors += 1

        print(f"Epoch {epoch + 1}: Errors = {total_errors}")

    return weights, bias


# Training data for the AND gate
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y = [0, 0, 0, 1]

# Train the model
weights, bias = train_perceptron(X, y)

print("\nLearned weights:", weights)
print("Learned bias:", bias)


# Test the trained model
print("\nTesting the perceptron:")

for inputs in X:
    weighted_sum = (
        inputs[0] * weights[0]
        + inputs[1] * weights[1]
        + bias
    )

    prediction = activation_function(weighted_sum)

    print(f"Input: {inputs} -> Prediction: {prediction}")