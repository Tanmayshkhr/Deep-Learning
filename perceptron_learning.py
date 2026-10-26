"""Train a perceptron classifier from scratch."""


def activation_function(value: float) -> int:
    return 1 if value >= 0 else 0


def predict(inputs: list[float], weights: list[float], bias: float) -> int:
    """Return the binary prediction for one input row."""
    if len(inputs) != len(weights):
        raise ValueError("Each input row must match the number of weights.")

    weighted_sum = sum(value * weight for value, weight in zip(inputs, weights))
    return activation_function(weighted_sum + bias)


def train_perceptron(
    X: list[list[float]],
    y: list[int],
    learning_rate: float = 0.1,
    epochs: int = 10,
) -> tuple[list[float], float]:
    """Train a perceptron and return its learned weights and bias."""
    if not X:
        raise ValueError("Training data must contain at least one input row.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if any(len(inputs) != len(X[0]) for inputs in X):
        raise ValueError("All input rows must contain the same number of features.")
    if any(target not in (0, 1) for target in y):
        raise ValueError("Targets must be binary values: 0 or 1.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    weights = [0.0] * len(X[0])
    bias = 0.0

    for epoch in range(epochs):
        total_errors = 0

        for inputs, target in zip(X, y):
            prediction = predict(inputs, weights, bias)
            error = target - prediction

            for index, value in enumerate(inputs):
                weights[index] += learning_rate * error * value
            bias += learning_rate * error

            if error != 0:
                total_errors += 1

        print(f"Epoch {epoch + 1}: Errors = {total_errors}")
        if total_errors == 0:
            break

    return weights, bias


def main() -> None:
    """Train and evaluate a perceptron on the AND gate."""
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 0, 0, 1]

    weights, bias = train_perceptron(X, y)

    print("\nLearned weights:", weights)
    print("Learned bias:", bias)
    print("\nTesting the perceptron:")

    for inputs in X:
        print(f"Input: {inputs} -> Prediction: {predict(inputs, weights, bias)}")


if __name__ == "__main__":
    main()
