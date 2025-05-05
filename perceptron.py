import numpy as np
import os

class Perceptron:

    def __init__(self, bias, weights_np):
        if not isinstance(weights_np, np.ndarray):
             raise TypeError("Weights must be provided as a NumPy array.")
        if weights_np.ndim != 1:
             raise ValueError("Weights array must be one-dimensional.")
        if weights_np.shape[0] == 0:
             print("Warning: Perceptron initialized with no weights.")

        self.bias = bias
        self.weights = weights_np
        self.num_inputs_expected = self.weights.shape[0]


    def calculate_weighted_sum(self, inputs_np):
        if not isinstance(inputs_np, np.ndarray):
            raise TypeError("Inputs must be provided as a NumPy array.")
        if inputs_np.ndim != 1:
            raise ValueError("Inputs array must be one-dimensional.")

        if inputs_np.shape[0] != self.num_inputs_expected:
            raise ValueError(f"Error: Number of inputs ({inputs_np.shape[0]}) does not match "
                             f"the expected number of weights ({self.num_inputs_expected}).")

        z = np.dot(self.weights, inputs_np) + self.bias
        return z

    def predict(self, inputs_np, activation_function):
        if not callable(activation_function):
             raise TypeError("activation_function must be a callable function.")

        z = self.calculate_weighted_sum(inputs_np)
        output = activation_function(z)
        return output

# --- Activation Functions ---

def step_activation(z):
    return 1 if z >= 0 else 0

def sign_activation(z):
    return 1 if z >= 0 else -1

def tanh_activation(z):
    """
    Hyperbolic Tangent (Tanh) activation function.
    Returns a value between -1 and 1.
    """
    return np.tanh(z)

def sigmoid_activation(z):
    """
    Sigmoid / Logistic activation function.
    Returns a value between 0 and 1.
    """
    return 1 / (1 + np.exp(-z))

def relu_activation(z):
    """
    Rectified Linear Unit (ReLU) activation function.
    Returns z if z > 0, otherwise 0.
    """
    return np.maximum(0, z) # Use numpy.maximum for numerical stability/compatibility


# --- Configuration Reading Function ---
def read_configuration(file_path='config.txt'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: Configuration file '{file_path}' not found.")


    try:
        with open(file_path, 'r') as f:
            line = f.readline().strip()
            if not line:
                raise ValueError("Error: Configuration file is empty.")

            values_str = line.split(',')
            if len(values_str) < 2: # Must have at least bias and one weight
                  raise ValueError("Error: Configuration file must contain at least bias and one weight, separated by commas.")

            values_float = [float(val.strip()) for val in values_str]

            bias = values_float[0]
            weights_np = np.array(values_float[1:])

            if weights_np.shape[0] == 0:
                 raise ValueError("Error: No weights found in the configuration file.")

            return bias, weights_np



    except ValueError as e:
        raise ValueError(f"Error processing '{file_path}': Ensure it contains numbers separated by commas. Detail: {e}")
    except Exception as e:
        raise IOError(f"Unexpected error reading configuration file '{file_path}': {e}")
    




