# Simple-Perceptron
First project for the emergent computing course in Metropolitan University of Caracas.
This project is a simple implementation of a single-layer perceptron with a Graphical User Interface (GUI) built using Python's Tkinter library. It allows users to load perceptron configurations (bias and weights) from a file, select different activation functions, and process input data either manually via the keyboard or from a text file.

## Features

* **Configuration Loading:** Reads perceptron bias and weights from a `config.txt` file.
* **Multiple Activation Functions:** Supports Step, Sign, Tanh (Hyperbolic Tangent), Sigmoid (Logistic), and ReLU (Rectified Linear Unit).
* **Input Methods:** Process single input vectors from manual keyboard entry or multiple input vectors from a text file.
* **Graphical User Interface (GUI):** Intuitive interface using Tkinter.
* **NumPy Integration:** Utilizes NumPy for efficient vector operations.
* **Modular Design:** Code is separated into logical files (`perceptron.py`, `perceptron_gui.py`, `main.py`).

## File Structure

* `main.py`: The main entry point of the application. Handles initial setup and starts the GUI.
* `perceptron.py`: Contains the `Perceptron` class, activation function definitions, and the configuration reading logic.
* `perceptron_gui.py`: Implements the Tkinter GUI class, handling user interactions and displaying results.
* `config.txt`: Text file containing the perceptron's bias and weights (e.g., `-1.5, 1, 1`).
* `ExampleInputs.txt` (Optional): Example text file for processing multiple inputs (one vector per line, comma-separated, e.g., `0,0\n0,1\n1,0\n1,1`).

## How to Use

1.  **Clone or Download:** Get the project files onto your local machine.
2.  **Install Dependencies:** Make sure you have Python installed. Install the required libraries using pip:
    ```bash
    pip install numpy tkinter
    ```
3.  **Setup `config.txt`:** Create or edit the `config.txt` file in the same directory as the Python scripts. The format should be the bias followed by the weights, separated by commas. For example, for an AND gate perceptron:
    ```
    -1.5, 1, 1
    ```
    The number of weights determines the expected number of inputs for the perceptron.
4.  **(Optional) Setup `inputs.txt`:** If you want to test file input, create a text file (e.g., `inputs.txt`) with one input vector per line, values separated by commas. Ensure the number of values per line matches the number of weights in `config.txt`.
5.  **Run the Application:** Open your terminal or command prompt, navigate to the project directory, and run the main script:
    ```bash
    python main.py
    ```
6.  **Interact with the GUI:** Use the graphical interface to load the configuration, select an activation function, and process inputs from the keyboard or a selected file.

## Requirements

* Python 3.x
* `numpy` library
* `tkinter` library (usually included with Python)

---

Feel free to contribute or suggest improvements!
