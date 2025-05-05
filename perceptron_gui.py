import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import os

from perceptron import (Perceptron, step_activation, sign_activation, tanh_activation, sigmoid_activation, relu_activation, read_configuration)

class PerceptronApp:

    def __init__(self, root):
        self.root = root
        root.title("Simple Perceptron with Tkinter and NumPy")

        self.perceptron = None
        self.config_file_path = 'config.txt'

        # Update default value and add more options
        self.activation_function_var = tk.StringVar(value="step")

        self.mainframe = ttk.Frame(root, padding="10")
        self.mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        # --- Configuration Section ---
        config_frame = ttk.LabelFrame(self.mainframe, text="Perceptron Configuration", padding="10")
        config_frame.grid(column=0, row=0, sticky=(tk.W, tk.E), pady=5)
        config_frame.columnconfigure(0, weight=1)
        config_frame.columnconfigure(1, weight=3)
        ttk.Label(config_frame, text="Bias (b):").grid(column=0, row=0, sticky=tk.W)
        self.bias_label = ttk.Label(config_frame, text="Not loaded")
        self.bias_label.grid(column=1, row=0, sticky=(tk.W, tk.E))
        ttk.Label(config_frame, text="Weights (w):").grid(column=0, row=1, sticky=tk.W)
        self.weights_label = ttk.Label(config_frame, text="Not loaded", wraplength=400)
        self.weights_label.grid(column=1, row=1, sticky=(tk.W, tk.E))
        ttk.Button(config_frame, text=f"Load Configuration ({self.config_file_path})", command=self.load_config).grid(column=0, row=2, columnspan=2, pady=5)

        # --- Activation Function Section ---
        activation_frame = ttk.LabelFrame(self.mainframe, text="Activation Function", padding="10")
        # Adjust grid layout to accommodate more options
        activation_frame.grid(column=0, row=1, sticky=(tk.W, tk.E), pady=5)
        # Use multiple columns or stack vertically if too many
        activation_frame.columnconfigure(0, weight=1)
        activation_frame.columnconfigure(1, weight=1)
        activation_frame.columnconfigure(2, weight=1)


        ttk.Radiobutton(activation_frame, text="Step (0 or 1)", variable=self.activation_function_var, value="step").grid(column=0, row=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(activation_frame, text="Sign (-1 or 1)", variable=self.activation_function_var, value="sign").grid(column=1, row=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(activation_frame, text="Tanh (-1 to 1)", variable=self.activation_function_var, value="tanh").grid(column=2, row=0, sticky=tk.W, padx=5)
        ttk.Radiobutton(activation_frame, text="Sigmoid (0 to 1)", variable=self.activation_function_var, value="sigmoid").grid(column=0, row=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(activation_frame, text="ReLU (max(0, z))", variable=self.activation_function_var, value="relu").grid(column=1, row=1, sticky=tk.W, padx=5)


        # --- Input Section ---
        input_frame = ttk.LabelFrame(self.mainframe, text="Input", padding="10")
        input_frame.grid(column=0, row=2, sticky=(tk.W, tk.E), pady=5)
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=3)
        input_frame.columnconfigure(2, weight=1)
        ttk.Label(input_frame, text="Inputs (comma-separated):").grid(column=0, row=0, sticky=tk.W)
        self.input_entry = ttk.Entry(input_frame, width=50)
        self.input_entry.grid(column=1, row=0, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(input_frame, text="Process Keyboard Input", command=self.process_keyboard_input).grid(column=2, row=0, sticky=tk.E)
        ttk.Label(input_frame, text="Input File:").grid(column=0, row=1, sticky=tk.W)
        self.file_input_label = ttk.Label(input_frame, text="No file selected", wraplength=300)
        self.file_input_label.grid(column=1, row=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(input_frame, text="Select File", command=self.browse_file).grid(column=2, row=1, sticky=tk.E)
        ttk.Button(input_frame, text="Process File Input", command=self.process_file_input).grid(column=0, row=2, columnspan=3, pady=5)


        # --- Results Section ---
        results_frame = ttk.LabelFrame(self.mainframe, text="Results", padding="10")
        results_frame.grid(column=0, row=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        self.results_text = tk.Text(results_frame, height=10, width=60, state=tk.DISABLED)
        self.results_text.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.grid(column=1, row=0, sticky=(tk.N, tk.S))
        self.results_text['yscrollcommand'] = scrollbar.set

        self.load_config()

    def load_config(self):
        try:
            bias, weights = read_configuration(self.config_file_path)
            self.perceptron = Perceptron(bias, weights)
            self.bias_label.config(text=str(self.perceptron.bias))
            self.weights_label.config(text=str(self.perceptron.weights.tolist()))
            print(f"Configuration loaded. Bias: {self.perceptron.bias}, Weights: {self.perceptron.weights.tolist()}, Expected inputs: {self.perceptron.num_inputs_expected}")
        except (FileNotFoundError, ValueError, IOError, TypeError) as e:
            messagebox.showerror("Configuration Error", f"Could not load configuration: {e}\nMake sure you have a valid '{self.config_file_path}' file.")
            self.perceptron = None
            self.bias_label.config(text="Error loading")
            self.weights_label.config(text="Error loading")

    def get_selected_activation_function(self):

        choice = self.activation_function_var.get()
        if choice == "step":
            return step_activation, "Step"
        elif choice == "sign":
            return sign_activation, "Sign"
        elif choice == "tanh":
            return tanh_activation, "Tanh"
        elif choice == "sigmoid":
            return sigmoid_activation, "Sigmoid"
        elif choice == "relu":
            return relu_activation, "ReLU"
        else:
            return None, "Unknown"

    def clear_results(self):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state=tk.DISABLED)

    def append_result(self, text):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, text + "\n")
        self.results_text.config(state=tk.DISABLED)
        self.results_text.see(tk.END)

    def process_input_data(self, data_list, source_name=""):
        if self.perceptron is None:
            messagebox.showwarning("Warning", "Perceptron configuration is not loaded. Load 'config.txt' first.")
            return

        activation_function, activation_name = self.get_selected_activation_function()
        if activation_function is None: # Handle case where selection is somehow invalid
             self.append_result(f"Error: Invalid activation function selected: {activation_name}.")
             return

        self.append_result(f"--- Processing {source_name} with '{activation_name}' activation ---")

        is_single_input_vector = isinstance(data_list, (list, tuple, np.ndarray)) and (not data_list or not isinstance(data_list[0], (list, tuple, np.ndarray)))

        inputs_to_process = [data_list] if is_single_input_vector and data_list else data_list

        if not inputs_to_process:
            self.append_result("No valid input data provided to process.")
            self.append_result("-" * 30)
            return

        processed_count = 0
        for i, input_vector in enumerate(inputs_to_process):
            line_info = f"Line {i+1}" if source_name.startswith("File") else "Keyboard Input"
            try:
                input_np = np.array(input_vector)

                output = self.perceptron.predict(input_np, activation_function)

                # Format output for display depending on activation function type
                # Step and Sign return int, others might return float
                output_display = f"{output}" if isinstance(output, int) else f"{output:.6f}"


                self.append_result(f"{line_info}: Inputs: {input_np.tolist()}, Output: {output_display}")
                processed_count += 1

            except (ValueError, TypeError) as e:
                self.append_result(f"Error on {line_info}: {e}. Skipping.")
            except Exception as e:
                self.append_result(f"Unexpected error on {line_info}: {e}. Skipping.")

        if processed_count == 0 and inputs_to_process:
             self.append_result("No valid entries processed.")

        self.append_result("-" * 30)

    def process_keyboard_input(self):

        self.clear_results()
        input_str = self.input_entry.get().strip()

        if not input_str:
            self.append_result("Error: Keyboard input field is empty.")
            return

        try:
            values_str = input_str.split(',')
            input_list = [float(val.strip()) for val in values_str]
            self.process_input_data(input_list, source_name="Keyboard")

        except ValueError as e:
             self.append_result(f"Error processing keyboard input: {e}\nMake sure you enter numbers separated by commas.")
        except Exception as e:
             self.append_result(f"An unexpected error occurred while processing keyboard input: {e}")

    def browse_file(self):
        filename = filedialog.askopenfilename(
            initialdir=".",
            title="Select Input File",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if filename:
            self.input_file_path = filename
            self.file_input_label.config(text=os.path.basename(filename))
        else:
            self.input_file_path = ""
            self.file_input_label.config(text="No file selected")

    def process_file_input(self):
        self.clear_results()

        if not hasattr(self, 'input_file_path') or not self.input_file_path or not os.path.exists(self.input_file_path):
            self.append_result("Error: No valid input file has been selected.")
            return

        try:
            inputs_from_file = []
            with open(self.input_file_path, 'r') as f_in:
                for num_line, line in enumerate(f_in, 1):
                    line = line.strip()
                    if not line: continue

                    try:
                        values_str = line.split(',')
                        input_list = [float(val.strip()) for val in values_str]
                        inputs_from_file.append(input_list)
                    except ValueError as e:
                         self.append_result(f"Error reading line {num_line} from file: {e}. This line will be skipped.")
                    except Exception as e:
                         self.append_result(f"Unexpected error reading line {num_line} from file: {e}. This line will be skipped.")

            self.process_input_data(inputs_from_file, source_name=f"File '{os.path.basename(self.input_file_path)}'")

        except FileNotFoundError:
             self.append_result(f"Error: The file '{self.input_file_path}' was not found during processing.")
        except Exception as e:
             self.append_result(f"An unexpected error occurred while reading the file '{self.input_file_path}': {e}")


