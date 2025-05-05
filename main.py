import tkinter as tk
import os
import sys

from perceptron_gui import PerceptronApp

# Create a sample config.txt file if it doesn't exist
def create_sample_config(file_path='config.txt'):
    """Creates a sample config.txt file with AND gate parameters if it doesn't exist."""
    if not os.path.exists(file_path):
        print(f"Creating sample '{file_path}' file...")
        try:
            with open(file_path, 'w') as f_cfg:
                # Write an example configuration
                f_cfg.write("-1.5, 1, 1\n")
            print(f"Sample '{file_path}' file created with AND gate configuration.")
        except IOError as e:
            print(f"Could not create the sample configuration file: {e}")
            sys.exit(1) 

if __name__ == "__main__":
    create_sample_config('config.txt')
    root = tk.Tk()
    app = PerceptronApp(root)
    root.mainloop()
    print("\n--- Program Finished ---")



