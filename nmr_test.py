import numpy as np

# Load the .npz file
file_path = 'pointcloud.npz'  # Replace with the path to your .npz file
data = np.load(file_path)

# List all arrays stored in the file
print("Keys in the .npz file:", data.files)

# Access the arrays by their keys
for key in data.files:
    print(f"Array '{key}':\n{data[key].shape}\n")

# Don't forget to close the file if it's not needed anymore
data.close()