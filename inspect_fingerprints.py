import numpy as np

def load_fingerprints(file_path):
    try:
        # Load the fingerprints from the .npy file
        fingerprints = np.load(file_path)
        print("Fingerprints loaded successfully.")
        return fingerprints
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def main():
    file_path = "fingerprints/fingerprints_crystals_part2.npy"  # Update the path if needed
    fingerprints = load_fingerprints(file_path)
    
    if fingerprints is not None:
        # Print the fingerprints array and its shape
        print(f"Loaded fingerprints: {fingerprints}")
        print(f"Shape of the fingerprints array: {fingerprints.shape}")

        # If the array is too large, consider printing a subset of it
        # For example, print the first 5 rows
        print(f"First 5 rows of fingerprints:\n{fingerprints[:5]}")

        # Optionally, save the array to a CSV file for easier inspection
        np.savetxt("fingerprints/fingerprints_int.csv", fingerprints, delimiter=",")
        print("Fingerprints saved to fingerprints/fingerprints_int.csv")

if __name__ == "__main__":
    main()
