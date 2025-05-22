import struct
import numpy as np

def read_dsp_sift_desc(file_path, print_descriptors=False):
    """
    Reads DSP-SIFT descriptors from a Meshroom .descs binary file and optionally prints their content.

    Args:
        file_path (str): The path to the .descs file.
        print_descriptors (bool, optional): If True, prints the content of each descriptor. Defaults to False.

    Returns:
        numpy.ndarray: A 2D numpy array where each row is a 128-dimensional
                       DSP-SIFT descriptor (as floats), or None if an error occurs.
    """
    try:
        print(f"Attempting to read descriptors from: {file_path}")
        with open(file_path, 'rb') as f:
            num_descriptors_bytes = f.read(8)
            if not num_descriptors_bytes:
                print("Warning: Empty .descs file.")
                return np.array([])

            num_descriptors = struct.unpack('<Q', num_descriptors_bytes)[0]
            print(f"Number of descriptors found in file: {num_descriptors}")
            descriptors = []
            descriptor_size_bytes = 128 * 4
            for i in range(num_descriptors):
                descriptor_bytes = f.read(descriptor_size_bytes)
                if not descriptor_bytes:
                    print(f"Error: Premature end of file while reading descriptor {i+1}. Expected {descriptor_size_bytes} bytes, got 0.")
                    return np.array(descriptors, dtype=np.float32) # Return what was read so far
                if len(descriptor_bytes) != descriptor_size_bytes:
                    raise ValueError(f"Error reading descriptor {i+1}. Expected {descriptor_size_bytes} bytes, got {len(descriptor_bytes)}.")
                descriptor = struct.unpack('<128f', descriptor_bytes)
                descriptors.append(list(descriptor))
                if print_descriptors:
                    print(f"Descriptor {i+1}:")
                    print(descriptor)

            return np.array(descriptors, dtype=np.float32)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except struct.error as e:
        print(f"Error: Could not unpack binary data: {e}")
        return None
    except ValueError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

if __name__ == "__main__":
    desc_file_path = r"C:\Users\zarif\OneDrive\Documents\Photogrammetry\MeshroomCache\SuperPointFeatureExtraction\b2b7edde9e7d1c15f166f7d736793714105673c2\572544492.dspsift.desc"  # Replace with the actual path
    print_all_descriptors = False  # Set to True to print all descriptors

    descriptors = read_dsp_sift_desc(desc_file_path, print_all_descriptors)

    if descriptors is not None:
        print(f"\nSuccessfully read {descriptors.shape[0]} descriptors.")
        if descriptors.shape[0] > 0 and not print_all_descriptors:
            print("First descriptor:")
            print(descriptors[0])
            print("Shape of descriptors array:", descriptors.shape)