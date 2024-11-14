from pprint import pprint
import numpy as np
import sys

# Checking command line arguments
if len(sys.argv) != 3:
    print("Usage: python print_data.py <path_to_data> <index>")
    sys.exit(1)


# Getting command line arguments
path = sys.argv[1]
index = int(sys.argv[2])


# Printing the data
data = np.load(path, allow_pickle=True)
pprint(data[index])
