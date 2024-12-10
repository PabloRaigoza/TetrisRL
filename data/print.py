from pprint import pprint
import numpy as np
import sys


# Checking command line arguments
if len(sys.argv) != 3:
    print("Usage: python print.py <path_to_data> <index>")
    sys.exit(1)


# Getting command line arguments
path = sys.argv[1]
index = int(sys.argv[2])


# Printing the data
data = np.load(path, allow_pickle=True)
pprint(data[index])
# print all non-zero entries in data[index]['reward']
# print([x for x in data[index]['reward'] if x != 0])
counter = 0
for index, data in enumerate(data):
    x = data['reward']
    if x != 0:
        counter += 1
        print(index, x)
print(counter)
