from pprint import pprint
import numpy as np
import sys


# # Checking command line arguments
# if len(sys.argv) != 3:
#     print("Usage: python print.py <path_to_data> <index>")
#     sys.exit(1)


# # Getting command line arguments
# path = sys.argv[1]
# index = int(sys.argv[2])


# # Printing the data
# data = np.load(path, allow_pickle=True)
# pprint(data[index])

# counter = 0
# for index, data in enumerate(data):
#     x = data['reward']
#     if x != 0:
#         counter += 1
#         print(index, x)
# print(counter)


# Go through BC data folder and load files checking for the ones that have at least 500 length
import os
import numpy as np
import sys

path = "data/BC"
files = os.listdir(path)
counter = 0
for file in files:
    data = np.load(f"{path}/{file}", allow_pickle=True)
    not_empty = [x for x in data if len(x) > 0]
    if len(not_empty) >= 750:
        counter += 1
        print(file, len(data))
print(counter)
