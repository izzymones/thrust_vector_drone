import json
from math import sqrt
from time import sleep

def countdown(count):
     for i in range(count, 0, -1):
        print("\rCountdown: {}".format(i), end="")
        sleep(1)

def output_data(data, filename):
    # Serializing json
    json_object = json.dumps(data, indent=4)

    # Writing to sample.json
    with open(filename, "w") as outfile:
        outfile.write(json_object)


def import_data(filename):
    # Open and read the JSON file
    data = {}
    with open(filename, 'r') as file:
        data = json.load(file)
    return data


# compute the distance between two vectors
def distance(v1, v2):
    assert len(v1) == len(v2)
    sum = 0
    for i in range(len(v1)):
        sum += (v1[i] - v2[i])**2
    return sqrt(sum)


# some helper functions we will need
def clip(value, min, max):
       if value < min:
               return min
       if value > max:
               return max
       return value