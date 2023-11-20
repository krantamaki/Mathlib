"""
COLLECTION OF HELPER FUNCTIONS
"""
import os
import numpy as np


# Function for converting a value to valid type
def convert(val, conversion_type):
  try:
    ret = conversion_type(val)
  except ValueError:
    return None
  
  return ret


# Function that asks for user input until a valid choice is given
def get_input(question, conversion_type, choices=[]):

  print(question)
  val = convert(input(), conversion_type)

  while (val is None) or ((len(choices) != 0) and (val not in choices)):
    print(f"Improper value {val} passed. Please try again")
    print(question)
    val = convert(input(), conversion_type)

  print()

  return val


# Function that forms the config file passed to solvers
# Takes a list of tuples of form (key, type, val) as parameter
def form_config(tups, filename):
  with open(filename, "w") as f:
    for tup in tups:
      f.write(f"{tup[0]} = {tup[1]} {tup[2]}\n")


# Function that finds files with certain substring in them
def find_files(substr, path):

  all_files = os.listdir(path)
  matching_files = [f for f in all_files if substr in f]

  return matching_files


# Function that sorts files in a more sensible order ([202, 10, 3] -> [3, 10, 202])
def sort_files(files):
  files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))


# Function that reads a vector file as given by the C++ library and converts it into a numpy Python
def read_matrix(filename, path, rows, cols):

  tmp = np.loadtxt(path + '/' + filename)
  tmp = np.delete(tmp, 0, 1)  # Delete the index column
  mat = tmp.reshape(rows, cols)

  return mat
