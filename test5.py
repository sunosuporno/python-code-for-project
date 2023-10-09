# def StringChallenge(input_string):
#   """Reverses the given string and returns it.

#   Args:
#     input_string: A string to be reversed.

#   Returns:
#     A string in reversed order.
#   """

#   reversed_string = input_string[::-1]
#   print(reversed_string)
#   return reversed_string

# # Example usage:
# print(StringChallenge(input("coderbyte")))


def reverse_string(input_string):
    # Reverse the string using slicing
    reversed_string = input_string[::-1]
    return reversed_string

# Get input from the user
input_string = input("Enter a string: ")

# Call the function and print the reversed string
reversed_result = reverse_string(input_string)
print("Reversed string:", reversed_result)




import numpy as np
import torch
from sklearn.linear_model import LinearRegression

def load_data(filename):

  with open(filename, "r") as f:
    X = np.array([float(x) for x in f.readline().strip().split(",")])
    Y = np.array([float(y) for y in f.readline().strip().split(",")])

  return X, Y

def linear_regression(X, Y):

  model = LinearRegression()
  model.fit(X[:, np.newaxis], Y)

  return model

def coefficient_of_determination(model, X, Y):

  Y_pred = model.predict(X[:, np.newaxis])
  SStot = np.sum((Y - Y.mean())**2)
  SSres = np.sum((Y_pred - Y)**2)

  return 1 - SSres / SStot

if __name__ == "__main__":
  X, Y = load_data("data.txt")

  model = linear_regression(X, Y)

  r_squared = coefficient_of_determination(model, X, Y)

  print(f"coefficient: {r_squared:.4f}")
