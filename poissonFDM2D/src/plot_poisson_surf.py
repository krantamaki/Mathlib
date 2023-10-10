import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


# Defien global variables for the width and height of the plate
height = 10
width = 5


# Define global variables for the number of steps in width and height directions
n_height_steps = 50
n_width_steps = 25


# Define the duration of the simulation
duration = 1


# Define global variables for minimum and maximum temperatures in the plot
max_temp = 2
min_temp = -2


# Define global arrays that hold the times and the solved matrices
times = []
matrices = []


# Find the matrices
matrix_name = "heat_eq_t"

def find_files(substr):
  all_files = os.listdir()
  matching_files = [f for f in all_files if substr in f]
  return matching_files

matrix_files = find_files(matrix_name)

# Sort the files
matrix_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))


# Load to memory
n_matrices = len(matrix_files)
time_step = duration / n_matrices

for i in range(n_matrices):
  tmp = np.loadtxt(matrix_files[i])
  tmp = np.delete(tmp, 0, 1)  # Delete the index column
  mat = tmp.reshape(n_height_steps, n_width_steps)
  matrices.append(mat)
  times.append(i * time_step)


def plot_surf(matrix, time):
  plt.clf()
  plt.title(f"Temp at time {time:.2f}")
  plt.xlabel("Width")
  plt.ylabel("Height")

  plt.pcolormesh(matrix, cmap=plt.cm.jet, vmin=min_temp, vmax=max_temp)
  plt.colorbar()

  return plt


def animate(t_i):
  plot_surf(matrices[t_i], times[t_i])


# Define the animation
anim = animation.FuncAnimation(plt.figure(figsize=(4, 8)), animate, interval=1, frames=len(matrices), repeat=False)
anim.save("heat_equation_solution.gif", fps=60)