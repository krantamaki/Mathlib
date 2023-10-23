"""
MAIN PYTHON SOURCE
"""
import shutil
import atexit
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from tools import *


# Required global variables. These will be modified in the main function
WIDTH = -1.
N_WIDTH_POINTS = 0.
HEIGHT = -1.
N_HEIGHT_POINTS = 0.
MIN_TEMP = -1.
MAX_TEMP = -1.
MATRICES = []
TIMES = []


# Constant predefined values for the config file
METHOD = "cg"
VERBOSITY = 3
CONVERGENCE_TOLERANCE = 1e-6
MAX_ITER = 10000
SAVE_DIR = "tmp"
SAVE_NAME = "heat_eq"
STOP_UNCONVERGED = "false"
N_TIME_POINTS = -1


# Other constant variables
FRAME_INTERVAL = 10
FPS = 30


# Function that cleans up a directory
def cleanup():
  for filename in os.listdir(SAVE_DIR):
    if filename == "README.MD":
      continue

    filepath = os.path.join(SAVE_DIR, filename)
    if os.path.isfile(filepath) or os.path.islink(filepath):
      os.unlink(filepath)
    elif os.path.isdir(filepath):
      shutil.rmtree(filepath)

atexit.register(cleanup)  # Empty the tmp directory at exit


# Function for plotting the surface
def plot_surf(matrix, time):

  plt.clf()
  plt.title(f"Temp at time {time:.2f}")
  plt.xlabel("Width")
  plt.ylabel("Height")

  plt.pcolormesh(np.arange(0, WIDTH, WIDTH / N_WIDTH_POINTS), 
                 np.arange(0, HEIGHT, HEIGHT / N_HEIGHT_POINTS), 
                 matrix, cmap=plt.cm.jet, vmin=MIN_TEMP, vmax=MAX_TEMP)
  plt.colorbar()

  return plt


# Function called per frame in animation
def animate(t_i):
  plot_surf(MATRICES[t_i], TIMES[t_i])


def main():

  # Define the global variables which we will modify
  global MIN_TEMP, MAX_TEMP, MATRICES, TIMES, WIDTH, N_WIDTH_POINTS, HEIGHT, N_HEIGHT_POINTS

  # Gather the required inputs from user
  print("Please specify the following information:\n")

  upper_bound = get_input("What is the temperature on upper boundary? (float)", float)
  lower_bound = get_input("What is the temperature on lower boundary? (float)", float)
  left_bound = get_input("What is the temperature on left boundary? (float)", float)
  right_bound = get_input("What is the temperature on right boundary? (float)", float)
  height = get_input("What is the height of the surface? (float)", float)
  n_height_points = get_input("Into how many points should the height be divided? (int)", int)
  width = get_input("What is the width of the surface? (float)", float)
  n_width_points = get_input("Into how many points should the width be divided? (int)", int)
  duration = get_input("What is set as the duration of the simulation? (float)", float)
  init_temp = get_input("What is the initial temperature of the surface? (float)", float)
  thermal_diffusivity = get_input("What is the thermal diffusivity? (float)", float)
  anim_name = get_input("What should be used as the filename for the final animation? (str)", str)

  # Set the global variables
  MIN_TEMP = min([init_temp, upper_bound, left_bound, lower_bound, right_bound])
  MAX_TEMP = max([init_temp, upper_bound, left_bound, lower_bound, right_bound])
  HEIGHT = height
  WIDTH = width
  N_HEIGHT_POINTS = n_height_points
  N_WIDTH_POINTS = n_width_points

  # Form the config file
  config_filename = "tmp/py_config.txt"
  config_tups = [("upper_bound", "float", upper_bound),
                 ("lower_bound", "float", lower_bound),
                 ("left_bound", "float", left_bound),
                 ("right_bound", "float", right_bound),
                 ("height", "float", height),
                 ("width", "float", width),
                 ("n_height_points", "int", n_height_points),
                 ("n_width_points", "int", n_width_points),
                 ("duration", "float", duration),
                 ("initial_temp", "float", init_temp),
                 ("thermal_diffusivity", "float", thermal_diffusivity),
                 ("method", "string", METHOD),
                 ("verbosity", "int", VERBOSITY),
                 ("convergence_tolerance", "float", CONVERGENCE_TOLERANCE),
                 ("max_iter", "int", MAX_ITER),
                 ("save_dir", "string", SAVE_DIR),
                 ("save_name", "string", SAVE_NAME),
                 ("stop_unconverged", "bool", STOP_UNCONVERGED),
                 ("n_time_points", "int", N_TIME_POINTS)]

  form_config(config_tups, config_filename)

  # Solve the problem
  print("Solving the problem. This may take a while...\n")

  completedProc = subprocess.run(["./poissonFDM2D.o", config_filename, f"{SAVE_DIR}/log.txt"], capture_output=True)
  success = True

  if completedProc.returncode == 0:
    print("Problem solved successfully")
  elif completedProc.returncode == -6:
    print("Solver exited with runtime error!")
    print("The error was:")
    print(completedProc.stderr)
    success = False
  else:
    print("Error occured!")
    success = False

  # Exit in failure
  if not success:
    print("Exiting program...")
    exit()

  # Otherwise output the log and continue to animation

  print()

  if (get_input("Do you want to print the solver log file? (yes/no)", str, choices=["yes", "no"]) == "yes"):
    with open(f"{SAVE_DIR}/log.txt", "r") as f:
      for line in f:
        print(line, end='')

  print()

  # Find the matrix files
  matrix_name = SAVE_NAME + "_t"

  matrix_files = find_files(matrix_name, SAVE_DIR)
  sort_files(matrix_files)

  # Load them to memory
  n_matrices = len(matrix_files)
  time_step = duration / n_matrices

  for i in range(n_matrices):
    if (i % FRAME_INTERVAL == 0):
      MATRICES.append(read_matrix(matrix_files[i], SAVE_DIR, n_height_points, n_width_points))
      TIMES.append(i * time_step)

  # Define figure size as a sensible multiple of width and height
  fig_mult = height / 10.0
  figsize = (int(fig_mult * width), int(fig_mult * height))

  # Animate!
  print("Animating the solution. This may take a while...")
  anim = animation.FuncAnimation(plt.figure(figsize=figsize), animate, interval=1, frames=len(MATRICES), repeat=False)

  # Save the animation
  if len(anim_name.split('.')) == 1:
    anim.save(anim_name + ".gif", fps=FPS)
  else:
    anim.save(anim_name, fps=FPS)


if __name__ == "__main__":
  main()
