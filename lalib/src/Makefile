CC      = g++                # Choose the compiler
CFLAGS  = -fopenmp -Wall     # Needed C-flags like openMP
LIBS    = -lm                # Needed libraries
SRCS    = $(wildcard *.cpp)  # Collect all .cpp files from current directory

lalib.o:
	$(CC) $(CFLAGS) $(SRCS) $(GEN) $(LIBS) -o $@
