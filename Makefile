# Indicates compilator to use
CC      = g++

# Specifies compilator options
CFLAGS  = -O2 -Wall 
LDFLAGS = 
LDLIBS  = 

# Files extensions .cpp, .o
SUFFIXES = .cpp .o 
.SUFFIXES: $(SUFFIXES) .

# Name of the main program
PROG  = refine

# Object files .o necessary to build the main program
OBJS  = main.o
 
all: $(PROG)

# Compilation and link
$(PROG): $(OBJS)
	$(CC) $(LDFLAGS) -o $(PROG) $(OBJS) $(LDLIBS)

.cpp.o:
	$(CC)   $(CFLAGS) -c $< -o $@

clean:
	-rm -f $(PROG)
	-rm -f *.o
