# Varun Reddy Patlolla
# Project 2: Image Retrieval Makefile

# Compiler and Flags
CXX = g++
# -std=c++17 is required for std::filesystem used in FeatureFile.cpp
CXXFLAGS = -std=c++17 -Wall -g `pkg-config --cflags opencv4`
LDLIBS = `pkg-config --libs opencv4`

# Shared Object Files (The "Library" part of your code)
SH_OBJS = features.o csv_util.o
DEPS = features.h csv_util.h

# Final Executable Names
TARGETS = FeatureFile Matcher

# Default 'make' command builds everything
all: $(TARGETS)

# Link Rule for FeatureFile (The "Offline" Extraction Program)
FeatureFile: FeatureFile.o $(SH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

# Link Rule for Matcher (The "Online" Search Program)
Matcher: Matcher.o $(SH_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

# Generic Rule for compiling .cpp files into .o files
# This ensures that if a .h file changes, the .cpp files are recompiled
%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up rule to reset the workspace
clean:
	rm -f *.o $(TARGETS) database.csv