# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/esdavide/Desktop/Coding/MH/cec2017real/MH-CEC17

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/esdavide/Desktop/Coding/MH/cec2017real/MH-CEC17

# Include any dependencies generated for this target.
include CMakeFiles/testrandom.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/testrandom.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/testrandom.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/testrandom.dir/flags.make

CMakeFiles/testrandom.dir/testrandom.cc.o: CMakeFiles/testrandom.dir/flags.make
CMakeFiles/testrandom.dir/testrandom.cc.o: testrandom.cc
CMakeFiles/testrandom.dir/testrandom.cc.o: CMakeFiles/testrandom.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/esdavide/Desktop/Coding/MH/cec2017real/MH-CEC17/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/testrandom.dir/testrandom.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/testrandom.dir/testrandom.cc.o -MF CMakeFiles/testrandom.dir/testrandom.cc.o.d -o CMakeFiles/testrandom.dir/testrandom.cc.o -c /home/esdavide/Desktop/Coding/MH/cec2017real/MH-CEC17/testrandom.cc

CMakeFiles/testrandom.dir/testrandom.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/testrandom.dir/testrandom.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/esdavide/Desktop/Coding/MH/cec2017real/MH-CEC17/testrandom.cc > CMakeFiles/testrandom.dir/testrandom.cc.i

CMakeFiles/testrandom.dir/testrandom.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/testrandom.dir/testrandom.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/esdavide/Desktop/Coding/MH/cec2017real/MH-CEC17/testrandom.cc -o CMakeFiles/testrandom.dir/testrandom.cc.s

# Object files for target testrandom
testrandom_OBJECTS = \
"CMakeFiles/testrandom.dir/testrandom.cc.o"

# External object files for target testrandom
testrandom_EXTERNAL_OBJECTS =

testrandom: CMakeFiles/testrandom.dir/testrandom.cc.o
testrandom: CMakeFiles/testrandom.dir/build.make
testrandom: libcec17_test_func.so
testrandom: CMakeFiles/testrandom.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/esdavide/Desktop/Coding/MH/cec2017real/MH-CEC17/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable testrandom"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/testrandom.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/testrandom.dir/build: testrandom
.PHONY : CMakeFiles/testrandom.dir/build

CMakeFiles/testrandom.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/testrandom.dir/cmake_clean.cmake
.PHONY : CMakeFiles/testrandom.dir/clean

CMakeFiles/testrandom.dir/depend:
	cd /home/esdavide/Desktop/Coding/MH/cec2017real/MH-CEC17 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/esdavide/Desktop/Coding/MH/cec2017real/MH-CEC17 /home/esdavide/Desktop/Coding/MH/cec2017real/MH-CEC17 /home/esdavide/Desktop/Coding/MH/cec2017real/MH-CEC17 /home/esdavide/Desktop/Coding/MH/cec2017real/MH-CEC17 /home/esdavide/Desktop/Coding/MH/cec2017real/MH-CEC17/CMakeFiles/testrandom.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/testrandom.dir/depend

