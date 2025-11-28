# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra
OPTFLAGS = -O3 -march=native -DNDEBUG
DEBUGFLAGS = -g -O0 -DDEBUG -fsanitize=address,undefined
PROFFLAGS = -O3 -g -pg

# Source files
SOURCES = linear_model.cpp neural_net_model.cpp rmi.cpp
HEADERS = learned_index.hpp btree.hpp dataset_loader.hpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)
DEBUG_OBJECTS = $(SOURCES:.cpp=_debug.o)

# Executables
BENCHMARK = benchmark
BENCHMARK_DEBUG = benchmark_debug
TEST = test_loaders

# Default target - optimized benchmark
.PHONY: all
all: $(BENCHMARK)

# Optimized benchmark build
$(BENCHMARK): $(OBJECTS) benchmark.cpp $(HEADERS)
	@echo "Building optimized benchmark..."
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ $(OBJECTS) benchmark.cpp
	@echo "✓ Built: $(BENCHMARK)"

# Pattern rule for optimized object files
%.o: %.cpp $(HEADERS)
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -c $< -o $@

# Debug build
.PHONY: debug
debug: CXXFLAGS += $(DEBUGFLAGS)
debug: $(BENCHMARK_DEBUG)

$(BENCHMARK_DEBUG): $(DEBUG_OBJECTS) benchmark.cpp $(HEADERS)
	@echo "Building debug benchmark..."
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) -o $@ $(DEBUG_OBJECTS) benchmark.cpp
	@echo "✓ Built: $(BENCHMARK_DEBUG)"

# Pattern rule for debug object files
%_debug.o: %.cpp $(HEADERS)
	@echo "Compiling $< (debug)..."
	$(CXX) $(CXXFLAGS) $(DEBUGFLAGS) -c $< -o $@

# Profile build
.PHONY: profile
profile: CXXFLAGS += $(PROFFLAGS)
profile: clean
	@echo "Building profiling benchmark..."
	$(CXX) $(CXXFLAGS) $(PROFFLAGS) -o $(BENCHMARK) $(SOURCES) benchmark.cpp
	@echo "✓ Built for profiling. Run with: make run-profile"

# Test loaders
$(TEST): $(OBJECTS) test_loaders.cpp $(HEADERS)
	@echo "Building test_loaders..."
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) -o $@ $(OBJECTS) test_loaders.cpp
	@echo "✓ Built: $(TEST)"

.PHONY: test
test: $(TEST)
	@echo "Running loader tests..."
	./$(TEST)

# Run targets
.PHONY: run
run: $(BENCHMARK)
	@echo "Running benchmark..."
	./$(BENCHMARK)

.PHONY: run-debug
run-debug: debug
	@echo "Running debug benchmark..."
	./$(BENCHMARK_DEBUG)

.PHONY: run-profile
run-profile: profile
	@echo "Running profiling benchmark..."
	./$(BENCHMARK)
	@echo "Generating profile report..."
	gprof $(BENCHMARK) gmon.out > profile_report.txt
	@echo "✓ Profile saved to profile_report.txt"

# Memory check with valgrind
.PHONY: valgrind
valgrind: debug
	@echo "Running valgrind..."
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(BENCHMARK_DEBUG)

.PHONY: memcheck
memcheck: debug
	@echo "Running memory check..."
	valgrind --tool=memcheck --leak-check=full ./$(BENCHMARK_DEBUG)

# Clean targets
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(OBJECTS) $(DEBUG_OBJECTS)
	rm -f $(BENCHMARK) $(BENCHMARK_DEBUG) $(TEST)
	rm -f gmon.out profile_report.txt
	rm -f *.o *_debug.o
	@echo "✓ Clean complete"

.PHONY: cleanall
cleanall: clean
	@echo "Cleaning data files..."
	rm -rf data/*.csv data/*.txt
	@echo "✓ Clean all complete"

# Rebuild
.PHONY: rebuild
rebuild: clean all

# Help target
.PHONY: help
help:
	@echo "Learned Index Benchmark - Makefile"
	@echo "===================================="
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build optimized benchmark (default)"
	@echo "  make all          - Same as default"
	@echo "  make debug        - Build with debug symbols and sanitizers"
	@echo "  make profile      - Build with profiling enabled"
	@echo "  make test         - Build and run loader tests"
	@echo ""
	@echo "Run targets:"
	@echo "  make run          - Build and run optimized benchmark"
	@echo "  make run-debug    - Build and run debug version"
	@echo "  make run-profile  - Run with gprof profiling"
	@echo ""
	@echo "Analysis targets:"
	@echo "  make valgrind     - Run valgrind memory checker"
	@echo "  make memcheck     - Run detailed memory check"
	@echo ""
	@echo "Utility targets:"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make cleanall     - Remove build artifacts and data files"
	@echo "  make rebuild      - Clean and rebuild"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Build info:"
	@echo "  Compiler: $(CXX)"
	@echo "  Flags:    $(CXXFLAGS)"
	@echo "  Optimize: $(OPTFLAGS)"

# Info target
.PHONY: info
info:
	@echo "Project: Learned Index Structures"
	@echo "=================================="
	@echo "Source files:  $(SOURCES)"
	@echo "Headers:       $(HEADERS)"
	@echo "Executables:   $(BENCHMARK), $(TEST)"
	@echo "Compiler:      $(CXX)"
	@echo "C++ Standard:  C++17"
	@echo ""
	@echo "Datasets:"
	@if [ -f data/florida_nodes.csv ]; then \
		echo "  ✓ Florida OSM nodes"; \
	else \
		echo "  ✗ Florida OSM nodes (missing)"; \
	fi
	@if [ -f data/NASA_access_log_Jul95 ]; then \
		echo "  ✓ NASA web logs"; \
	else \
		echo "  ✗ NASA web logs (missing)"; \
	fi


# Plot generation
.PHONY: plot
plot: $(BENCHMARK)
	@echo "Running benchmark and generating plots..."
	@./$(BENCHMARK)
	@if command -v python3 >/dev/null 2>&1; then \
		python3 plot_results.py; \
	else \
		echo "Error: python3 not found. Install Python 3 to generate plots."; \
		exit 1; \
	fi

.PHONY: plot-only
plot-only:
	@echo "Generating plots from existing results..."
	@if [ ! -f benchmark_results.json ]; then \
		echo "Error: benchmark_results.json not found. Run 'make run' first."; \
		exit 1; \
	fi
	@python3 plot_results.py

.PHONY: install-plot-deps
install-plot-deps:
	@echo "Installing Python plotting dependencies..."
	pip3 install matplotlib numpy
	@echo "✓ Dependencies installed"

# Phony targets for safety
.PHONY: all debug profile test run run-debug run-profile valgrind memcheck clean cleanall rebuild help info

