NVCC = nvcc
CXX = g++
CXXFLAGS = -O3 -Wall
OMPFLAGS = -fopenmp
NVCCFLAGS = -O3 -arch=sm_86
CUDALIBFLAGS = -L/usr/local/cuda/lib64 -lcudart

CPU_BIN = run_cpu
OMP_BIN = run_omp
GPU_BIN = run_gpu

# Powers of 2 from 2^10 to 2^21
SIZES = 1024 4096 16384 65536 262144 1048576 4194304 33554432

all: $(CPU_BIN) $(OMP_BIN) $(GPU_BIN)

$(CPU_BIN): benchmark.cpp bitonic_cpu.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CUDALIBFLAGS)

$(OMP_BIN): benchmark.cpp bitonic_omp.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@ $(CUDALIBFLAGS)

$(GPU_BIN): benchmark.cpp bitonic_gpu.cu
	$(NVCC) $(NVCCFLAGS) -x cu $^ -o $@

bench_cpu: $(CPU_BIN)
	@echo "Size,Type,Bitonic_ms,StdSort_ms,Bubble_ms" > result_cpu.csv
	@for n in $(SIZES); do \
		for t in 0 1 2 3; do ./$(CPU_BIN) $$n $$t >> result_cpu.csv; done \
	done

bench_omp: $(OMP_BIN)
	@echo "Size,Type,Bitonic_ms,StdSort_ms,BubbleOMP_ms" > result_omp.csv
	@for n in $(SIZES); do \
		for t in 0 1 2 3; do ./$(OMP_BIN) $$n $$t >> result_omp.csv; done \
	done

bench_gpu: $(GPU_BIN)
	@echo "Size,Type,Bitonic_ms,ThrustSort_ms,BubbleCUDA_ms" > result_gpu.csv
	@for n in $(SIZES); do \
		for t in 0 1 2 3; do ./$(GPU_BIN) $$n $$t >> result_gpu.csv; done \
	done

clean:
	rm -f $(CPU_BIN) $(OMP_BIN) $(GPU_BIN) *.csv