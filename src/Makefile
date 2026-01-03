NVCC = nvcc
CXX = g++
CXXFLAGS = -O3 -Wall
OMPFLAGS = -fopenmp
NVCCFLAGS = -O3 -arch=sm_86
CUDALIBFLAGS = -L/usr/local/cuda/lib64 -lcudart

CPU_BIN = run_cpu
OMP_BIN = run_omp
GPU_BIN = run_gpu

COLUMNS = "Size,Type,Bitonic_ms,g_kernel_time_ms,elements_per_sec,throughput_cycles,Baseline_ms"
TYPE = 0 1 2 3

all: $(CPU_BIN) $(OMP_BIN) $(GPU_BIN)

$(CPU_BIN): benchmark.cpp bitonic_cpu.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ $(CUDALIBFLAGS)

$(OMP_BIN): benchmark.cpp bitonic_omp.cpp
	$(CXX) -march=native $(CXXFLAGS) $(OMPFLAGS) $^ -o $@ $(CUDALIBFLAGS)

$(GPU_BIN): benchmark.cpp bitonic_gpu.cu
	$(NVCC) $(NVCCFLAGS) -x cu $^ -o $@


bench_cpu: $(CPU_BIN)
	@echo $(COLUMNS) > result_cpu.csv
	@for e in $$(seq 10 26); do \
		n=$$((1<<e)); \
		for t in $(TYPE); do ./$(CPU_BIN) $$n $$t >> result_cpu.csv; done; \
	done

bench_omp: $(OMP_BIN)
	@echo $(COLUMNS) > result_omp.csv
	@for e in $$(seq 10 26); do \
		n=$$((1<<e)); \
		for t in $(TYPE); do ./$(OMP_BIN) $$n $$t >> result_omp.csv; done; \
	done

bench_gpu: $(GPU_BIN)
	@echo $(COLUMNS) > result_gpu.csv
	@for e in $$(seq 10 26); do \
		n=$$((1<<e)); \
		for t in $(TYPE); do ./$(GPU_BIN) $$n $$t >> result_gpu.csv; done; \
	done

clean:
	rm -f $(CPU_BIN) $(OMP_BIN) $(GPU_BIN) *.csv