CC = g++
CFLAGS = -O3 -Wall
TARGET = sort_benchmark

HARDWAREINFO = hardware_specs.txt

all: $(TARGET)

$(TARGET): main.cpp
	@$(CC) $(CFLAGS) main.cpp -o $(TARGET)

bench: $(TARGET)
	@echo "Data size,Data type,Bitonic sort(ms),std::sort(ms),Bubble sort(ms)" > result.csv
	@for e in {10..25}; do \
		n=$((1<<e)) \
		for t in 0 1 2 3; do \
			./bitonic_single_core $n $t >> results.csv \
		done \
	done
	@echo "Benchmark complete. Results saved to result.csv"

clean:
	@rm -f $(TARGET) result.csv $(OUTPUT_FILE)

# Target to generate the hardware specification file
# can only run on wsl, not compatible with linux
hardware_spec:
	@echo "Gathering hardware specifications for Mac..."
	# Clear the output file if it exists, then add a header
	@echo "--- Hardware Specifications ---" > $(HARDWAREINFO)

	# Get CPU and processor info (model, cores, speed)
	@echo "--- CPU/Processor Info ---" >> $(HARDWAREINFO)
	# Model Name
	@sysctl -n machdep.cpu.brand_string | sed 's/^/Model Name: /' >> $(HARDWAREINFO)
	# Total Number of Cores
	@sysctl -n machdep.cpu.core_count | sed 's/^/Total Cores: /' >> $(HARDWAREINFO)
	# Logical Processors (Threads)
	@sysctl -n machdep.cpu.thread_count | sed 's/^/Logical Processors (Threads): /' >> $(HARDWAREINFO)
	# Current CPU Frequency (Max)
	@sysctl -n hw.cpufrequency | awk '{print "Max CPU Frequency: " $0 " Hz"}' >> $(HARDWAREINFO)
	@echo "" >> $(HARDWAREINFO)

	# Get Memory Info
	@echo "--- Memory Info ---" >> $(HARDWAREINFO)
	# Total Physical Memory (bytes) 1073741824
	@echo "" >> $(HARDWAREINFO)
	@sysctl -n hw.memsize | awk '{printf "Total Physical Memory: %.2f GB\n", $$1/1073741824}' >> $(HARDWAREINFO)
	@echo "Hardware specifications have been successfully saved to $(HARDWAREINFO)"
