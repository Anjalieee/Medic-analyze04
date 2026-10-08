CXX      ?= clang++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -ffp-contract=off -Isrc -isystem third_party
# -ffp-contract=off : forbid fused multiply-add so floating point results match across builds

COMMON = src/common/image_io.cpp src/common/stages_common.cpp src/common/pipeline.cpp
SEQ    = src/sequential/stages_seq.cpp
OMP    = src/openmp/stages_omp.cpp

all: bin/mia_seq bin/mia_omp

bin/mia_seq: src/main.cpp $(COMMON) $(SEQ)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $^

bin/mia_omp: src/main.cpp $(COMMON) $(SEQ) $(OMP)
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -Xclang -fopenmp -DUSE_OPENMP -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -o $@ $^

deps:
	mkdir -p third_party/stb
	curl -L -o third_party/stb/stb_image.h       https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
	curl -L -o third_party/stb/stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

test: all
	python3 tests/test_correctness.py

clean:
	rm -rf bin

.PHONY: all deps test clean
