CUDA_INCLUDE_PATH=/opt/cuda/include
NVCC_OPTS=-Wno-deprecated-gpu-targets

Main: Kmeans.o Main.o
	nvcc Kmeans.o Main.o -o Main $(NVCC_OPTS)

Main.o: Main.cpp
	g++ -c Main.cpp -I $(CUDA_INCLUDE_PATH)

Kmeans.o: Kmeans.cu Kmeans.h
	nvcc -c Kmeans.cu $(NVCC_OPTS)

clean:
	rm -f *.o Main

exe:
	./Main
