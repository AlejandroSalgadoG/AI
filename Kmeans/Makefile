CUDA_INCLUDE_PATH=/opt/cuda/include
NVCC_OPTS=-Wno-deprecated-gpu-targets

Main: Kmeans.o Utilities.o Main.o
	nvcc $^ -o Main $(NVCC_OPTS)

Main.o: Main.cpp
	g++ -c Main.cpp -I $(CUDA_INCLUDE_PATH)

Utilities.o: Utilities.cpp Utilities.h
	g++ -c Utilities.cpp

Kmeans.o: Kmeans.cu Kmeans.h
	nvcc -c Kmeans.cu $(NVCC_OPTS)

clean:
	rm -f *.o Main

exe:
	./Main
