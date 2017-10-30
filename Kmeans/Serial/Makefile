Main: Kmeans.o Utilities.o Main.o
	g++ $^ -o Main

Main.o: Main.cpp
	g++ -c Main.cpp

Utilities.o: Utilities.cpp Utilities.h
	g++ -c Utilities.cpp

Kmeans.o: Kmeans.cpp Kmeans.h
	g++ -c Kmeans.cpp

clean:
	rm -f *.o Main

exe:
	./Main
