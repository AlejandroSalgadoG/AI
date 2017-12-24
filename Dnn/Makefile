Main: NNet.o Main.o
	g++ $^ -o $@

Main.o: Main.cpp
	g++ -c $<

NNet.o: NNet.cpp NNet.h
	g++ -c $<

clean:
	rm -f *.o Main

exe:
	./Main
