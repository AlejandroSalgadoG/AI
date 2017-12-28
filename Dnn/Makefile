Main: Loss.o Activations.o NNet.o Main.o
	g++ $^ -o $@

Main.o: Main.cpp
	g++ -c $<

NNet.o: NNet.cpp NNet.h
	g++ -c $<

Activations.o: Activations.cpp Activations.h
	g++ -c $<

Loss.o: Loss.cpp Loss.h
	g++ -c $<

clean:
	rm -f *.o Main

exe:
	./Main
