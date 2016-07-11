OneParam: OneParam.o
	g++ OneParam.o -o OneParam

OneParam.o: OneParam.C
	g++ -c OneParam.C

clean:
	rm -f *.o OneParam

exe:
	./OneParam Data.txt 5
