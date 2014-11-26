all: parallel_SOM

parallel_SOM: parallel_SOM.o drawHTMLMap.o
	g++ -o parallel_SOM parallel_SOM.o drawHTMLMap.o

parallel_SOM.o: parallel_SOM.cpp
	g++ -c parallel_SOM.cpp

drawHTMLMap.o: ../drawHTMLmap/drawHTMLMap.cpp
	g++ -c ../drawHTMLmap/drawHTMLMap.cpp