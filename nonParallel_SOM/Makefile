all: nonParallel_SOM

nonParallel_SOM: nonParallel_SOM.o drawHTMLMap.o
	g++ -o nonParallel_SOM nonParallel_SOM.o drawHTMLMap.o

nonParallel_SOM.o: nonParallel_SOM.cpp
	g++ -c nonParallel_SOM.cpp

drawHTMLMap.o: ../drawHTMLmap/drawHTMLMap.cpp
	g++ -c ../drawHTMLmap/drawHTMLMap.cpp