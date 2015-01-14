all: parallel_SOM

parallel_SOM: parallel_SOM.o drawPPMmap.o PPM.o
	g++ -o parallel_SOM parallel_SOM.o drawPPMmap.o PPM.o

parallel_SOM.o: parallel_SOM.cpp
	g++ -c parallel_SOM.cpp

drawHTMLMap.o: ../drawHTMLmap/drawHTMLMap.cpp
	g++ -c ../drawHTMLmap/drawHTMLMap.cpp

drawPPMmap.o: ../ppm/drawPPMmap.cpp
	g++ -c ../ppm/drawPPMmap.cpp

PPM.o: ../ppm/PPM.c
	g++ -c ../ppm/PPM.c