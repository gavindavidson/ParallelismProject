all: parallel_SOM

parallel_SOM: parallel_SOM.o drawPPMmap.o PPM.o
	g++ -o parallel_SOM parallel_SOM.o drawPPMmap.o PPM.o -lrt  /opt/AMDAPP/lib/x86_64/libOpenCL.so
	#-lOpenCL
	#/opt/AMDAPP/lib/x86_64/libOpenCL.so

parallel_SOM.o: parallel_SOM.cpp
	g++ -c -Wno-deprecated-declarations parallel_SOM.cpp -I/opt/AMDAPP/include
	#-I/opt/AMDAPP/include

drawHTMLMap.o: ../drawHTMLmap/drawHTMLMap.cpp
	g++ -c ../drawHTMLmap/drawHTMLMap.cpp

drawPPMmap.o: ../ppm/drawPPMmap.cpp
	g++ -c ../ppm/drawPPMmap.cpp

PPM.o: ../ppm/PPM.c
	g++ -c ../ppm/PPM.c

# manhattan_distance.o: manhattan_distance.c
# 	g++ -c manhattan_distance.c

# update_weight.o: update_weight.c
# 	g++ -c update_weight.c
