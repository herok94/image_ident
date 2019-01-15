CC = gcc
CXX = g++

CXXFLAGS = -I/home/zhangliu/work/opencv/alinx_heijin_opencv/opencv_lib/include -I/home/zhangliu/work/opencv/alinx_heijin_opencv/opencv_lib/include/opencv -I/home/zhangliu/work/opencv/alinx_heijin_opencv/opencv_lib/include/opencv2

LIBS = -L/home/zhangliu/work/opencv/alinx_heijin_opencv/opencv_lib/lib64  -l

objects = basicopencv.o  Image.o main.o 

app : $(objects)  
	$(CXX) -o app $(objects) $(LIBS)

basicopencv.o: basicopencv.cpp  basicopencv.h
	g++ -c $(CXXFLAGS) $< -o $@

Image.o: Image.cpp Image.h
	g++ -c $(CXXFLAGS) $< -o $@

.PHONY :clean
clean:
	rm -rf app $(objects)

