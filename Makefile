all:
	nvcc *.cu *.cpp -o main -lineinfo -arch=sm_80 -I .

run:
	./main 2048 2048 2048

clean:
	rm main *.ncu-rep