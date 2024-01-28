all:
	nvcc *.cu *.cpp -o main -lineinfo -arch=sm_80 -I .

clean:
	rm main 