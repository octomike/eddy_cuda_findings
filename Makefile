.PHONY: example saxpy

example:
	nvcc -lcublas example.cu -o example

saxpy:
	nvcc -gencode arch=compute_62,code=sm_62 \
		 saxpy.cu -o saxpy_incompatible
	nvcc -gencode arch=compute_70,code=sm_70 \
		 saxpy.cu -o saxpy_loosely_compatible
	nvcc -gencode arch=compute_62,code=compute_62 \
		 saxpy.cu -o saxpy_future_compatible
