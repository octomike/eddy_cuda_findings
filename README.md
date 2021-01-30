## 1. gencode compatibility

What I understand from the [GPU Compilation docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#application-compatibility) from Nvidia is that we can build fat binaries to include `cubin` architecture-specific files and/or `ptx` virtual architecture files that can be JIT compiled on *any* GPU newer than the ptx version.

In essence that means, when we compile with `arch=compute_62,code=sm_62`, the fat binary contains only one cubin that is only compatible with `compute_62` ([latest Pascal](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)).

The following example binaries built from the [Makefile](Makefile), based on https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/, are being tested with CUDA 9.2 runtime, driver 418.152.00 (compatible up to Cuda 10.1) and a Quadro RTX 5000 (Turing, `sm_75`):

```
$ cuobjdump saxpy_incompatible

Fatbin elf code:
================
arch = sm_62
code version = [1,7]
producer = <unknown>
host = linux
compile_size = 64bit

Fatbin elf code:
================
arch = sm_62
code version = [1,7]
producer = cuda
host = linux
compile_size = 64bit

$ ./saxpy_incompatible
CUDA ERROR code: no kernel image is available for execution on the device
Max error: 2.000000
```

To be future proof we can use `arch=compute_62,code=compute_62`,  (`compute_62` is the virtual architecture, `sm_62` the actual architectre) which creates ptx code that is JIT compiled on newer devices and should always work, but loads much slower:

```
$ cuobjdump saxpy_future_compatible

Fatbin ptx code:
================
arch = sm_62
code version = [6,2]
producer = cuda
host = linux
compile_size = 64bit
compressed
ptxasOptions =

$ ./saxpy_future_compatible
Max error: 0.000000
```

Interestingly this example code runs on a `sm_75` device with cubin actual code
compiled for `sm_70` (`-gencode arch=compute_70,code=sm_70`) as well, probably
because `sm_70` is a somewhat compatible subset of `sm_75`:


```
$ cuobjdump saxpy_loosely_compatible

Fatbin elf code:
================
arch = sm_70
code version = [1,7]
producer = <unknown>
host = linux
compile_size = 64bit

Fatbin elf code:
================
arch = sm_70
code version = [1,7]
producer = cuda
host = linux
compile_size = 64bit

$ ./saxpy_loosely_compatible
Max error: 0.000000
```

In [another location](https://docs.nvidia.com/cuda/turing-compatibility-guide/index.html#building-turing-compatible-apps-using-cuda-10-0) in the CUDA docs I found the following pattern, which combines the best of the two worlds:

```
/usr/local/cuda/bin/nvcc
  -gencode=arch=compute_50,code=sm_50
  -gencode=arch=compute_52,code=sm_52
  -gencode=arch=compute_60,code=sm_60
  -gencode=arch=compute_61,code=sm_61
  -gencode=arch=compute_70,code=sm_70
  -gencode=arch=compute_75,code=sm_75
  -gencode=arch=compute_75,code=compute_75
  -O2 -o mykernel.o -c mykernel.cu
```

and I think this is what could be use for eddy_cuda as well. It will contain cubin files for every actual architecture available at compile time (for Cuda 9.1 this goes up to `compute_72`). That highest virtual architecture should then *also* be available as virtual ptx code so it can be JIT compiled on newer devices with newer Cuda runtimes/drivers.


## 2. cuBLAS issue with `cublasSsyrk`

Using a [minimal example](example.cu) that calls `cublasSsyrk` once, adapted from the [cuBLAS docs](https://docs.nvidia.com/cuda/cublas/index.html#example-code) I found there seems to be a runtime specific issue for *CUDA 9.1 only* that breaks execution on the Quadro device.

I pulled some cuda development/runtime containers with:

```
singularity pull cuda-10.1.sif docker://nvidia/cuda:10.1-devel-ubuntu18.04
singularity pull cuda-9.2.sif docker://nvidia/cuda:9.2-devel-ubuntu18.04
singularity pull cuda-9.1.sif docker://nvidia/cuda:10.1-devel-ubuntu16.04
singularity pull cuda-8.sif docker://nvidia/cuda:8.0-devel-ubuntu16.04
```

and ran the example on every one of them (driver supports only up to 10.1):

```
$ singularity exec --nv cuda-8.sif bash -c 'make example ; ./example'
nvcc -lcublas example.cu -o example
nvcc warning : The 'compute_20', 'sm_20', and 'sm_21' architectures are deprecated, and may be removed in a future release (Use -Wno-deprecated-gpu-targets to suppress warning).
      4
      4      4
      4      4      4
```

```
$ singularity exec --nv cuda-9.1.sif bash -c 'make example ; ./example'
nvcc -lcublas example.cu -o example
cublasSsyrk failed: CUBLAS_STATUS_EXECUTION_FAILED
```

```
$ singularity exec --nv cuda-9.2.sif bash -c 'make example ; ./example'
nvcc -lcublas example.cu -o example
      4
      4      4
      4      4      4
```

```
$ singularity exec --nv cuda-10.1.sif bash -c 'make example ; ./example'
nvcc -lcublas example.cu -o example
      4
      4      4
      4      4      4

```
