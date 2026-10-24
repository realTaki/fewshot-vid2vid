# flake8: noqa
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


cuda_version = os.getenv('CUDA_VERSION')
print('CUDA_VERSION: {}'.format(cuda_version))

# Let PyTorch honor the caller's explicit TORCH_CUDA_ARCH_LIST.
# The installer sets its default; compile-only CI selects sm70 only.
nvcc_args = list()
nvcc_args.append('-Xcompiler')
nvcc_args.append('-Wall')
nvcc_args.append('-std=c++17')

setup(
    name='upfirdn2d_cuda',
    py_modules=['upfirdn2d'],
    ext_modules=[
        CUDAExtension('upfirdn2d_cuda', [
            './src/upfirdn2d_cuda.cc',
            './src/upfirdn2d_cuda_kernel.cu'
        ], extra_compile_args={'cxx': ['-Wall', '-std=c++17'],
                               'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
