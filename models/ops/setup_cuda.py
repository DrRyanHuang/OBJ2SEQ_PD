from paddle.utils.cpp_extension import CUDAExtension, setup

setup(
    name='MultiScaleDeformableAttention',
    ext_modules=CUDAExtension(
        sources=['relu_cuda.cc', 'relu_cuda.cu']
    )
)