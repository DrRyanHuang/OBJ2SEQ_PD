/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

// No need hand file?

#include <vector>
#include "paddle/extension.h"


// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>


paddle::Tensor
ms_deform_attn_cpu_forward(
    const paddle::Tensor &value, 
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_loc,
    const paddle::Tensor &attn_weight,
    const int im2col_step)
{
    // AT_ERROR("Not implement on cpu");
    PD_THROW("Not implement on cpu");
}

std::vector<paddle::Tensor>
ms_deform_attn_cpu_backward(
    const paddle::Tensor &value, 
    const paddle::Tensor &spatial_shapes,
    const paddle::Tensor &level_start_index,
    const paddle::Tensor &sampling_loc,
    const paddle::Tensor &attn_weight,
    const paddle::Tensor &grad_output,
    const int im2col_step)
{
    // AT_ERROR("Not implement on cpu");
    PD_THROW("Not implement on cpu");
}

