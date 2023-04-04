#!/bin/bash

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -x

PADDLE_BUILD_ROOT=/shixiaowei02/Paddle/PaddleCUDA/Paddle/build_cuda
CINN_BUILD_ROOT=/shixiaowei02/CINN/build
PADDLE_NLP_ROOT=/shixiaowei02/PaddleNLP

export PYTHONPATH=$PYTHONPATH:$PADDLE_BUILD_ROOT/python

export CUDA_VISIBLE_DEVICES=0
export FLAGS_CONVERT_GRAPH_TO_PROGRAM=1
export FLAGS_fraction_of_gpu_memory_to_use=0.8
export NVIDIA_TF32_OVERRIDE=1
export FLAGS_cudnn_exhaustive_search=1
export FLAGS_conv_workspace_size_limit=4000

export FLAGS_use_cinn=1
#export FLAGS_prim_all="true"
export FLAGS_deny_cinn_ops="conv2d;conv2d_grad"
export FLAGS_use_reduce_split_pass=1


export FLAGS_cinn_open_fusion_optimize=1
export FLAGS_cinn_use_new_fusion_pass=1
export FLAGS_cinn_ir_schedule=1
export FLAGS_enable_pe_launch_cinn=0
export FLAGS_cinn_sync_run=1

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PADDLE_BUILD_ROOT}/third_party/mklml/src/extern_mklml/lib

USE_CINN=true

#rm -rf ${PADDLE_BUILD_ROOT}/third_party/CINN/src/external_cinn-build/libcinnapi.so
#rm -rf ${PADDLE_BUILD_ROOT}/third_party/CINN/src/external_cinn-build/dist/cinn/lib/libcinnapi.so
#ln -s ${CINN_BUILD_ROOT}/libcinnapi.so ${PADDLE_BUILD_ROOT}/third_party/CINN/src/external_cinn-build/libcinnapi.so
#ln -s ${CINN_BUILD_ROOT}/libcinnapi.so ${PADDLE_BUILD_ROOT}/third_party/CINN/src/external_cinn-build/dist/cinn/lib/libcinnapi.so

python $PADDLE_NLP_ROOT/model_zoo/ernie-3.0/run_seq_cls.py  --model_name_or_path ernie-3.0-medium-zh  --dataset afqmc --output_dir ./best_models --export_model_dir best_models/ --do_train --do_eval --do_export --config=$PADDLE_NLP_ROOT/model_zoo/ernie-3.0/configs/default.yml

