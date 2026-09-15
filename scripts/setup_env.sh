#!/usr/bin/env bash
# DraftRouter 云端(AutoDL RTX4090, CUDA 13 驱动宿主)环境初始化
# 用法: ssh autodl "bash /root/autodl-tmp/draftrouter/scripts/setup_env.sh"
# 目标: 实例重建后一条命令恢复环境。P6 复现文档底稿。
# 前提: nvidia-smi 显示 CUDA Version >= 13.0(宿主驱动 >= 580);12.8 宿主跑不动 vLLM >= 0.27
set -euo pipefail

CONDA=/root/miniconda3/bin
HF=/root/autodl-tmp/hf-cache

# 0. 驱动检查:CUDA 13 宿主才允许装 0.29 栈
DRV_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRV_CUDA" -lt 580 ]; then
  echo "FATAL: 宿主驱动 $DRV_CUDA < 580 (CUDA 12.x)。vLLM 0.29 需 CUDA 13 宿主,请更换实例。"; exit 1
fi

# 1. 独立 conda 环境(Python 3.10)
if [ ! -d /root/miniconda3/envs/draftrouter ]; then
  $CONDA/conda create -y -n draftrouter python=3.10
fi
PIP=/root/miniconda3/envs/draftrouter/bin/pip
PY=/root/miniconda3/envs/draftrouter/bin/python

# 2. vLLM 0.29.0(其依赖自带 torch 2.13 + CUDA 13 库,勿预装 torch——会装错变体再被覆盖)
$PIP install 'vllm==0.29.0'

# 3. 项目依赖(pip 走默认国内源,不开学术加速——加速只对 github/HF 有效且拖慢 pip)
$PIP install fasttext transformers accelerate datasets

# 4. 模型权重下载到数据盘(关机不丢)。此段必须开学术加速,并绕开 Xet 协议(401 问题)
source /etc/network_turbo 2>/dev/null || true
export HF_HOME=$HF HF_HUB_DISABLE_XET=1
MODELS="Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B"
for m in $MODELS; do
  for try in 1 2 3; do
    $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$m')" \
      && echo "OK $m" && break \
      || { echo "RETRY($try) $m"; sleep 10; [ "$try" = 3 ] && echo "FAIL $m —— 重跑本脚本可续传"; }
  done
done

# 5. 自检
$PY -c "import torch,vllm; assert torch.cuda.is_available(); print('torch',torch.__version__,'| vllm',vllm.__version__,'|',torch.cuda.get_device_name(0))"
echo "环境就绪。python: $PY"
