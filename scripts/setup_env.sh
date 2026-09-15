#!/usr/bin/env bash
# DraftRouter 云端(AutoDL RTX4090)环境初始化
# 用法: ssh autodl "bash /root/autodl-tmp/draftrouter/scripts/setup_env.sh"
# 目标: 实例释放重建后,一条命令恢复全部环境。也是 P6 复现文档的底稿。
set -euo pipefail

# AutoDL 学术加速(GitHub/HuggingFace 拉取提速),失败不影响安装
source /etc/network_turbo 2>/dev/null || true

CONDA=/root/miniconda3/bin
PROJECT=/root/autodl-tmp/draftrouter

# 1. 独立 conda 环境(Python 3.10,匹配 vLLM V1 支持范围)
if [ ! -d /root/miniconda3/envs/draftrouter ]; then
  $CONDA/conda create -y -n draftrouter python=3.10
fi
PIP=/root/miniconda3/envs/draftrouter/bin/pip

# 2. PyTorch(CUDA 12.x)
$PIP install torch --index-url https://download.pytorch.org/whl/cu121

# 3. vLLM:先装官方 V1 版本跑通基线;P3 阶段再 fork 源码替换(见 docs/p3-design.md)
$PIP install vllm

# 4. 项目依赖
$PIP install fasttext transformers accelerate datasets

# 5. 模型权重下载到数据盘(只下载一次,关机不丢)
HF=/root/autodl-tmp/hf-cache
mkdir -p $HF
export HF_HOME=$HF
for m in Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B Qwen/Qwen3-8B; do
  $CONDA/../envs/draftrouter/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('$m')" || echo "WARN: $m 下载失败,重跑本脚本可续传"
done

# 6. 自检
/root/miniconda3/envs/draftrouter/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
$CONDA/../envs/draftrouter/bin/python -c "import vllm; print('vllm', vllm.__version__)"
echo "环境就绪。激活方式: conda activate draftrouter"
