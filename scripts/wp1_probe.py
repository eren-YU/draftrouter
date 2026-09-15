#!/usr/bin/env python
"""WP1 · 探针(不进矩阵,P0-action-plan.md §3 WP1)。

云端执行(约 15-25 分钟,三个引擎串行启动):
  ssh autodl "cd /root/autodl-tmp/draftrouter && export PATH=/root/miniconda3/envs/draftrouter/bin:\\$PATH \
    HF_HOME=/root/autodl-tmp/hf-cache HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 && \
    nohup /root/miniconda3/envs/draftrouter/bin/python scripts/wp1_probe.py > results/wp1.log 2>&1 &"

三项:
  1. 词表一致性:8B/1.7B/0.6B tokenizer 加载,断言 vocab 151936
  2. 显存探针:8B+0.6B、8B+1.7B(max_model_len=10240)各启动一次,
     从引擎日志提取 KV cache token 容量与 8192-token 请求的最大并发数、启动内存峰值
  3. LoRA 草稿头探针:1.7B+dummy LoRA(rank 8)独立引擎,记录相对 1.7B 裸载的
     KV token 余量差 → 换算到 8B target 的可保留 KV 余量(只做内存探针,不验证 spec 集成)

产物:results/p0-env/wp1.json;D1 冻结依据。
"""
import json
import os
import re
import subprocess
import sys

RESULTS_DIR = "results/p0-env"
os.makedirs(RESULTS_DIR, exist_ok=True)
PY = sys.executable
TARGET = "Qwen/Qwen3-8B"
report = {}

ENGINE_SNIPPET = r"""
import sys, vllm
model, spec = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
kwargs = {}
if spec == "0.6B":
    kwargs = {"speculative_config": {"method": "draft_model", "model": "Qwen/Qwen3-0.6B", "num_speculative_tokens": 1}}
elif spec == "1.7B":
    kwargs = {"speculative_config": {"method": "draft_model", "model": "Qwen/Qwen3-1.7B", "num_speculative_tokens": 1}}
elif spec == "lora":
    kwargs = {"enable_lora": True, "max_lora_rank": 8,
              "lora_modules": None}
llm = vllm.LLM(model=model, max_model_len=10240, gpu_memory_utilization=0.95,
               enable_prefix_caching=False, seed=0, **kwargs)
if spec == "lora":
    from vllm.lora.request import LoRARequest
    llm.generate(["hi"], vllm.SamplingParams(max_tokens=1),
                 lora_request=LoRARequest("dummy", 1, "/root/autodl-tmp/dummy_lora"))
import torch
print("PROBE_RESERVED_GIB", round(torch.cuda.memory_reserved()/2**30, 2))
"""


def run_engine(name, model, spec, log_path):
    """启动引擎子进程,从日志提取 KV 容量/并发/峰值显存。"""
    with open(log_path, "w") as log:
        p = subprocess.run([PY, "-c", ENGINE_SNIPPET, model, spec],
                           stdout=log, stderr=subprocess.STDOUT, timeout=1500)
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    kv = re.findall(r"KV cache size[:=]\s*([\d,]+)\s*tokens", txt)
    conc = re.findall(r"concurrency[:=]\s*([\d.]+)", txt, re.IGNORECASE)
    reserved = re.findall(r"PROBE_RESERVED_GIB\s+([\d.]+)", txt)
    return {
        "returncode": p.returncode,
        "kv_cache_tokens": int(kv[-1].replace(",", "")) if kv else None,
        "max_concurrency_8k": float(conc[-1]) if conc else None,
        "reserved_gib": float(reserved[-1]) if reserved else None,
        "log": log_path,
    }


def check_vocab():
    from transformers import AutoTokenizer
    out = {}
    for m in ("Qwen/Qwen3-8B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"):
        tok = AutoTokenizer.from_pretrained(m)
        n = len(tok)
        assert n == 151936, f"{m} vocab={n} != 151936"
        out[m] = n
    report["vocab"] = {"status": "pass", **out}
    print("[1/3] 词表一致性 PASS:", out)


def make_dummy_lora():
    """rank-8 dummy LoRA adapter(q/k/vproj),存盘供引擎加载。"""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16, device_map="cpu")
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                     target_modules=["q_proj", "k_proj", "v_proj"])
    get_peft_model(model, cfg)
    model.save_pretrained("/root/autodl-tmp/dummy_lora")
    del model
    print("dummy LoRA 已生成")


def main():
    try:
        check_vocab()
    except Exception as e:
        report["vocab"] = {"status": "fail", "error": repr(e)}
        print("[1/3] 词表 FAIL:", repr(e))

    print("[2/3] 显存探针:8B+0.6B ...")
    report["mem_8b_06b"] = run_engine("8b+0.6B", TARGET, "0.6B",
                                      f"{RESULTS_DIR}/wp1-engine-8b06b.log")
    print("   ", report["mem_8b_06b"])
    print("[2/3] 显存探针:8B+1.7B ...")
    report["mem_8b_17b"] = run_engine("8b+1.7B", TARGET, "1.7B",
                                      f"{RESULTS_DIR}/wp1-engine-8b17b.log")
    print("   ", report["mem_8b_17b"])

    print("[3/3] LoRA 探针 ...")
    try:
        make_dummy_lora()
        base = run_engine("1.7B", "Qwen/Qwen3-1.7B", None,
                          f"{RESULTS_DIR}/wp1-engine-17b.log")
        lora = run_engine("1.7B+LoRA", "Qwen/Qwen3-1.7B", "lora",
                          f"{RESULTS_DIR}/wp1-engine-17blora.log")
        report["lora"] = {"base_17b": base, "lora_17b": lora}
        if base["kv_cache_tokens"] and lora["kv_cache_tokens"]:
            delta_tok = base["kv_cache_tokens"] - lora["kv_cache_tokens"]
            report["lora"]["kv_token_delta_17b"] = delta_tok
            b17 = report["mem_8b_17b"]["kv_cache_tokens"]
            if b17:
                report["lora"]["est_8b_target_kv_remaining"] = max(0, b17 - delta_tok)
                report["lora"]["est_8k_concurrency"] = round(
                    max(0, b17 - delta_tok) / 8192, 2)
        print("   ", report["lora"])
    except Exception as e:
        report["lora"] = {"status": "fail", "error": repr(e)}
        print("[3/3] LoRA FAIL:", repr(e))

    with open(f"{RESULTS_DIR}/wp1.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("WP1 完成 →", f"{RESULTS_DIR}/wp1.json")


if __name__ == "__main__":
    main()
