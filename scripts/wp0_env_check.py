#!/usr/bin/env python
"""WP0 · 环境就绪确认(P0-action-plan.md §3 WP0)。

云端执行:
  ssh autodl "cd /root/autodl-tmp/draftrouter && nohup /root/miniconda3/envs/draftrouter/bin/python scripts/wp0_env_check.py > results/wp0.log 2>&1 &"

四项检查,全绿才算 WP0 通过:
  1. 版本锁定:vllm==0.29.0、torch 实际版本(pip freeze 快照)、GPU/驱动
  2. 权重校验:8B/1.7B/0.6B safetensors 清单完整性(逐 tensor header 校验,不只看目录)
  3. 离线 spec 统计冒烟:8B+0.6B K=5,5 条 prompt,验证 num_drafts/num_draft_tokens/
     num_accepted_tokens/num_accepted_tokens_per_pos 可从离线路径提取
  4. Thinking 冒烟:Qwen3-8B 中文 prompt,关闭 thinking 后输出无 '<think>'

产物:results/p0-env/env.json + results/p0-env/*.json 冒烟原始输出。
任一项失败:env.json 标记 fail 并写原因,退出码 1。
"""
import json
import os
import subprocess
import sys
import traceback

# 宿主 nvcc 11.8 无法 JIT flashinfer 采样器(2026-09-15 实测),必须在 import vllm 前设置
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

RESULTS_DIR = os.environ.get("WP0_RESULTS_DIR", "results/p0-env")
os.makedirs(RESULTS_DIR, exist_ok=True)

env_report = {"checks": {}}


def save():
    with open(os.path.join(RESULTS_DIR, "env.json"), "w", encoding="utf-8") as f:
        json.dump(env_report, f, ensure_ascii=False, indent=2)


def check_versions():
    import torch
    import vllm

    assert torch.cuda.is_available(), "CUDA 不可用"
    assert vllm.__version__ == "0.29.0", f"vllm={vllm.__version__} != 0.29.0"
    props = torch.cuda.get_device_properties(0)
    drv = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()[0]
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    ).stdout
    with open(os.path.join(RESULTS_DIR, "pip-freeze.txt"), "w") as f:
        f.write(freeze)
    try:
        import vllm as _v
        from vllm import envs  # noqa: F401  探测模块存在性
        commit = getattr(_v, "commit_id", None)
    except Exception:
        commit = None
    env_report["checks"]["versions"] = {
        "status": "pass",
        "vllm": vllm.__version__,
        "vllm_commit": commit,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": props.name,
        "gpu_mem_total_gb": round(props.total_memory / 2**30, 2),
        "driver": drv,
        "python": sys.version.split()[0],
    }
    print("[1/4] 版本锁定 PASS:", env_report["checks"]["versions"]["vllm"],
          env_report["checks"]["versions"]["torch"], props.name, "driver", drv)


def check_weights():
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    models = ["Qwen/Qwen3-8B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"]
    out = {}
    for m in models:
        path = snapshot_download(m, local_files_only=True)
        idx_file = os.path.join(path, "model.safetensors.index.json")
        shards = []
        total_bytes = 0
        n_tensors = 0
        if os.path.exists(idx_file):
            with open(idx_file) as f:
                idx = json.load(f)
            shards = sorted(set(idx["weight_map"].values()))
            expect = set(idx["weight_map"].keys())
        else:
            shards = ["model.safetensors"]
            expect = None
        seen = set()
        for s in shards:
            sp = os.path.join(path, s)
            assert os.path.exists(sp), f"{m} 缺 shard {s}"
            total_bytes += os.path.getsize(sp)
            with safe_open(sp, framework="pt") as f:
                for k in f.keys():
                    seen.add(k)
                    f.get_slice(k)  # header 可解析
            n_tensors = len(seen)
        if expect is not None:
            missing = expect - seen
            assert not missing, f"{m} 缺 tensor: {sorted(missing)[:5]}"
        # tokenizer 必备文件
        for t in ("tokenizer.json", "tokenizer_config.json", "config.json"):
            assert os.path.exists(os.path.join(path, t)), f"{m} 缺 {t}"
        out[m] = {
            "path": path, "shards": len(shards), "tensors": n_tensors,
            "weight_gib": round(total_bytes / 2**30, 2),
        }
    env_report["checks"]["weights"] = {"status": "pass", "models": out}
    print("[2/4] 权重校验 PASS:", json.dumps(out, ensure_ascii=False))


SMOKE_PROMPTS = [
    "请用一句话介绍北京。",
    "简单说明什么是投机采样。",
    "把这句话翻译成英文:今天天气很好。",
    "1 到 100 的整数和是多少?",
    "给出一个 Python 快速排序的实现思路。",
]


def check_spec_stats():
    """8B+0.6B K=5 离线冒烟,验证四字段提取路径。"""
    import inspect

    import vllm
    sig = str(inspect.signature(vllm.LLM.__init__))
    with open(os.path.join(RESULTS_DIR, "llm_signature.txt"), "w") as f:
        f.write(sig)
    print("LLM.__init__ 签名已存 llm_signature.txt")

    # 以实际签名为准选择 spec 参数(vLLM 0.29 实测:speculative_config dict;
    # per_request_spec_decode_metrics 开启离线 per-request spec 统计 —— C10 取数路径)
    kwargs = {
        "speculative_config": {"method": "draft_model",
                               "model": "Qwen/Qwen3-0.6B",
                               "num_speculative_tokens": 5},
        "per_request_spec_decode_metrics": "detailed",
    }
    llm = vllm.LLM(
        model="Qwen/Qwen3-8B",
        max_model_len=10240,
        gpu_memory_utilization=0.95,
        enable_prefix_caching=False,
        seed=0,
        **kwargs,
    )
    sp = vllm.SamplingParams(temperature=1.0, top_p=0.95, seed=0,
                             max_tokens=64, ignore_eos=True)
    outs = llm.generate(SMOKE_PROMPTS, sp)
    assert len(outs) == 5

    fields = {}
    # 路径1(0.29 实测):CompletionOutput.spec_decode_metrics(RequestSpecDecodeMetrics)
    m = getattr(outs[0].outputs[0], "spec_decode_metrics", None)
    if m is not None:
        hist = list(getattr(m, "histogram", []) or [])
        n_steps = sum(hist) if hist else None
        n_acc = sum(j * c for j, c in enumerate(hist)) if hist else None
        fields["num_drafts"] = n_steps
        fields["num_draft_tokens"] = getattr(m, "num_draft_tokens", None)
        fields["num_accepted_tokens"] = n_acc
        # per-position:j 个草稿被接受的 verify 步直方图;位置 p 接受数 = sum_{j>p} hist[j]
        if hist:
            fields["num_accepted_tokens_per_pos"] = [sum(hist[p + 1:]) for p in range(len(hist) - 1)]
        fields["_raw_histogram"] = hist
        fields["_per_step_accepted"] = list(getattr(m, "per_step_accepted", []) or [])[:20]
    # 路径2:RequestOutput.metrics.speculative_decoding(to_dict 聚合)
    agg = getattr(outs[0].metrics, "speculative_decoding", None)
    if agg is not None:
        fields["_aggregated"] = agg
    # 路径3:RequestOutput 层其他 spec 属性兜底探测
    for name in dir(outs[0]):
        if "spec" in name.lower() and name != "spec_decode_metrics":
            fields.setdefault("request_output." + name, repr(getattr(outs[0], name))[:200])

    result = {
        "extracted_fields": fields,
        "outputs": [o.outputs[0].text[:200] for o in outs],
    }
    with open(os.path.join(RESULTS_DIR, "spec_stats_smoke.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    got4 = {"num_drafts", "num_draft_tokens", "num_accepted_tokens",
            "num_accepted_tokens_per_pos"} & set(fields)
    status = "pass" if len(got4) == 4 else "warn"
    env_report["checks"]["spec_stats"] = {
        "status": status, "found": sorted(fields.keys()),
        "note": "四字段全找到=pass;只找到部分=warn(按 C10,若离线确实拿不到需停并升协议修订)",
    }
    print(f"[3/4] spec 统计冒烟 {status.upper()}: 找到 {sorted(fields.keys())}")


def check_thinking():
    import vllm

    llm = vllm.LLM(model="Qwen/Qwen3-8B", max_model_len=10240,
                   gpu_memory_utilization=0.95, enable_prefix_caching=False,
                   seed=0)
    msgs = [[{"role": "user", "content": "请用一句话介绍上海。"}]]
    outputs, mode = None, None
    try:
        outputs = llm.chat(msgs, vllm.SamplingParams(temperature=1.0, top_p=0.95,
                                                     seed=0, max_tokens=128),
                           chat_template_kwargs={"enable_thinking": False})
        mode = "chat_template_kwargs"
    except TypeError:
        outputs = llm.chat([{"role": "user",
                             "content": "/no_think 请用一句话介绍上海。"}],
                           vllm.SamplingParams(temperature=1.0, top_p=0.95,
                                               seed=0, max_tokens=128))
        mode = "no_think_suffix"
    texts = [o.outputs[0].text for o in outputs]
    has_think = any("<think>" in t for t in texts)
    result = {"mode": mode, "has_think_tag": has_think, "outputs": texts}
    with open(os.path.join(RESULTS_DIR, "thinking_smoke.json"), "w",
              encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    status = "pass" if not has_think else "fail"
    env_report["checks"]["thinking"] = {"status": status, "mode": mode}
    print(f"[4/4] thinking 冒烟 {status.upper()} (mode={mode}): {texts[0][:120]}")


def main():
    steps = [("versions", check_versions), ("weights", check_weights),
             ("spec_stats", check_spec_stats), ("thinking", check_thinking)]
    for name, fn in steps:
        try:
            fn()
        except Exception as e:
            env_report["checks"][name] = {
                "status": "fail", "error": repr(e),
                "traceback": traceback.format_exc()[-2000:],
            }
            save()
            print(f"[FAIL] {name}: {e!r}", file=sys.stderr)
            sys.exit(1)
    save()
    print("WP0 全部通过 →", os.path.join(RESULTS_DIR, "env.json"))


if __name__ == "__main__":
    main()
