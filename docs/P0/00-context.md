# P0 共享上下文(每个任务会话开工前必读)

> 口径依据:`docs/P0-action-plan.md`(§0 执行口径已冻结)与 `docs/P0-baseline.md`。
> 本文件是各任务会话的交接上下文;与 action-plan 冲突处,以 action-plan 为准。
> 更新:2026-09-15(WP0 完成、WP1 脚本就绪未跑、WP2 数据接线修复中)

## 环境与硬约束

- 本地 `D:\code\draftrouter`:唯一事实源,编辑/单测/commit 在这里;**push 前必须人工确认**。
  同步云端:`scp -r scripts/* autodl:/root/autodl-tmp/draftrouter/scripts/`(git push 未授权前走 scp)。
- 云端:`ssh autodl`,项目 `/root/autodl-tmp/draftrouter`,数据 `/root/autodl-tmp/data`,
  python 用全路径 `/root/miniconda3/envs/draftrouter/bin/python`;超 5 分钟任务必须 nohup 后台。
- **云端运行 vLLM 的环境变量四件套**(缺一必挂,踩坑实录):
  ```bash
  export PATH=/root/miniconda3/envs/draftrouter/bin:$PATH   # EngineCore 子进程要找到 ninja
  export HF_HOME=/root/autodl-tmp/hf-cache                  # 否则找不到模型缓存
  export HF_HUB_OFFLINE=1                                   # 离线读缓存
  export VLLM_USE_FLASHINFER_SAMPLER=0                      # 宿主 nvcc 仅 11.8,flashinfer JIT 必失败
  ```
  `scripts/bench/config.py` 与 `scripts/wp0_env_check.py` 已内置 setdefault 兜底,直接跑 python 脚本可不设最后一个。
- 下载才开 `source /etc/network_turbo`(仅对 GitHub/HF 有效);pip 一律不开。
- 卡时记账:结果 JSON 记实例运行时长;对照 200 卡时天花板。

## 已实测的 vLLM 0.29 关键事实(WP0 冒烟确认,勿再试探)

- spec 配置走 `speculative_config` dict:`{"method": "draft_model"|"ngram", "model": <draft>, "num_speculative_tokens": K, "prompt_lookup_max"/"prompt_lookup_min": L}`;顶层 `speculative_model` 参数已不存在(TypeError)。
- 离线 spec 统计(C10 已打通):引擎构造加 `per_request_spec_decode_metrics="detailed"`,然后
  `RequestOutput.outputs[0].spec_decode_metrics` = {histogram(len=K+1), num_draft_tokens, per_step_accepted, per_step_drafted};
  聚合版在 `RequestOutput.metrics.speculative_decoding`。四字段映射见 `scripts/bench/spec_stats.py::spec_stats_from_request_output`。
- thinking 关闭:`llm.chat(..., chat_template_kwargs={"enable_thinking": False})` 实测可用,输出无 `<think>`。
- Qwen3 权重已校验:8B=15.26GiB/5 shards/399 tensors;vocab 三模型均 151936(待 WP1 正式断言)。
- 数据链路存活(2026-09-15 实测):DuReader BOS `https://bj.bcebos.com/paddlenlp/datasets/dureader_robust-data.tar.gz`(20.5MB)、
  ALCE `.../ALCE-data/resolve/main/ALCE-data.tar`(451MB,已下载 sha256=eda837bf...)、LongBench fallback `.../THUDM/LongBench/resolve/main/data.zip`(114MB)。
- repobench-c HF 仓只有 loader 脚本,新版 datasets 不能加载;必须走 parquet 分支
  `.../tianyang/repobench-c/resolve/refs%2Fconvert%2Fparquet/{python_cff|java_cff}/test/0.parquet`。

## 当前进度(2026-09-15 收盘)

| 工作包 | 状态 |
| --- | --- |
| WP0 环境确认 | ✅ 全绿:`results/p0-env/env.json`(版本/权重/spec四字段/thinking) |
| WP1 探针 | 脚本就绪 `scripts/wp1_probe.py`(已上云),**未执行** |
| WP2 数据 | 下载✅(manifest 含 sha256);数据源接线修复进行中(见 T2) |
| WP3 bench 脚手架 | ✅ 代码+33 单测绿;0.29 实测参数已回填;**reviewer 未审**(见 T3) |
| WP4 / WP5 / 报告 | 未开始(T4/T5/T6) |

## 通用验收标准(所有任务)

1. 云端长任务全部 nohup,结束后 `nvidia-smi` 确认 GPU 释放;
2. 结果 JSON 落 `results/`,关键结论数字回写 docs;
3. 涉及脚本的改动:本地 pytest 全绿后才算完成;
4. commit 即里程碑(本地),push 留给用户确认;
5. 完成后在本文"当前进度"表更新状态。
