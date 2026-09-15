# DraftRouter

Domain-adaptive draft routing for speculative decoding — a same-family draft pool + request-level router on vLLM V1, plus prompt-lookup for copy-heavy RAG workloads.

## Status

P0（立项与基线）— 协议已锁定，基线实测进行中。结果表见下。

## 结果表（待填）

| 配置 | BS | 场景 | 输入档 | TPOT | 吞吐 | 接受数 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A · plain | 1 | — | — | 待测 | 待测 | — | 无投机基线 |
| B · fixed-draft | 1 | — | — | 待测 | 待测 | 待测 | 主对比基线 |
| C · ngram | 1 | — | — | 待测 | 待测 | 待测 | 零成本草稿档 |

## 环境

- vLLM `0.29.0`（锁 commit + 镜像 digest）
- target `Qwen3-8B`，草稿池 `Qwen3-0.6B / Qwen3-1.7B / Qwen3-4B`（同族同词表 151936）
- 单卡 RTX 4090 24G

## 仓库结构（规划）

- `scripts/bench/` 评测脚本（可复现）
- `results/baseline/summary.json` 基线汇总（原始明细不入库）
- `docs/` 本地协议与规划文档（不入库）
