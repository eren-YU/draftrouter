# P0 · 立项与基线协议

> 状态：P0 进行中 —— 协议已锁定，实测值待填（见 §14）
> 项目：DraftRouter —— 领域自适应草稿路由（speculative decoding）
> 锁定日期：2026-09-15

## 0. 命题

1. **固定草稿跨领域加速不稳定**（与上下文长度无关）：单个固定 draft model 在中文问答 / 代码 / RAG 三类请求上的接受率差异显著，一组参数无法覆盖三类请求。
2. **复制型长上下文场景可由零成本草稿捕获**（与长度相关）：引用型 RAG 输出大量复用输入片段，prompt-lookup (n-gram) 在这类请求上提供近零成本的草稿来源。

## 0.1 边界（明确不做）

- 不 fork vLLM 引擎主干（上游月更，fork 会被持续吞掉）
- 不重复实现上游已有能力：prompt-lookup(n-gram)、按 batch size 排 K 的投机长度、自适应验证预算、spec-decode 统计
- 不做引擎级 Block 分配改造（draft token 的 slot 分配与回滚是上游现有行为）

## 1. 目标模型与草稿池

- target：`Qwen3-8B`（vocab 151936，fp16，不量化）
- 草稿池：`Qwen3-0.6B` / `Qwen3-1.7B` / `Qwen3-4B`（同族同词表，vLLM 词表校验可直接通过）
- 跨词表组合（Qwen2.5 系 151936 vs 152064）仅作 P5 消融，不进主线；上游 `vllm/v1/spec_decode/vocab_mapping.py` 支持跨词表映射（草稿 logits 约束到交集、未命中回退 UNK），代价是接受率
- 评测统一关闭 thinking（`enable_thinking=false` / `/no_think`）

## 2. 基线三档

| 档 | 配置 | 作用 |
| --- | --- | --- |
| A · plain | 无投机 | 判断瓶颈是否存在、加速空间多大 |
| B · fixed-draft | 固定单草稿 + 固定 K | **主对比轴**；K ∈ {3,5,7} 扫描后取最优 |
| C · ngram | prompt-lookup | 零成本草稿档；投机长度 {3,5,7} × lookup 上限 {3,5} 小网格 |

- 主指标口径：B1 全局固定草稿 → 路由版
- B2 分场景 oracle 最优固定草稿：从 K 扫描数据免费派生，仅作脚注参照
- 固定草稿必须扫过 K 后再定，避免对标一个未调优的稻草人

## 3. 指标口径（只用 vLLM 原生，不新增名词）

- 延迟 / 吞吐：vLLM 原生 latency/throughput 指标（TPOT、TTFT、e2e、输出 token/s）
- 接受相关：`vllm/v1/spec_decode/metrics.py` 的 `SpecDecodingStats` —— `num_drafts` / `num_draft_tokens` / `num_accepted_tokens` / `num_accepted_tokens_per_pos`，以及日志中的 draft / accepted throughput
- 每个结果 JSON 必须记录：配置、seed、数据集与长度档、样本数、vLLM 版本、镜像 digest、GPU 型号、完整命令

## 4. 采样

- 主口径：`temperature=1.0, top_p=0.95, seed=0`
- 附加对照：greedy
- 无损性验证必须建立在采样口径上（greedy 下无损平凡成立）

## 5. 输入协议（分层长度）

- 中文场景、代码场景：**原生长度**，不合成、不拉伸，如实报告长度分布
- RAG 场景：保留 **4-8K 合成长档**（prompt-lookup 的命中空间随输入长度增长，这是唯一真随长度增长的机制）
- 长档构造：在原生检索段落基础上增加段落数量至目标 token 区间
- 矩阵规模：≈54-63 组（替代原「512/2K/4K/8K × 3 场景」全交叉的 108 组）

## 6. 输出协议

- 加长输出：先给答案 → 2-4 句理由 →（RAG）逐字引用原文片段
- 两档：`自然停止`（真实形态）与 `固定 512 / ignore_eos`（token 对齐，跨场景可比）
- 中文场景采用**结构化抽取**：输入中文长文档，输出固定四字段 JSON（公司名 / 金额 / 时间 / 产品名），字段值必须逐字取自原文
- 报告中必须注明：这是面向解码吞吐的输出协议，不是原始基准的准确率评测
- 输出形态对照（隔离混杂因素）：同一批中文篇章分别以 JSON 与自然段落两种形态输出并比较接受率，用于区分「领域效应」与「输出结构复用效应」——防止命题 1 的跨领域结论实际由输出格式驱动

## 7. 数据集

| 场景 | 主数据集 | 对照 / 备注 |
| --- | --- | --- |
| 中文 | 自建结构化抽取协议（`PaddlePaddle/dureader_robust` 篇章 + 四字段 JSON；HF 仓库仅加载脚本、无数据文件，取数链路见 §8） | `ceval/ceval-exam` 仅作短答案对照（NC+SA） |
| 代码 | `tianyang/repobench-c`（python/java 配置，CC-BY-NC-ND-4.0；勿用 `tianyang/repobench`——该 ID 实际解析到 repobench-r 检索变体） | `openai/openai_humaneval`（MIT）降为正确性 / 无损验证集 |
| RAG | `princeton-nlp/ALCE-data`（引用型，扩长到 4-8K；单 tar 包约 451MB，仓库无许可证标签） | `hotpotqa/hotpot_qa`（CC-BY-SA-4.0）作短答案对照 |

备选池：`THUDM/LongBench`、`zai-org/LongBench-v2`（apache-2.0）、`xanhho/2WikiMultihopQA`、`dgslibisey/MuSiQue`（两者为个人镜像，启用前需重验）、`b-mc2/sql-create-context`、`Salesforce/xlam-function-calling-60k`（gated=auto）、`tianyang/repobench_raw_v1.1`（CC-BY-4.0，RepoBench 的许可备选）。

许可风险（P6 需处理）：C-Eval 为 NC+SA，RepoBench（repobench-c）为 NC+ND；ALCE-data 无许可证标签且含 ELI5/KILT 等衍生数据，许可链不明；RAG 输出含逐字引用，开源 results JSON 是否构成 HotpotQA（CC-BY-SA-4.0）的再分发、署名与同方式共享义务，P6 前给出明确口径。

## 8. 前提盘点（实测）

- 本地：Python 3.12.8，**无 NVIDIA GPU**，未安装 vllm/torch；`~/.ssh/config` 不存在（云链路未打通）
- 云 GPU：RTX4090 24G 单卡，平台 AutoDL，**尚未租用**；中途不换平台；项目总预算 **200 卡时天花板**
- vLLM：最新稳定 `0.29.0`（2026-09-09）；`vllm/v1/spec_decode/` 已含 `ngram_proposer` / `ngram_proposer_gpu` / `suffix_decoding` / `medusa` / `eagle` / `custom_class_proposer` / `vocab_mapping` / `metrics`
- 上游已有（本项目不重复实现）：prompt-lookup(n-gram)、`num_speculative_tokens_per_batch_size`、`enable_adaptive_verification`、spec-decode 统计（含逐位置接受分布）
- 上游空白（本项目护城河）：**请求级草稿路由**、**多草稿模型共存与按请求切换**
- 数据获取链路：HF 直连不可用（元数据请求 308），走 `HF_ENDPOINT=https://hf-mirror.com`，ModelScope 兜底；2026-09-15 复测：HF 直连超时，hf-mirror 可达但偶发并发限流，ModelScope API 可达
- DuReader 取数链路（P0 必测）：`PaddlePaddle/dureader_robust` 的 HF 仓库只有加载脚本、无数据文件，datasets≥3 已不支持脚本加载；需实测脚本指向的百度 BOS URL 存活且从 AutoDL 可达，通过后方可定为唯一来源；ModelScope（仅 QG 变体）与 `dirtycomputer/dureader_robust-data`（仅 train、无许可证）均不完整，不达预期则改用备选池换血
- AutoDL 公共数据目录（`/root/autodl-pub`）全部为 CV / 视频 / 点云数据，无本项目任何数据集与 Qwen 权重，不作数据来源

## 9. 瓶颈复现实验（P0 必做）

同一批请求上同时记录 TTFT 与 TPOT，比较原生短档与 8K 长档，产出显式结论：「瓶颈在 X，量级 Y」。

判据：8K vs 短档 **TPOT 恶化 ≥1.3×** 或 **TTFT 恶化 ≥3×** → 认定瓶颈存在，并写明是哪一种。

## 10. 执行方式与矩阵

- 离线 batch，BS ∈ {1,4,8}；在线并发压测留到 P5，两类数字分开报
- 每场景每长度档 N=100，固定 seed 抽样：其中 50 条用于调参（K / ngram 参数），50 条为最终报告集
- **路由器训练样本与评测样本零重叠**（P2 前置约束）
- 两阶段：smoke → 全量

## 11. smoke 通过门槛

1. 链路通：三档 × 8K × BS{1,8} × 3 场景全部跑完，无 OOM、无崩溃
2. 瓶颈复现：8K vs 原生短档 TPOT 恶化 ≥1.3× 或 TTFT 恶化 ≥3×
3. 量级合理：BS=1 下 fixed-draft 相对 plain 加速比 ≥1.3×；低于则先修草稿选型或 K
4. 数据协议成立：DuReader 取数链路实测通过；四字段（公司名 / 金额 / 时间 / 产品名）在 ≥200 条抽检样本上的非空率达标（建议阈值 60%，可按实际分布校准），不达标则触发 §7 备选池换血

时限 ≤3 天；超时则 P0 降级为「未实测立项」并如实标注。

## 12. 产物与落盘

- `results/<phase>/*.json`：原始结果（配置 / seed / 命令 / vLLM 版本 / 镜像 digest）
- `results/baseline/summary.json`：汇总（唯一入库的结果文件）
- `scripts/bench/`：可复现脚本
- 本报告与规划文档留在 `docs/`，**不入库**
- README 只维护结果表

## 13. 卡时预算表 v2（待 smoke 实测标定）

| 阶段 | 预算（卡时） | 实测 | 备注 |
| --- | --- | --- | --- |
| P0 基线 | 待标定 | | smoke 后回填 |
| P1 基建 | 待标定 | | |
| P2 路由与草稿池 | 待标定 | | |
| P3 引擎集成 | 待标定 | | 不 fork 路线 |
| P4 RAG 专项 + LoRA | 待标定 | | |
| P5 评测与无损 | 待标定 | | |
| P6 开源 | 待标定 | | |
| P7 复盘 | 待标定 | | |

工作假设：80-120 卡时；上限 200 卡时。

## 14. 待填实测值

- [ ] 三个模型词表一致性实测（151936）
- [ ] DuReader 取数链路实测（BOS URL 存活 + AutoDL 可达 + datasets≥3 绕行方案）
- [ ] DuReader 四字段非空率抽检
- [ ] 输出形态对照结果（同输入 JSON vs 自然段落的接受率差）
- [ ] 瓶颈复现：TTFT / TPOT @ 原生短档 vs 8K
- [ ] 三档接受率与 TPOT（BS 1/4/8）
- [ ] K 扫描结果（K = 3/5/7）
- [ ] ngram 参数网格（投机长度 × lookup 上限）
- [ ] LoRA 草稿头探针（决定多草稿显存路线 b / a）
- [ ] 单组挂钟时间（用于回填 §13 预算表 v2）
- [ ] 各数据集实际长度分布
