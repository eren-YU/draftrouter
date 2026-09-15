# P0 · 立项与基线协议

> 状态：待人工确认 —— 协议已冻结（2026-09-15），与 `docs/P0-action-plan.md` §0 同口径；确认后开工。
> 项目：DraftRouter —— 领域自适应草稿路由（speculative decoding）
> 锁定日期：2026-09-15

## 0. 命题

1. **固定草稿跨领域加速不稳定**：单个固定 draft model 在中文问答 / 代码 / RAG 三类请求上的接受率差异，在共同长度桶或长度配对后仍显著，一组参数无法覆盖三类请求。
2. **复制型长上下文场景可由零成本草稿捕获**：引用型 RAG 输出复用输入片段，prompt-lookup 在这类请求上提供近零成本草稿来源；必须通过 `C_RAGlong vs A_RAGlong` 与 `verbatim vs non-verbatim` 两个对照。

## 0.1 边界（明确不做）

- 不 fork vLLM 引擎主干（上游月更，fork 会被持续吞掉）。
- 不重复实现上游已有能力：prompt-lookup(n-gram)、按 batch size 排 K 的投机长度、自适应验证预算、spec-decode 统计。
- 不做引擎级 Block 分配改造（draft token 的 slot 分配与回滚是上游现有行为）。
- P0 不测 Qwen3-4B 草稿;4B 草稿路线已排除(见下条显存核算),P3 显存路线相应只剩 1.7B 基座 + LoRA 与 0.6B+1.7B 常驻。

## 1. 目标模型与草稿池

- target：`Qwen3-8B`（vocab 151936，fp16，不量化）。
- P0 主线草稿池：`Qwen3-0.6B` / `Qwen3-1.7B`（同族同词表，vLLM 词表校验可直接通过）。
- 4B 不进 P0：8B+4B 裸权重合计约 22.75GiB，`gpu_memory_utilization=0.95` 下只剩约 0.05GiB，无法容纳 activation / draft KV / CUDA graph。
- 跨词表组合（Qwen2.5 系 151936 vs 152064）仅作 P5 消融，不进主线；上游 `vllm/v1/spec_decode/vocab_mapping.py` 支持跨词表映射，代价是接受率。
- 评测统一关闭 thinking：chat completions 用 `chat_template_kwargs={"enable_thinking": false}`，completions 用 `/no_think`；WP0 冒烟断言输出无 ` thinking`。

## 2. 基线三档

| 档 | 配置 | 作用 |
| --- | --- | --- |
| A · plain | 无投机 | 判断瓶颈是否存在、加速空间多大 |
| B · fixed-draft | 固定单草稿 + 固定 K | 主对比轴；0.6B/1.7B × K∈{3,5,7} 全网格 |
| C · ngram | prompt-lookup | 零成本草稿档；num_speculative_tokens{3,5,7} × prompt_lookup_max{3,5} 小网格 |

- B1：在 B 全网格中，取 3 主组合上 BS=1 输出 token/s 加速比中位数最大者；并列时取 3 主组合上 `num_accepted_tokens/num_draft_tokens` 方差（等权）更小者。
- B2：每个主组合的 B 网格最优配置，作为分场景 oracle，仅作脚注参照。
- C_best：全局中位数最优配置（并列时同 B1 取方差更小者）；同时记录每个主组合的 C oracle。
- 固定草稿必须扫过 K 后再定，避免对标未调优的稻草人。

## 3. 指标口径（只用 vLLM 原生，不新增名词）

- 执行方式：离线 batch `LLM.generate`，BS∈{1,4,8}；在线并发压测留到 P5，两类数字分开报。
- 延迟 / 吞吐：每请求从 `RequestOutput.metrics` 提取 TTFT、TPOT、e2e、输出 token/s；报告 median 与 p95；主口径与门槛用 fixed512。
- 接受相关：`vllm/v1/spec_decode/metrics.py` 的 `SpecDecodingStats` —— `num_drafts` / `num_draft_tokens` / `num_accepted_tokens` / `num_accepted_tokens_per_pos`，以及日志中的 draft / accepted throughput。
- 接受率分母固定为 `num_draft_tokens`。
- 每个结果 JSON 必须记录：配置、seed、数据集与长度档、样本 ID、样本数、vLLM 版本/commit、镜像 digest、GPU/驱动、完整命令、BS、achieved running batch、preemption、prefix cache 开关、thinking 开关、`output_mode`、wall time、实例运行时长。
- 卡时口径：卡时 = 云实例运行小时，不是 GPU 活跃小时。

## 4. 采样

- 主口径：`temperature=1.0, top_p=0.95, seed=0`。
- 附加对照：greedy（exploratory，不作为无损性证明）。
- 无损性验证必须建立在采样口径上；P0 做固定 seed 的 A/B1/C_best 逐 token 一致性冒烟，P5 再做完整分布一致性比对。

## 5. 输入协议（分层长度）

- 中文场景、代码场景：**原生长度**，不合成、不拉伸，如实报告长度分布。
- RAG 场景：short = ALCE 原生样本；long = 同一批 ALCE 样本按原始检索段落顺序扩到 4-8K，主判档取 [7500,8192]。
- **3 主组合 = 中文原生 / 代码原生 / RAG long[7500,8192]**。
- `max_model_len=10240`（8192 输入 + 2048 输出余量）；fixed512 档请求实际最多用 8704；natural-stop 档 `max_tokens=2048`。主矩阵只纳入 tokenizer 长度 ≤8192 的样本，超长样本在 WP2 剔除并记录。
- 长度分层：中文/代码按自然输入长度三分位桶；G5 必须在共同长度桶内或 RAG short/long 配对后成立，raw 跨场景差异只作描述。
- 矩阵规模：tuning 45 组 + report 52 组 = 97 组；WP4 复用 30 组，WP5 新增 67 组；G6 另有 1 个 non-verbatim 控制组，不计入 97 矩阵。

## 6. 输出协议

- 主口径 fixed512：`ignore_eos=True, max_tokens=512`；用于跨组合可比与所有门槛。
- 次要 natural-stop：`ignore_eos=False, max_tokens=2048`；用于真实形态对照。
- 加长输出：先给答案 → 2-4 句理由 →（RAG）逐字引用原文片段。
- 中文场景采用结构化抽取：输入中文长文档，输出固定四字段 JSON（公司名 / 金额 / 时间 / 产品名），字段值必须逐字取自原文。
- 输出形态对照：
  - 同一批中文篇章分别以 JSON 与自然段落输出；
  - 同一批 RAG long 样本分别以 verbatim（逐字引用）与 non-verbatim（禁止逐字引用，同长度改写/理由输出）输出。
- 报告中必须注明：这是面向解码吞吐的输出协议，不是原始基准的准确率评测。

## 7. 数据集

| 场景 | 主数据集 | 对照 / 备注 |
| --- | --- | --- |
| 中文 | 自建结构化抽取协议（`PaddlePaddle/dureader_robust` 篇章 + 四字段 JSON） | `ceval/ceval-exam` 仅作短答案对照（NC+SA） |
| 代码 | `tianyang/repobench-c`（python/java 配置，CC-BY-NC-ND-4.0；勿用 `tianyang/repobench`） | `openai/openai_humaneval`（MIT）降为正确性 / 无损验证集 |
| RAG | `princeton-nlp/ALCE-data`（引用型，short=原生，long=扩到 4-8K） | `hotpotqa/hotpot_qa`（CC-BY-SA-4.0）作短答案对照 |

- 四字段数据侧预筛：保留至少含 3 类字段（公司名/金额/时间/产品名）的篇章；在 data-valid 200 上统计每类字段出现率，每类 ≥60% 才进入模型侧验证。60% 沿用 §11 的建议阈值，P0 冻结为硬门槛，不启用事后校准。
- 模型侧：在字段出现的文档子集上统计非空率 ≥60%；非空字段 100% 逐字取自原文。
- D2 fallback：DuReader 链路不通或四字段不达标 → 首选 `THUDM/LongBench` 的 `dureader` 子集（中文长文，Apache-2.0），config/split/revision 在链路探测时记录实际值并写入数据清单；若该子集不可达或筛查仍不达标，P0 数据协议失败，停止全量。
- 备选池（仅作 P5/P6 参考）：`zai-org/LongBench-v2`、`xanhho/2WikiMultihopQA`、`dgslibisey/MuSiQue`、`b-mc2/sql-create-context`、`Salesforce/xlam-function-calling-60k`、`tianyang/repobench_raw_v1.1`。
- 许可风险（P6 需处理）：C-Eval 为 NC+SA；RepoBench（repobench-c）为 NC+ND；ALCE-data 无许可证标签且含 ELI5/KILT 等衍生数据；RAG 输出含逐字引用，开源 results JSON 是否构成 HotpotQA（CC-BY-SA-4.0）的再分发、署名与同方式共享义务，P6 前给出明确口径。

## 8. 前提盘点（实测）

- 本地：Python 3.12.8，**无 NVIDIA GPU**，未安装 vllm/torch；ssh 云链路已打通。
- 云 GPU：RTX4090 24G 单卡，平台 AutoDL，环境安装进行中；中途不换平台；项目总预算 **200 卡时天花板**。
- 云端路径：项目 `/root/autodl-tmp/draftrouter`，conda 环境 `draftrouter`，模型缓存 `/root/autodl-tmp/hf-cache`；非交互 ssh 用绝对 python 路径。
- vLLM：锁定 `0.29.0`（2026-09-09）；安装脚本必须 `vllm==0.29.0`；torch 版本由该 wheel 依赖解析后在 WP0 用 `pip freeze` 固定；镜像 digest、vLLM commit、CUDA/驱动写入 `results/p0-env/env.json`。
- 上游已有（本项目不重复实现）：prompt-lookup(n-gram)、`num_speculative_tokens_per_batch_size`、`enable_adaptive_verification`、spec-decode 统计（含逐位置接受分布）。
- 上游空白（本项目护城河）：**请求级草稿路由**、**多草稿模型共存与按请求切换**。
- 数据获取链路：HF 直连不可用，走 `HF_ENDPOINT=https://hf-mirror.com`，ModelScope 兜底；hf-mirror 可达但偶发并发限流。
- DuReader 取数链路（P0 必测）：HF 仓库只有加载脚本、无数据文件；datasets≥3 已不支持脚本加载；需实测脚本指向的百度 BOS URL 存活且从 AutoDL 可达。
- 离线 spec 统计（P0 必测）：WP0 用 5 条 prompt 证明 `SpecDecodingStats` 四字段可提取；若离线拿不到 per-position 统计，停止并升级协议修订，不静默切 server。
- AutoDL 公共数据目录（`/root/autodl-pub`）无本项目数据集与 Qwen 权重，不作数据来源。

## 9. 瓶颈复现实验（P0 必做）

- 配置：A 档 × RAG paired{ALCE 原生 short, ALCE 4-8K long[7500,8192]} × BS{1,8}，fixed512，N=20。
- 记录每请求 TTFT 与 TPOT，报告 median 与 p95。
- 判据：median TPOT_long/TPOT_short ≥1.3 或 median TTFT_long/TTFT_short ≥3.0 → 认定瓶颈存在，并写明是哪一种。
- 不允许把中文/代码合成到 8K；8K 档只指 RAG long。

## 10. 执行方式与矩阵

- 离线 batch，BS∈{1,4,8}；在线并发压测留到 P5，两类数字分开报。
- BS = 离线 batch size，不是客户端并发；结果 JSON 同时记录命名 BS、achieved running batch、preemption；名义 BS 与 achieved BS 分开报。
- P0 主线草稿池：0.6B / 1.7B；4B 不进 P0 矩阵。
- 数据池：tuning 50 / report 50 / data-valid 200 / router-train 独立池；四池样本 ID 落盘、两两零交集。
- 矩阵：
  - tuning 45 = A 3 + B 24（3 主组合 18 + RAG short 配对 6）+ C 18；
  - report 52 = fixed512 主矩阵 36 + natural-stop 12 + 输出形态新增 4（JSON/verbatim 复用主矩阵）；
  - 合计 97；WP4 复用 30 组，WP5 新增 67 组；G6 另有 1 个 non-verbatim 控制组。
- 配置：`enable_prefix_caching=False`、`max_model_len=10240`、thinking 关闭、`temperature=1.0, top_p=0.95, seed=0`。
- 两阶段：smoke → 全量。

## 11. smoke 通过门槛

| 门槛 | 配置 | 组数 | 通过判据 | 失败分支 |
| --- | --- | --- | --- | --- |
| G1 管道贯通 | A / B_prov(0.6B,K=5) / C_prov(K=5,lookup_max=5) × 3 主组合 × BS{1,8}，fixed512 | 3×3×2=18 | 全部跑完，无 OOM/崩溃；记录 achieved batch 与 preemption | 修 runner/显存；WP4 时限内仍不全绿 → P0 降级“未实测立项” |
| G2 瓶颈复现 | A × RAG paired{ALCE 原生 short, ALCE 4-8K long} × BS{1,8}，fixed512 | 1×2×2=4 | median TPOT_long/TPOT_short ≥1.3 或 median TTFT_long/TTFT_short ≥3.0 | 停止全量，记录 bottleneck not reproduced |
| G3 量级检查 | A + B1（G5 全网格选出）× 3 主组合 × BS=1，fixed512，tuning N=50 | A 3 组新增；B1 3 组复用 G5 | median 输出 token/s 加速比 ≥1.3×，且无组合 <1.0×；G3 只作 in-sample 量级门，最终结论以 report set 复现 | 一次草稿/K 调参重测；仍失败 → 停止，记录 fixed-draft magnitude not met |
| G4 数据协议 | data-valid 200 数据侧覆盖 + 模型侧中文 JSON 200 条 | 1 组模型侧 | 链路通过；每类字段在数据侧出现率 ≥60%；在字段出现的文档上，模型输出该字段非空率 ≥60%；非空字段 100% 逐字取自原文 | D2 换源；仍失败 → P0 数据协议失败 |
| G5 命题1：固定草稿跨域异质性 | B 全网格（0.6B/1.7B × K{3,5,7}）× 3 主组合 × BS=1 + B 全网格 × RAG short × BS=1，fixed512，tuning N=50 | 24 | 在至少一个共同长度桶内（或同一 ALCE 样本的 RAG short/long 配对后），存在场景对使最优草稿排名翻转且两草稿接受率 bootstrap 95% CI 不重叠；同时报告 raw 跨场景异质性 | 不存在任何场景对同时满足上述条件，或没有共同长度桶/配对条件 → 命题1未获支持；停止 P1/P2 并重审定位 |
| G6 命题2：ngram 复制优势 | C_prov(K=5,lookup_max=5) × RAG long{verbatim,non-verbatim} × BS=1 + C_prov × 中文/代码 × BS=1，fixed512，tuning N=50；A × RAG long 复用 G3 | 新增 3 个矩阵组 + 1 个 non-verbatim 控制组（另复用 G3 的 A×RAG long 1 组） | (a) C_RAGlong vs A_RAGlong 的 paired 输出 token/s CI 下界 > 0；(b) verbatim > non-verbatim 的 paired 接受率 CI 下界 > 0 | 任一不满足 → 命题2未获支持；停止 P4 的 RAG 专项声明并重审定位 |

- 统计口径：长度桶边界取 tuning+report 池上三场景自然长度的全局三分位（一次计算后冻结）；RAG 配对按同一 ALCE 样本 ID 的 short/long 成对；bootstrap 为 paired（同一重采样索引作用于两配置）、重采样 10000 次、seed=0。
- D3 总决策：G1-G6 全绿 → 进 WP5；G1 失败 → 修 runner/显存，WP4 时限内仍不全绿则 P0 降级；G2 失败 → 停止全量并记录 bottleneck not reproduced；G3 失败 → 一次调参重测，仍失败则停止并记录 fixed-draft magnitude not met；G4 失败 → D2 换源，仍失败则 P0 数据协议失败；G5 失败 → 停止 P1/P2 并重审定位；G6 失败 → 停止 P4 的 RAG 专项声明并重审定位。
- 时限：WP4 从开工起 ≤3 天；超时则 P0 降级“未实测立项”并如实标注。

## 12. 产物与落盘

- `results/p0-env/env.json`：环境、版本、镜像 digest、GPU/驱动、权重校验（云端）。
- `results/p0-smoke/*.json`、`results/p0/*.json`：原始结果（云端）。
- `results/baseline/summary.json`：汇总（唯一入库的结果文件）。
- `scripts/bench/`：离线 runner、指标提取、日志解析、本地单测。
- `scripts/data_prep/`：确定性样本构造、四池 ID、长度统计。
- `docs/p0-baseline-report.md`：基线报告（入库）。
- P0 期间 `docs/` 入库，作为本地唯一事实源的一部分；P6 再决定公开净化版。
- `README.md`：只维护结果表。

## 13. 卡时预算表 v2（待 smoke 实测标定）

| 阶段 | 预算（卡时） | 实测 | 备注 |
| --- | --- | --- | --- |
| P0 基线 | 18-32（分项严格和 17.5-31.5，向上取整） | | smoke 前粗估；WP4 后回填 |
| P1 基建 | 待标定 | | |
| P2 路由与草稿池 | 待标定 | | |
| P3 引擎集成 | 待标定 | | 不 fork 路线 |
| P4 RAG 专项 + LoRA | 待标定 | | |
| P5 评测与无损 | 待标定 | | |
| P6 开源 | 待标定 | | |
| P7 复盘 | 待标定 | | |

- 卡时口径 = 云实例运行小时；实例用完即关，下载/构样也计入。
- 工作假设：80-120 卡时；上限 200 卡时。

## 14. 待填实测值

- [ ] 三个模型（8B/1.7B/0.6B）词表一致性实测（151936）
- [ ] DuReader 取数链路实测（BOS URL 存活 + AutoDL 可达 + datasets≥3 绕行方案）
- [ ] D2 fallback（THUDM/LongBench `dureader`）可用性实测
- [ ] 四字段数据侧覆盖率与模型侧非空率/逐字率
- [ ] 输出形态对照结果（中文 JSON vs 自然段落；RAG verbatim vs non-verbatim）
- [ ] 瓶颈复现：RAG short vs long 的 TTFT / TPOT
- [ ] 三档接受率与 TPOT（BS 1/4/8）
- [ ] B 全网格 K 扫描（0.6B/1.7B × K=3/5/7）
- [ ] ngram 参数网格（num_speculative_tokens × prompt_lookup_max）
- [ ] LoRA 草稿头显存探针（决定多草稿显存路线 b / a）
- [ ] 离线 `SpecDecodingStats` 四字段可提取性
- [ ] thinking 关闭冒烟
- [ ] 单组挂钟时间（用于回填 §13 预算表 v2）
- [ ] 各数据集实际长度分布
- [ ] GPU 启动内存峰值、achieved running batch、preemption 计数
