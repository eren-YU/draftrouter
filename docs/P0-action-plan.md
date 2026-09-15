# P0 · 行动方案

> 依据：`docs/P0-baseline.md`（协议，已锁定）。本文件是 P0 的唯一执行口径；与 `P0-baseline.md` 正文冲突处，以本文件 §0 为准。
> 状态：待人工确认 —— §0 执行口径已冻结；确认后按 §3 顺序开工。
> 日期：2026-09-15

## 0. 执行口径冻结

| # | 项目 | P0 最终口径 | 依据 / 理由 |
| --- | --- | --- | --- |
| C1 | 执行方式 | 离线 batch：`LLM.generate`，BS∈{1,4,8}；`vllm serve` 在线并发压测留到 P5 | baseline §10；两类数字不混报 |
| C2 | BS 定义 | BS = 离线 batch size，不是客户端并发；结果 JSON 同时记录命名 BS、achieved running batch、preemption 次数 | 防止“名义 BS=8、实际 running batch=2”被当成通过 |
| C3 | 长度档 | 中文={原生}，代码={原生}，RAG={short=ALCE 原生，long=同一批 ALCE 扩到 4-8K，主判档取 [7500,8192]}；中文/代码不做 8K 合成 | baseline §5；同一批 ALCE 做 short/long 配对，避免数据集混淆 |
| C4 | 长度无关性检验 | 每个场景按自然输入长度分三分位桶；G5 必须在共同长度桶内或长度配对后成立，raw 跨场景差异只作描述 | 否则“域效应”与“长度效应”无法分离 |
| C5 | max_model_len | 10240（8192 输入 + 2048 输出余量）；fixed512 档请求实际最多用 8704；natural-stop 档 `max_tokens=2048`；主矩阵只纳入 tokenizer 长度 ≤8192 的样本，超长样本在 WP2 剔除并记录 | 8704 会让 8K 档 natural-stop 退化成 fixed512；超长样本会挤占输出余量 |
| C6 | 输出两档 | fixed512：`ignore_eos=True, max_tokens=512`，主口径/门槛；natural-stop：`ignore_eos=False, max_tokens=2048`，次要对照 | baseline §6；固定长度用于跨组合可比 |
| C7 | 草稿池 | P0 主线只用 0.6B / 1.7B；8B+4B 裸权重合计约 22.75GiB，仅剩约 0.05GiB，无法容纳 activation/draft KV/CUDA graph，P0 不测，P3 再评估 | baseline §1 的三草稿池保留到 P3；P0 不为 4B 花卡时 |
| C8 | Prefix cache | P0 统一 `enable_prefix_caching=False`，结果 JSON 记录该值 | 防止组间预热压低 TTFT，污染门槛 2 |
| C9 | Thinking | 统一关闭；WP3 明确实现（chat completions 用 `chat_template_kwargs={"enable_thinking": false}`，completions 用 `/no_think`），WP0 冒烟断言输出无 ` thinking` | baseline §1；否则四字段正则、引用和接受率全部失真 |
| C10 | Spec-decode 统计 | 离线引擎的 `SpecDecodingStats` / 引擎日志差分；WP0 用 5 条 prompt 证明四字段可提取。若离线确实拿不到 per-position 统计，停止并升级协议修订，不静默切 server | baseline §3；先验证取数路径再建脚手架 |
| C11 | 版本锁定 | `vllm==0.29.0`；torch 版本由 vLLM wheel 依赖解析后在 WP0 用 `pip freeze` 固定并记录；镜像 digest、vLLM commit、CUDA/驱动写入 `results/p0-env/env.json` | 清单要求锁 commit+digest；不手写未验证的 torch 版本 |
| C12 | 数据池 | tuning 50 / report 50 / data-valid 200 / router-train 独立池；四池样本 ID 落盘、两两零交集 | baseline §10/§11；避免调参、报告、路由训练互相泄漏 |
| C13 | 命题判据 | 新增 G5（固定草稿跨域异质性）与 G6（ngram 复制优势）；G1-G4 保留 | 原四门槛无法证伪两个命题 |
| C14 | 矩阵 | 矩阵 = tuning 45 组 + report 52 组 = 97 组；G3 的 A 3 组、G5 的 24 组、G6 的 3 组在 WP4 完成并复用（WP4 复用 30 组）；WP5 新增 67 组；G6 另有 1 个 non-verbatim 控制组，不计入 97 矩阵 | 矩阵以显式计数表为准，可逐项复算 |
| C15 | 卡时口径 | 卡时 = 云实例运行小时，不是 GPU 活跃小时 | AutoDL 按实例开机计费；下载/构样也计入 |
| C16 | docs 落盘 | P0 期间 `docs/` 入库，作为本地唯一事实源的一部分；P6 再决定公开净化版 | 与现状一致，消除 README/baseline 的“docs 不入库”冲突 |
| C17 | 验证 | verifier 独立重跑至少一个固定种子配置；reviewer 审 WP3、D3 报告、WP5 最终 summary/report | AGENTS 完成标准 |

术语：**3 主组合 = 中文原生 / 代码原生 / RAG long[7500,8192]**；主判档 = RAG long。后文所有“3 主组合”均指此定义。

## 1. 前置状态

- 云端（AutoDL 4090 24G）：环境安装进行中；开工前先把 `scripts/setup_env.sh` 改为 `vllm==0.29.0`；torch 版本由该 wheel 的依赖解析后在 WP0 用 `pip freeze` 固定并记录。
- 数据链路未实测（DuReader BOS 存活、ALCE tar 下载）；bench 脚手架未建；词表一致性未验证。
- `docs/` 已随库推送；P0 期间按 C16 入库，baseline §12 已同步为入库。

## 2. 显存预算

fp16 权重按官方 safetensors 文件大小：Qwen3-8B ≈ 15.26GiB，0.6B/1.7B/4B ≈ 1.40/3.78/7.49GiB；`gpu_memory_utilization=0.95` 下可用 ≈ 22.8GiB。

KV 每 token = 36 层 × 8 KV 头 × head_dim 128 × 2 字节 × {K,V} = 147,456 B = **144KiB/token**；8K 单条 = **1.125GiB / 1.208GB(十进制)**。

| 组合 | 权重合计 | 可用 KV 池 | KV token | 8K 并发 |
| --- | --- | --- | --- | --- |
| 8B plain | 15.26GiB | 7.54GiB | ≈54.9K | ≈6.71 |
| 8B + 0.6B | 16.66GiB | 6.14GiB | ≈44.7K | ≈5.46 |
| 8B + 1.7B | 19.04GiB | 3.76GiB | ≈27.4K | ≈3.34 |
| 8B + 4B | 22.75GiB | 0.05GiB | ≈0.37K | ≈0.05 |

推论：

- 8B+4B 裸权重已占约 22.75GiB，只剩约 0.05GiB（约52MiB）；加入 activation、draft KV、CUDA graph 后不可运行。P0 不测，按 C7 处理。
- 8B+1.7B 在 8K 档约支持 3.3 条并发；BS=4/8 会排队/抢占，但 BS=1/2 可测。
- 8B+0.6B 在 8K 档约支持 5.5 条并发；BS=8 会排队/抢占。
- 以上是上界，未计 draft 自身 KV、activation、CUDA graph、分页碎片；WP1 以实测为准。

结论：8K 长档下命名 BS 与实际 running batch 分离；结果 JSON 必须同时记录两者、preemption 计数、每请求 TTFT/TPOT。

## 3. 工作包与顺序

```
WP0 环境就绪确认 ──┬── WP1 探针（词表+显存+LoRA） ── D1 草稿池冻结
                   ├── WP2 数据链路与样本构造 ────── D2 数据源/字段冻结
                   └── WP3 bench 脚手架 ──────────── reviewer 审查
（三条支线并行；WP1/2/3 全绿 → WP4 smoke → WP5 全量矩阵）
```

### WP0 · 环境就绪确认

- 版本锁定：把 `scripts/setup_env.sh` 的 `pip install vllm` 改为 `vllm==0.29.0`；安装后用 `pip freeze` 固定 torch 实际版本；记录 vLLM wheel 版本、vLLM commit、镜像 digest、CUDA/驱动、GPU 型号。
- 权重校验：三个模型（8B/1.7B/0.6B）目录完整；用 safetensors 清单/sha256 校验，不能只看目录存在。
- 离线 spec 统计冒烟：5 条 prompt、8B+0.6B、K=5，证明 `num_drafts` / `num_draft_tokens` / `num_accepted_tokens` / `num_accepted_tokens_per_pos` 能从离线引擎日志或 `SpecDecodingStats` 提取；同时记录提取字段样例。
- Thinking 冒烟：Qwen3-8B 单条中文 prompt，确认关闭 thinking 后输出无 ` thinking`，四字段 JSON 可被正则解析。
- 产物：`results/p0-env/env.json` + 冒烟原始输出；通过判据=版本/权重/统计/thinking 四项全绿。

### WP1 · 探针（不进矩阵）

- 词表一致性：三个模型（8B/1.7B/0.6B）tokenizer 加载，断言 vocab 151936。
- 显存探针：分别启动 8B+0.6B、8B+1.7B，`max_model_len=10240`，记录 KV 池 token 容量、8K 可并发条数、启动内存峰值。
- LoRA 草稿头探针：在 1.7B 上加载一个 dummy LoRA adapter，记录 adapter 增量显存、8B target 下可保留的 KV token 余量和 8K 可并发条数；只做内存探针，不要求 P0 跑通 vLLM spec decode 集成。
- D1 冻结：P0 主线 = {0.6B, 1.7B}；4B 不进 P0 矩阵；LoRA 探针结果作为 P3 路线 a/b 的输入写入报告。

### WP2 · 数据链路与样本构造（不占 GPU，链路探测可立即开始）

1. 链路存活：从 hf-mirror 拉 `PaddlePaddle/dureader_robust` 加载脚本，提取 BOS URL，云端 curl 实测；不通则触发 D2。
2. 数据下载到 `/root/autodl-tmp/data/`：ALCE 451MB tar、repobench-c python/java、hotpotqa、ceval-exam。每条记录 dataset id、config、split、revision、文件 sha256。
3. RAG 扩长：以同一批 ALCE 原生样本为 short 档；按原始检索段落顺序拼接，用 Qwen3 tokenizer 计长，落入 [7500,8192] 为 long 档。记录段落数、token 长度、是否因不足切换同源样本。
4. 四池构造，样本 ID 落盘、两两零交集：
   - tuning 50：B/C 参数调参。
   - report 50：最终报告。
   - data-valid 200：四字段数据侧覆盖与模型侧协议验证。
   - router-train：独立可扩展池，P0 只负责隔离与落盘，规模在 P2 按路由特征维度扩容。
5. 四字段数据侧预筛：用正则/启发式先过滤，保留至少含 3 类字段（公司名/金额/时间/产品名）的篇章；在 data-valid 200 上统计每类字段出现率，每类 ≥60% 才进入模型侧验证。60% 沿用 baseline §11 的建议阈值，P0 冻结为硬门槛，不启用事后校准。
6. 长度分布：Qwen3 tokenizer 分词，按场景报告分位数；每个场景再切三分位桶，供 C4 长度分层分析；模型侧只在该字段出现的文档子集上统计非空率。
7. 输出形态控制：同一批 RAG long 样本配两套模板：verbatim（逐字引用）与 non-verbatim（禁止逐字引用，要求同长度改写/理由输出）；中文 JSON vs 自然段落同样配对；样本 ID 与 `output_mode` 落盘。
- D2 冻结：DuReader 链路不通，或 data-valid 四字段覆盖率不达标，或模型侧逐字约束失败 → 首选 `THUDM/LongBench` 的 `dureader` 子集（中文长文，Apache-2.0），config/split/revision 在链路探测时记录实际值并写入数据清单；若该子集不可达或四字段筛查仍不达标，P0 数据协议失败，停止全量。
- 产物：`data/*.jsonl`（云端，不入库）+ 构建脚本（入库，确定性可复现）+ 长度分布表（入 docs）。

### WP3 · bench 脚手架（`scripts/bench/`）

- 离线 runner：`LLM.generate`，BS∈{1,4,8}；每请求从 `RequestOutput.metrics` 提取 arrival/first-token/finish 时间，计算 TTFT、TPOT、e2e、输出 token/s；同时记录 batch 的 achieved running batch 与 preemption 计数。
- Spec 统计：优先 `SpecDecodingStats` 字段快照，日志解析兜底；字段口径 = `num_drafts`、`num_draft_tokens`、`num_accepted_tokens`、`num_accepted_tokens_per_pos`；接受率分母固定为 `num_draft_tokens`。
- 配置：`enable_prefix_caching=False`、`max_model_len=10240`、thinking 关闭、`temperature=1.0`、`top_p=0.95`、`seed=0`；greedy 只作附加对照。
- 输出两档：fixed512（`ignore_eos=True, max_tokens=512`）与 natural-stop（`ignore_eos=False, max_tokens=2048`）；主口径与门槛一律用 fixed512。
- Prompt 模板：中文 JSON / 中文自然段落 / RAG verbatim / RAG non-verbatim 四套模板版本化；结果 JSON 记录 `output_mode` 与模板版本。
- 预热与顺序：每个引擎启动后丢弃前 3 条请求；组顺序用固定 seed 打乱并记录；同一引擎内不跨 spec 配置复用，避免状态污染。
- 结果 JSON schema 至少含：配置、seed、数据集与长度档、样本 ID、样本数、vLLM 版本/commit、镜像 digest、GPU/驱动、完整命令、BS、achieved running batch、preemption、prefix cache 开关、thinking 开关、`output_mode`、wall time、实例运行时长。
- 本地单测：runner 逻辑、JSON schema、日志解析、mock 空引擎；GPU 部分云端跑。
- 产物：runner + mock 冒烟自检；完成后由 reviewer 子代理审查指标口径、计时方法、seed 处理。

### WP4 · smoke（G1-G6，≤3 天）

执行序：G1/G2 用 N=20 小样，G4 用 data-valid 200 数据侧 + 模型侧中文 JSON 200 条；G3 的 A 3 组、G5 的 24 组、G6 的 3 个矩阵组用 tuning N=50，数据计入 tuning 45，不重复跑；G6 的 non-verbatim 控制组不计入 97 矩阵。

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
- 产物：`results/p0-smoke/*.json` + smoke 小结（入 docs）。

### WP5 · 全量矩阵 + 回填

**Tuning 网格（45 组）**：tuning set N=50，fixed512，BS=1，`enable_prefix_caching=False`；其中 G3 的 A 3 组、G5 的 24 组、G6 的 3 组已在 WP4 完成并复用，WP5 新增 15 组。

- A：1 配置 × 3 主组合 = 3 组。
- B：2 草稿 × K{3,5,7} = 6 配置 × 3 主组合 = 18 组；另 RAG short 配对 6 配置 × BS=1 = 6 组；共 24 组。
- C：num_speculative_tokens{3,5,7} × prompt_lookup_max{3,5} = 6 配置 × 3 主组合 = 18 组。
- B1 选择：在 B 配置中，取 3 个主组合上 BS=1 输出 token/s 加速比的中位数最大者；并列时取 3 主组合上 `num_accepted_tokens/num_draft_tokens` 方差（等权）更小者。
- B2：每个主组合的 B 网格最优配置，作为分场景 oracle 记录。
- C_best：全局中位数最优配置（并列时同 B1 取方差更小者）；同时记录每个主组合的 C oracle。

**Report 网格（52 组）**：report set N=50，报告样本 ID 在所有 policy 间配对。

- fixed512 主矩阵：A / B1 / B2 / C_best × 3 主组合 × BS{1,4,8} = 4×3×3 = 36 组。
- natural-stop 次要矩阵：A / B1 × 3 主组合 × BS{1,8} = 2×3×2 = 12 组。
- 输出形态对照（fixed512，BS{1,8}）：新增 B1 × 中文 × 自然段落 = 2 组、B1 × RAG long × non-verbatim = 2 组，合计新增 4 组；JSON/verbatim 已在 fixed512 主矩阵内，不重复计。
- 总计：矩阵 97 组；WP5 新增 67 组（tuning 15 + report 52）；WP4 复用 30 组；G6 另有 1 个 non-verbatim 控制组。

**分析与回填**：

- 长度分层：每个场景按自然长度三分位桶报告接受率/加速比；RAG 另报 short vs long 配对结果。
- 无损冒烟：固定 seed、temperature=1.0、top_p=0.95，A vs B1 vs C_best，20 条中文 prompt，逐 token 对比；mismatch 必须为 0，否则停止并调查。
- greedy 对照：A / B1 / C_best × 中文 × BS=1 × report set fixed512 = 3 组，标记 exploratory，不作为无损性证明。
- 回填：baseline §13 卡时表 v2、baseline §14 其余项（LoRA 探针写入 P3 移交记录）、README 结果表；汇总数字经 verifier 独立复核后填。
- 产物：`results/p0/*.json`（云端）+ `results/baseline/summary.json`（入库）+ `docs/p0-baseline-report.md`（入库）。

## 4. 角色与节奏

- 主会话：本地写脚本/改码 → commit（人工确认后 push）→ ssh 云端执行 → 拉回结果。
- verifier 子代理：WP4 后独立重跑至少一个固定 seed 的预注册配置（同 GPU、同命令），对比原始 JSON 的汇总与逐位置接受分布；WP5 结束后再独立复核 summary。
- reviewer 子代理：WP3 完成后审脚手架；D3 报告完成后审 go/no-go 判据执行；WP5 收尾前审最终 summary 与基线报告。
- 你：确认本方案；D1/D2/D3 三个决策点；push 确认；smoke 与最终报告验收。
- 卡时记账：每个结果 JSON 记录实例运行时长；累计对照 200 卡时天花板；单次云端 GPU 任务前报预估卡时。

## 5. 卡时粗估（WP4 后回填 baseline §13）

| 项 | 估（卡时） |
| --- | --- |
| WP0 环境/下载/冒烟 | 1-2 |
| WP1 探针 | 0.5 |
| WP2 数据链路与构造 | 1-2 |
| WP4 smoke + G3/G5/G6 tuning 30 组 + 1 控制组 | 4-7 |
| WP5 新增 67 组 | 8-15 |
| 重试、引擎重启、调试余量 | 3-5 |
| **P0 合计** | **18-32**（分项严格和 17.5-31.5，向上取整） |

口径：卡时 = 云实例运行小时；上表为 smoke 前粗估，WP4 结束后用实测单组挂钟时间回填。

## 6. 风险与预案

| # | 风险 | 概率 | 预案 |
| --- | --- | --- | --- |
| R1 | 8K 高并发显存不足导致排队/抢占 | 高 | 记录 achieved running batch 与 preemption；名义 BS 与 achieved BS 分开报 |
| R2 | DuReader 链路死或四字段覆盖不达标 | 中 | D2 固定切 `THUDM/LongBench` 的 `dureader` 子集并重做同一套筛查；仍不达标则 P0 数据协议失败 |
| R3 | B 档加速 <1.3× | 中 | 草稿/K 全网格选型；仍不达则停止并记录 fixed-draft magnitude not met |
| R4 | 命题1未获支持（固定草稿跨域差异不显著） | 中 | G5 失败即停止 P1/P2 并重审定位 |
| R5 | 命题2未获支持（ngram 复制优势不显著） | 中 | G6 失败即停止 P4 的 RAG 专项声明并重审定位 |
| R6 | 离线无法提取 per-position spec 统计 | 低 | WP0 早测；失败则停止并提交协议修订，不静默切 server |
| R7 | ALCE 许可链不明 | 确定 | P0 只用不发布；P6 前给出口径 |
| R8 | 卡时超支 | 中 | 按实例运行时长记账；每 WP 结束对账；超 20 卡时未到 smoke 即升级决策 |
| R9 | thinking 未关闭导致输出失真 | 低 | WP0 冒烟断言；WP3 请求构造强校验 |

## 7. 产物与落盘

- `scripts/bench/`：离线 runner、指标提取、日志解析、本地单测。
- `scripts/data_prep/`：确定性样本构造、四池 ID、长度统计。
- `results/p0-env/env.json`：环境、版本、镜像 digest、GPU/驱动、权重校验（云端）。
- `results/p0-smoke/*.json`、`results/p0/*.json`：原始结果（云端）。
- `results/baseline/summary.json`：汇总（入库）。
- `docs/p0-baseline-report.md`：基线报告（入库）。
- `README.md`：只维护结果表。
