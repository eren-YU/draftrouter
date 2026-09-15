# T5 · WP5 全量矩阵(8-15 卡时)+ T6 分析回填(0 GPU)

## T5 范围
action-plan §3 WP5:tuning 新增 15 组 + report 52 组(fixed512 主矩阵 36 + natural-stop 12 + 输出形态对照 4),
复用 WP4 已完成的 30 组。组清单必须显式计数(97 = 45 tuning + 52 report,可逐项复算)。

### 验收标准
- [ ] 组清单文件(入库):每行 = spec 配置 × 数据集 × 长度档 × BS × output_mode,总数=97,无重复无遗漏;
- [ ] `results/p0/*.json` 全量产出(云端),单组 JSON 合规;
- [ ] B1/B2/C_best 选择规则按 §3 定义执行(中位数最大/方差并列裁决),选择过程可复算;
- [ ] 无损冒烟:固定 seed、temp=1.0、top_p=0.95,A vs B1 vs C_best 20 条中文 prompt 逐 token 对比,mismatch=0,否则停止并调查;
- [ ] greedy 对照 3 组标记 exploratory;
- [ ] 卡时对账。

## T6 范围(纯本地/纯 CPU)
长度分层分析、summary 汇总、回填基线报告。

### 验收标准
- [ ] `results/baseline/summary.json`(入库)+ `docs/p0-baseline-report.md`(入库);
- [ ] baseline §13 卡时表 v2、§14 其余项(LoRA 探针写 P3 移交记录)、README 结果表回填;
- [ ] reviewer 审 D3 报告与最终 summary(go/no-go 判据执行是否忠实);
- [ ] verifier 独立复核 summary 数字(与原始 JSON 对比);
- [ ] 全部本地 commit;push 与报告验收交用户。
