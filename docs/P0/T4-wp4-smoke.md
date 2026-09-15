# T4 · WP4 smoke:G1-G6 + D3 决策(4-7 卡时;开工起 ≤3 天时限)

## 范围
action-plan §3 WP4 全表:G1(18 组)/G2(4 组)/G3(A 3 组新增)/G4(数据协议)/G5(24 组)/G6(3 组+1 控制组),
全用 tuning N=50 / fixed512 / BS 与配置见门槛表。G4 依赖 T2 的 data-valid 200。

## 前置
- T1(T2 至少 G4 部分就绪)、T3 完成;样本 jsonl 已在云端;
- 跑前写一个组清单生成脚本(97 矩阵中 WP4 的 30 组复用部分要显式列出,防漏防重)。

## 执行要点
- 每组 = 一次 `run_group.py` 调用(每 spec 配置新建 LLM);组间用固定 seed 打乱顺序并记录;
- 串行跑;单组超时/OOM 的处理:记入失败清单,G1 判据是"全部跑完无 OOM";
- G2/G5/G6 的统计判据:paired bootstrap 10000 次 seed=0,长度桶边界用 T2 冻结的 length_buckets.json;
- 卡时:每组 wall time 记录,过 20 卡时未到 smoke 完成即停(升级决策,R8)。

## 验收标准
- [ ] `results/p0-smoke/*.json`:每组的 schema 26 键齐全,含 achieved batch/preemption;
- [ ] G1-G6 每项 PASS/FAIL 按判据逐条判定(判据原文见 action-plan §3 WP4 表),失败走对应失败分支并记录;
- [ ] D3 总决策(进 WP5 / 降级 / 停止)写入本文件,附每组关键数字(median TPOT、接受率、加速比+CI);
- [ ] smoke 小结入 `docs/`;
- [ ] verifier 独立重跑至少 1 个固定 seed 预注册配置,对比原始 JSON 汇总与逐位置接受分布;
- [ ] GPU 释放,卡时对账回填 action-plan §5。

## D3 决策记录(执行后填写)
- (待填)
