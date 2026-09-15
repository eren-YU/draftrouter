# T2 · WP2 收尾:四池构造 + D2 冻结(1-2 卡时,其中 GPU 占用≈0)

## 范围
数据源接线修复与四池产出。接续状态:下载已全部成功(manifest 含 sha256);
ALCE tar/LongBench zip/repobench parquet/dureader BOS tar 的加载与四池构造未验证通过
(上次运行四池=0,根因是 repobench 脚本仓不可加载、中文场景 zip 路径没人下载)。
先检查是否有未完成的修复(上次会话曾派子代理修复,可能已有半成品 diff)。

## 执行
1. `git status` 查 data_prep 未提交改动;核对 `scripts/data_prep/`(loaders/base_records/download_data)
   是否已含:repobench parquet 加载、dureader BOS tar 解析、LongBench data.zip 路径修正;
   缺什么补什么(字段名先在云端实探一条记录再写解析)。
2. 本地 pytest 全绿 → scp → 云端跑 `build_all.py`(nohup,数据根 `/root/autodl-tmp/data`)。
3. 检查产物并做四字段数据侧筛查(field_screen,退出码 3 = 不达标触发 D2)。

## 验收标准
- [ ] `data/data_manifest.json`:每个数据集有 id/config/split/revision/sha256;
- [ ] `data/pools/*.ids.json`:tuning=50、report=50、data-valid=200、router-train 独立落盘,两两零交集断言通过;
- [ ] `data/samples/*.jsonl`:三场景齐(中文=dureader、代码=repobench-c、RAG=alce),schema 含 sample_id/scene/length_bucket/output_mode;
- [ ] RAG long 档存在且 token∈[7500,8192](主判档),short/long 同 ALCE 样本配对;
- [ ] `length_buckets.json`:三场景三分位边界冻结(tuning+report 池全局计算);
- [ ] field_screen:四字段每类出现率 ≥60%(不达标 → 记录数字,执行 D2 切 LongBench dureader 重筛;仍不达标 = P0 数据协议失败,停止并上报);
- [ ] D2 冻结:实际数据源/config/revision 写入本文件;长度分布表(p10/p50/p90)入 docs;
- [ ] 构建脚本改动本地 commit。

## D2 冻结记录(执行后填写)
- (待填)
