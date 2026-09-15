# T3 · reviewer 审查 WP3 bench 脚手架(0 GPU 时)

## 范围
派 reviewer 子代理独立审查 `scripts/bench/`(config/configs/prompts/spec_stats/runner/schema/run_group + tests),
重点是 0.29 实测参数回填后的正确性。只审不改;问题清单回本地修。

## 审查清单(给 reviewer 的输入)
- 指标口径:TTFT/TPOT/e2e 计算是否符合 baseline §3(TPOT=(finished-first_token)/(n-1)?分母口径审);
- spec 四字段映射(`spec_stats.py::spec_stats_from_request_output`)与聚合逻辑;
- 计时:wall_time 边界、预热丢弃 3 条是否真的在统计外;
- seed 处理:组顺序 shuffle 固定 seed=0 且记录;SamplingParams seed 传递;
- C2:BS 名义值 vs achieved running batch vs preemption 三者是否分开记录;
- C5-C9:max_model_len=10240 / prefix_caching=False / thinking 关闭 / fixed512(ignore_eos)与 natural-stop(2048)两档;
- schema.py 26 键是否覆盖 action-plan §3 WP3"结果 JSON schema 至少含"清单;
- 单测是否真的 mock 掉 GPU、无 vllm 环境可跑。

## 验收标准
- [ ] reviewer 输出结构化意见(P0-P3 分级);
- [ ] P0/P1 问题修复后 pytest 全绿并 commit;
- [ ] 结论"WP3 脚手架可支撑 WP4"写回本文件。
