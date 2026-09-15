# DraftRouter 工程约束

领域自适应投机采样推理系统:草稿模型池 + 请求级路由器 + Prompt-Lookup,基于 vLLM 二次开发。

## 环境拓扑

- **本地(D:\code\draftrouter)**:代码唯一事实源。所有编辑、纯逻辑单测(路由器/配置/调度决策,mock 掉 GPU 依赖)、review、commit 都在这里。
- **云端(AutoDL,`ssh autodl` 即达)**:只作为算力执行环境。RTX4090 24G,Ubuntu 22.04,项目在 `/root/autodl-tmp/draftrouter`,conda 环境 `draftrouter`,模型缓存 `/root/autodl-tmp/hf-cache`。
- 同步方式:本地 commit/push → `ssh autodl "cd /root/autodl-tmp/draftrouter && git pull"`。禁止在云端直接改代码(会被下次 pull 覆盖)。

## 云端操作规则

- 非交互 ssh 下 conda 不在 PATH:用 `/root/miniconda3/envs/draftrouter/bin/python`,或先 `source /root/miniconda3/etc/profile.d/conda.sh && conda activate draftrouter`。
- 预计超过 5 分钟的任务(微调、评测、大规模生成)必须放 tmux:`ssh autodl "tmux new -d -s <名字> '<命令>'"`,用 `tmux capture-pane -p -t <名字>` 查看输出。禁止留前台挂起的 ssh。
- 跑完任务随手确认 GPU 已释放(`nvidia-smi`),不留僵尸 python 进程占卡。
- 模型权重、数据集、checkpoint 一律不进 git(见 .gitignore),放云端数据盘。
- 换 GPU 型号需人工确认:性能数字只在同型号卡上可比。

## 完成标准

AI 报告"完成"之前必须自证:相关测试全绿;涉及性能的改动附与基线并排的数字;涉及 P3 引擎改动的附固定种子逐 token 一致性验证结果。交付前派 `verifier` 子代理独立重跑,派 `reviewer` 子代理独立审查。

## 阶段与文档

按 `以终为始-工程执行清单.md` 的 P0–P7 推进,一个阶段一个会话。每阶段开工时先产出短设计文档到 `docs/pN-design.md`,人工确认后再实现。参考指标(TPOT -57%、吞吐 2.3×、接受率 +31%、成本 -45%、BS≤8)是方向,不是硬验收;但性能回退必须标红。
