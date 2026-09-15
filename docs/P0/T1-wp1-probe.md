# T1 · WP1 探针 + D1 冻结(0.5 卡时)

## 范围
执行 `scripts/wp1_probe.py`(已上云,未跑),产出探针数据并冻结 D1。不改探针脚本逻辑(除非跑挂)。

## 执行
```bash
# 先确认 GPU 空闲,再后台跑(约 15-25 分钟,四个引擎串行启动)
ssh autodl "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
scp scripts/wp1_probe.py autodl:/root/autodl-tmp/draftrouter/scripts/   # 若有改动
ssh autodl "cd /root/autodl-tmp/draftrouter && source docs/P0/00-context.md 中环境变量四件套 && \
  nohup /root/miniconda3/envs/draftrouter/bin/python scripts/wp1_probe.py > results/wp1.log 2>&1 &"
# 轮询:tail results/wp1.log;结束后 cat results/p0-env/wp1.json
```
注意:LoRA 探针依赖 `peft`(云端已装)。若 `lora_modules`/LoRARequest 参数在 0.29 报错,
按实际签名修正 snippet 后重跑,只修不绕。

## 验收标准
- [ ] 词表:三模型 vocab=151936 断言通过,wp1.json `vocab.status=pass`;
- [ ] 8B+0.6B 与 8B+1.7B:各拿到 kv_cache_tokens、max_concurrency_8k、reserved_gib(全非 null);
- [ ] 与 §2 预算表对照(预算:8B+0.6B≈44.7K tokens/5.46 并发;8B+1.7B≈27.4K/3.34),偏差 >25% 需解释;
- [ ] LoRA:1.7B±LoRA 的 KV token 差值、换算到 8B target 的可保留 KV 余量与 8K 并发估计;
- [ ] D1 冻结结论写入本文件下方"P0 主线 = {0.6B, 1.7B};4B 不进 P0"确认段 + 卡时记账;
- [ ] `nvidia-smi` GPU 释放;env 产物拉回本地留档(或 JSON 摘要入 docs)。

## D1 冻结记录(执行后填写)
- (待填)

## 已知问题(2026-09-15 首跑 `results/p0-env/wp1.json`,T1 开工先修)
1. **词表断言口径错误**:`len(tokenizer)=151669`(8B),151936 是 config 的 embedding 行数。
   正确断言 = ①三模型 tokenizer 词表互相完全一致(长度+内容 hash);②三者 config vocab_size=151936。
2. **KV/显存解析失败**:kv_cache_tokens=10560 可疑(预算≈44.7K),reserved_gib=0.0、
   max_concurrency_8k=null —— `scripts/wp1_probe.py` 的正则与 0.29 日志措辞不符;
   修正法:跑一次看 `results/p0-env/wp1-engine-8b06b.log` 原文,按实际行收紧正则。
3. **8B+1.7B 引擎启动失败**(returncode=1):查 `results/p0-env/wp1-engine-8b17b.log` 根因(疑似显存/KV 配置)。
4. **LoRA 探针失败**:`lora_modules=None` 等参数不合法(TypeError),按 0.29 签名修正 snippet。
   修完只重跑失败子项,不必重跑已成功的步骤。
