# 信息面增量管线运行约定

`information_pipeline` 是公告、新闻语料和事后市场反应的生产调度入口。它不会向
券商发单，也不会把事后收益写回实时特征。

## 串行阶段

1. `cninfo_announcements_download`：按滚动日期窗发现并下载巨潮公告正文；默认只取
   监管类高信号公告。
2. `announcement_nlp`：在正文下载成功后才创建，按正文 SHA-256、提示词版本和模型
   幂等恢复。
3. `corpus_nlp`：消费已经落盘的新闻和互动问答语料；是否启用及来源范围由调度配置
   固定。
4. `event_market_response`：只读取已经通过阻断质量门的不可变快照，产出
   `training_label_only` 事后标签。

每个阶段都是单独的 durable job。后继任务只在前一阶段退出码为零时创建；同一调度
运行使用固定的 `pipeline_id` 和幂等键。已有行情下载、公告下载、NLP、快照或 Qlib
构建任务时，本次运行记为 `skipped`，不重复排队，也不争抢正在运行的 I/O。

## 安全默认值

- `enable_nlp=false`：新建调度默认只做原始监管公告增量，不产生 LLM 费用。
- 打开 NLP 后，`announcement_nlp_limit` 和 `corpus_nlp_limit` 必须在 1–10,000；默认
  每类每轮最多 500 条，禁止定时任务无限补积压。
- 事件标签默认选择运行日之前最新的、`verification.json.ok=true` 且无错误的不可变
  快照；也可用 `snapshot_name` 显式钉住。
- 真实数据不存在的年份保持缺失。公告正文当前可信发现边界为 2016-01-01，新闻和
  互动语料按各自最早真实落盘日期开始，不向 2008 年伪造文本或零信号。

## 创建示例

先以零 LLM 成本建立每日原始公告增量：

```json
{
  "name": "daily regulatory information update",
  "kind": "information_pipeline",
  "timezone": "Asia/Shanghai",
  "run_time": "21:30:00",
  "trading_days_only": true,
  "payload": {
    "lookback_days": 7,
    "regulatory_only": true,
    "enable_nlp": false
  },
  "misfire_grace_seconds": 1800,
  "actor": "quantlab-operator"
}
```

历史正文补齐、LLM 密钥可解密且容量预算确认后，再创建显式有界的完整信息管线：

```json
{
  "name": "daily governed information features",
  "kind": "information_pipeline",
  "timezone": "Asia/Shanghai",
  "run_time": "22:00:00",
  "trading_days_only": true,
  "payload": {
    "lookback_days": 7,
    "regulatory_only": true,
    "enable_nlp": true,
    "announcement_nlp_limit": 500,
    "include_corpus_nlp": true,
    "corpus_datasets": ["major_news", "cctv_news", "irm_qa_sh", "irm_qa_sz"],
    "corpus_nlp_limit": 500,
    "batch_size": 40,
    "major_news_per_day": 40,
    "irm_per_instrument_day": 2,
    "include_event_labels": true,
    "horizons": [1, 3, 5, 20],
    "benchmark_code": "000300.SH"
  },
  "misfire_grace_seconds": 1800,
  "actor": "quantlab-operator"
}
```

配置校验会拒绝未知字段、无界 NLP、未开启 NLP 却请求下游消费者、非法收益期限以及
未通过质量门的自动快照选择。生产启用仍须等当前长下载任务结束，在安全发布窗口部署
并完成一次人工观察运行后再激活日调度。
