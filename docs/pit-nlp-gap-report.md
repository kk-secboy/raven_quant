# PIT 与 NLP 链路现状差距清单

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-07-18 |
| 来源 | 目标"公告文本 → NLP 信号字段 → PIT 语义"第一阶段调查 |
| 调查范围 | 主工作区 `src/quant_data`、`src/quant_platform`、`tests`（不含 `.worktrees/` 副本） |
| 对照标准 | 《个人量化投资与模拟盘系统设计稿.md》3.3 节（PIT 与可用时间）、6.8 节（信号语义） |

## 一、PIT 语义现状

| 方面 | 评级 | 关键证据 |
| --- | --- | --- |
| 三级时间戳（effective/available/ingested） | 已实现 | effective_at≈源数据自带列（`end_date` 等）；available_at 由显式政策注册表声明（`src/quant_data/availability.py:71-103`，财务 `strictly_after_announcement_date`、行情衍生 `same_trade_date_after_close`、指数/行业元数据 `effective_date_with_lag(days=5)`）；ingested_at 行级落 parquet（`src/quant_data/storage.py:86-89`），快照合并保留并兼容旧单元 NULL（`storage.py:262-272`），manifest 记录 `ingested_at_min/max`（`storage.py:273-274`） |
| 可恢复等级（native_history/reconstructed/current_only/unavailable） | 已实现 | 四级词汇注册表 `RECOVERABILITY_LEVELS`（`src/quant_data/availability.py:105-164`，逐组注释依据；未标注数据集 fail-safe 归为 `unavailable`），进 snapshot manifest（`storage.py:275`）与 qlib research_feature_contract（`qlib_builder.py:938-941`），证据特征强制测试 `tests/test_qlib_builder.py:850` |
| 读取侧强制执行 | 已实现 | 构建期财务按公告日 ASOF 对齐（`src/quant_data/qlib_builder.py:821-833`）；资格矩阵公告后生效且 fail-closed（`src/quant_platform/eligibility.py:120-131,270-308`）；读取守卫 `filter_available`（`src/quant_data/availability.py:196-240`，无政策数据集 fail-closed 抛错）并接入临时读取路径（`scripts/run_multifactor_backtest.py:276,282`、`scripts/run_recommendation_refresh.py:390,395`）；组合回测层对行业/权重应用带版本保守滞后（`src/quant_platform/strategy_backtest.py:57,102,143,243-274`） |

已有的扎实实现（不重复开发）：

- 财务因子 `fina_indicator` 按"严格晚于公告日"进入 Qlib 特征，契约自声明 `strictly_after_announcement_date`（`qlib_builder.py:917-920`），并有防泄漏测试（`tests/test_qlib_builder.py:197`）。
- 退市/资格信息不回填历史（`eligibility.py` 全套 PIT 测试，`tests/test_eligibility.py:6-161`）。
- 行业成员 `index_member_all` 按 `in_date/out_date` 区间展开（`qlib_builder.py:451-468`），消费侧按决策时点取当时版本。
- 数据集身份钉住（`dataset_identity_sha256` + lineage）防止数据被替换（`research_store.py:93-109`）。

## 二、PIT 缺口清单

| # | 缺口 | 现状证据 | 工作量 | 关闭状态 |
| --- | --- | --- | --- | --- |
| P1 | 行级 `ingested_at` 不落数据文件，无法回答"这行数据平台何时取得"；原始响应体默认不保留（`storage.py:58` 的 `keep_raw=False`） | `storage.py:66-83`、`checkpoint.py:113-128` | S | 【已关闭】2026-07-18 |
| P2 | 无 `available_at` 抽象：财务用 `ann_date` 代替，但行情衍生字段（daily_basic）、指数权重、行业成员都默认"生效日即可知"，无可用滞后建模 | `qlib_builder.py:795-799`（`same_trade_date_after_close` 仅是约定）；`strategy_backtest.py:222-229` | M | 【已关闭】2026-07-18 |
| P3 | 财务修订无处理路径：按 period 重拉 + `SELECT DISTINCT` 合并导致修订行共存，构建期去重对同 `(ann_date, end_date)` 冲突行无 tie-break | `supplemental_data.py:572-593`、`storage.py:329-333`、`qlib_builder.py:819-822` | M | 【已关闭】2026-07-18 |
| P4 | reference 类数据（stock_basic、index_classify 等 29+ 个）实质 current_only：快照只取最新代，历史状态不可复原，且无字段级等级标注 | `reference_data.py:18-54,174-241` | M | 【已关闭】2026-07-18 |
| P5 | `disclosure_date`（披露日历）已下载但无任何消费方，未用于校准 `ann_date` 或生成 available_at | `planner.py:234-245`；grep 仅注册点 | S | 【已关闭】2026-07-18（只标记不阻断） |
| P6 | 读取侧无统一栅栏与回归测试：PIT 靠"构建好的数据集+自律"，新代码直接读 snapshot/parquet 可绕过全部规则；无财务重述/泄漏回归测试 | tests 中 PIT 测试集中在构建期，无读取侧守卫测试 | M | 【已关闭】2026-07-18 |

关闭证据（2026-07-18，测试文件均带 `pytestmark = pytest.mark.no_database`）：

- **P1**：`storage.write_unit` 为每行写 tz-aware UTC `ingested_at`（`src/quant_data/storage.py:86-89`，可注入 clock）；`build_snapshot` 经 `union_by_name` 保留该列、旧单元填 NULL，manifest 记录 `ingested_at_min/max`（`storage.py:262-276`）。测试：`tests/test_pit_semantics.py::test_write_unit_records_tz_aware_row_level_ingested_at`、`test_snapshot_preserves_ingested_at_and_null_fills_legacy_units`。
- **P2**：政策注册表 `AVAILABILITY_POLICIES`（`src/quant_data/availability.py:71-103`）；政策写入 qlib `research_feature_contract.availability_policy` + `availability_policy_version`（`qlib_builder.py:924-941`）；消费侧 `_snapshot`/`_industries_at` 应用带版本保守滞后（`AVAILABILITY_LAG_CONFIG_VERSION=1`、默认 5 个自然日，`availability.py:47-48`；`strategy_backtest.py:57,102,143,243-274`；注释写明"真实公布滞后无源数据，属近似"）。测试：`tests/test_pit_semantics.py` 的 `test_availability_registry_declares_all_policy_families`、`test_index_weight_applies_the_conservative_publication_lag`、`test_snapshot_helper_applies_publication_lag`、`test_industries_at_helper_applies_publication_lag`；`tests/test_qlib_builder.py` 泄漏测试已更新为新契约。
- **P3**：`_fundamental_revision_order`（`src/quant_data/qlib_builder.py:1108-1128`）：`end_date` 之后依次按 `f_ann_date`（列存在时）、`update_flag`（列存在时）、`ingested_at`（列存在时）取较新者，最终以投影列内容哈希 `md5(concat_ws(...))` 保证全序。测试：`tests/test_qlib_builder.py::test_fundamental_revision_conflict_prefers_newest_f_ann_date`、`test_fundamental_revision_conflict_uses_latest_ingested_at`、`test_fundamental_revision_dedup_is_deterministic_across_row_order`（换序+重跑逐帧相等）。
- **P4**：`RECOVERABILITY_LEVELS`（`availability.py:105-164`）逐组标注并在注释说明依据；进 snapshot manifest（`storage.py:275`）与 qlib 契约/provenance（`qlib_builder.py:938-941`，经 `research_feature_contract` 进入 identity 哈希）。强制测试：`tests/test_qlib_builder.py::test_research_contract_admits_only_evidence_grade_recoverability`（契约内数据集只允许 `native_history`/`reconstructed`）、`tests/test_pit_semantics.py::test_recoverability_levels_cover_the_audited_groups`。
- **P5**：`verify._verify_disclosure_reconciliation`（`src/quant_data/verify.py:406-498`）：财务行 `ann_date` 与披露日历 `coalesce(actual_date, ann_date)` 对账，只进 `warnings` 与 `disclosure_checks`，不阻断。测试：`tests/test_pit_semantics.py` 的 `test_disclosure_reconciliation_*` 三个用例。
- **P6**：读取守卫 `filter_available`（`availability.py:196-240`）：按注册表政策过滤、无政策数据集抛 `AvailabilityPolicyError`、日期列缺失抛错、不可解析日期 fail-closed 丢弃；已接入 `scripts/run_multifactor_backtest.py:276,282` 与 `scripts/run_recommendation_refresh.py:390,395` 的临时元数据读取路径。测试：(a) 重述确定性 `tests/test_qlib_builder.py::test_financial_restatement_applies_only_after_the_new_announcement`；(b) available_at 之前读不到 `tests/test_pit_semantics.py::test_financial_fields_are_invisible_before_the_announcement_date`；(c) 无政策被拒 `test_guard_fails_closed_for_unregistered_dataset`。rdagent/research 入口排查：`research_automation.py:196` 只读因子 artifact（自带 datetime 轴与 sha256 lineage），不直接读 snapshot parquet；`market_overview.py` 为盘中观察视图、按 `trade_date` 过滤展示，不作为研究证据读取路径。

## 三、NLP 链路现状

| 环节 | 评级 | 关键证据 |
| --- | --- | --- |
| 公告全文下载 | 已实现（2026-07-18） | `src/quant_data/cninfo_announcements.py`：以 `anns_d` 为发现源抓 PDF 正文，内容寻址不可变文件 + 每轮下载日志 + 限速 + 幂等跳过；元数据索引行级 `available_at`（公告次一交易日，fail-closed 依赖 trade_cal）与 `ingested_at`（`cninfo_announcements.py:213-552`） |
| 问询函/监管函 | 已实现（2026-07-18） | 标题关键词分类 `REGULATORY_TITLE_PATTERN`/`categorize_title`（问询函/关注函/监管函/警示函/纪律处分 → `regulatory_letter`，`cninfo_announcements.py:20-81`） |
| LLM/NLP 抽取 | 已实现（2026-07-18） | `src/quant_platform/announcement_nlp.py`：PDF 文本抽取 → LLM 抽取（事件类型、语气分数、关键数值）→ 结构化字段索引，行级 `available_at`/`ingested_at`，原子写、按 sha256+prompt_version+model 幂等、fail-closed |
| 信号字段落地 | 已实现（2026-07-18） | 因子库机制完整（`factor_candidates`/`factor_evaluations`，`database.py:143-256`）；announcement_nlp 因子值 artifact 按 datetime/instrument 单值序列 parquet + sha256 落盘（`announcement_nlp.py:513-549`）；入库通道已接：`announcement_factor_registry.register_announcement_factor`（`announcement_factor_registry.py:181`）校验 manifest sha256 后经 `ResearchStore.add_candidate` 落库，幂等键 (name, values_sha256)（`research_store.py:437`），CLI `quant-db register-announcement-factor`（`db_cli.py:58`） |
| 文本语料消费 | 部分实现（2026-07-18） | 公告正文语料已被 `announcement_nlp.py` 消费；`major_news` 的 `content` 全文（`supplemental_data.py:1086-1098`）与 `research_corpus` bundle 仍无消费方 |

信号落地关闭证据（2026-07-18）：`src/quant_platform/announcement_factor_registry.py` — `_verified_artifact`（:70）fail-closed 校验 manifest（schema、availability_policy、source 身份、parquet 实测 sha256 与 manifest 一致，任何不符在写库前抛错）；`register_announcement_factor`（:181）创建 kind=`announcement_nlp_factor_import` 研究运行（成功 mark `succeeded`、失败 mark `failed`，不占住 active-kind 唯一槽位），生成确定性谱系 code artifact（`_code_artifact_source` :127，内容可由 fields 索引重算因子值），availability_policy/prompt_version/model 写入候选 description+variables_json 现有字段（不加表字段）；幂等键 (name, values_sha256)（`research_store.py:437` `find_candidate`）：同 sha256 重注册返回既有候选不建行，新 sha256 建新候选（候选行不可变，新行即既有版本机制）。测试：`tests/test_announcement_factor_registry.py`（15 个：注册成功、幂等重注册、新版本、缺 artifact/校验和不符 fail-closed、CLI，其中 10 个带 `no_database`）。

## 四、可复用机制清单（开发约束，禁止另起炉灶）

- **待抓取清单**：`anns_d` 已按日全量落盘且含 PDF URL，直接作为公告正文下载的发现源，不另建公告发现机制。
- **密钥配置**：复用 `RuntimeSecretStore` 的 `"llm"` 记录（api_key/api_base/chat_model 三元组，`runtime_secret_store.py:13-135`、`api.py:1584-1602`），不新建密钥体系。
- **下载纪律**：不可变快照 + parquet/zstd 单元（`storage.py:66-74,124-216`）、下载日志（接口/参数/时间/行数/错误）、限速（`rate_limit.py`）、幂等重跑（`checkpoint.py`）。
- **PIT 契约写法**：参照 `qlib_builder.py:903-921` 的 `research_feature_contract` + `availability_policy`。
- **因子落地**：datetime/instrument 单值序列 parquet artifact + sha256 入 `factor_candidates`（`factor_evaluator.py:14-35`）。
- **任务注册**：新数据能力接入 `data_task_store.py` 任务目录，不旁路。

## 五、开发项与依赖顺序

| 顺序 | 开发项 | 关闭的缺口 | 状态 |
| --- | --- | --- | --- |
| 1 | 巨潮公告正文下载器：以 `anns_d` 为清单抓 PDF 正文，问询函按标题关键词分类（问询函/关注函/监管函/警示函），落原始快照纪律（不可变、校验和、下载日志、限速、幂等）+ 元数据索引（含 `available_at`/`ingested_at`） | NLP-公告全文、NLP-问询函；顺带实现 P1 的 ingested_at 落盘模式 | 已关闭（2026-07-18，`cninfo_announcements.py`） |
| 2 | NLP 加工层：PDF 文本抽取 → LLM 抽取（事件类型、语气分数、关键数值）→ 结构化信号字段（带 `available_at`/`ingested_at`）→ 因子值 artifact 入 `factor_candidates` | NLP-LLM 抽取、NLP-信号落地 | 已关闭（2026-07-18，`announcement_nlp.py` + 入库接线 `announcement_factor_registry.py`，见第三节） |
| 3 | PIT 语义补全：按第二节缺口逐项关闭（P1/P2/P5/P6 先行；P3/P4 评估后决定，若需改设计则按停止规则报告） | P1–P6 | 已关闭（2026-07-18，P1–P6 全部关闭，见第二节关闭证据） |

工作量标记：S = 小（单模块/单天级），M = 中（跨模块/数天级）。

## 六、验证证据（2026-07-18 收尾）

- 信号落地接线（2026-07-18）：新增 `src/quant_platform/announcement_factor_registry.py`（manifest sha256 fail-closed 校验 → 研究运行谱系 + `ResearchStore.add_candidate` 入库，幂等键 name+values_sha256）与 `quant-db register-announcement-factor` 子命令（`db_cli.py:58`）；`ResearchStore.find_candidate`（`research_store.py:437`）。新增测试 `tests/test_announcement_factor_registry.py` 15 个全部通过（10 个 `no_database` 纯逻辑 + 5 个真实 PG）；回归 `test_factor_governance.py`/`test_announcement_nlp.py`/`test_cninfo_announcements.py`/`test_pit_semantics.py` 79 个通过；`ruff check src tests` 通过。候选落库后状态为 `awaiting_evaluation`（与 RD-Agent 导入候选一致）；面向外部因子的评估执行器（provider input 非 daily_pv，现有 `factor-recompute-v1` 契约不适用）为后续独立开发项。

- 新增测试：`tests/test_cninfo_announcements.py`（16）、`tests/test_announcement_nlp.py`（39）、`tests/test_pit_semantics.py`（17）+ `tests/test_qlib_builder.py` 增补用例，全部通过；相关既有套件无回归（ruff 同步通过）。
- 真实链路演示（`.pytest-tmp/cninfo_e2e.py`）：以巨潮公开查询接口取得一条真实公告（600519.SH，2026-07-18 公告）→ 真实 HTTP 下载 PDF（67,261 字节，sha256 校验一致）→ 索引行 `available_at=2026-07-20`（周六公告顺延至次一交易日）、`ingested_at` 为 UTC 时间戳 → 决策时点 2026-07-19 读取返回 0 行、2026-07-20 返回 1 行；重跑 `skipped=1` 零 HTTP（幂等）。
- 凭证阻塞（按停止规则报告，未用 mock 冒充）：完整链路中"Tushare 拉取 anns_d 清单"需 `TUSHARE_TOKEN`、"NLP 加工层真实调用"需 LLM 密钥（`OPENAI_API_KEY` 或 `RuntimeSecretStore.get("llm")`），本环境两者均未配置，故这两个环节以单元/集成测试 + 上述真实下载演示作证据，待凭证就位后可直接重跑 CLI 验证。
- 可复现端到端入口：`tests/test_announcement_live_e2e.py` 已实现全真实链路判定（Tushare anns_d/trade_cal → cninfo PDF → NLP 结构化字段 → 因子 artifact → available_at 之前不可见），无任何 mock；凭证缺席时干净跳过，在环境中配置 `OPENAI_API_KEY`（及可选 `TUSHARE_TOKEN`）后运行 `pytest tests/test_announcement_live_e2e.py` 即完成第三条判定的最终验证。
- 【2026-07-18 已通过】真实链路验证：以 deepseek-v4-pro 为 LLM、巨潮公开查询接口为发现源（本环境 TUSHARE_TOKEN 未配置时的替代发现路径，Tushare 路径在同测试内保留），`pytest tests/test_announcement_live_e2e.py` **1 passed**：300093.SZ 权益变动公告 → 真实下载 PDF → LLM 产出 `event_type=equity_change`、`tone_score=-0.1`、`confidence=0.98`、关键数值（持股数/比例）→ 字段 `available_at=2026-07-20`（周末公告顺延）、`ingested_at` UTC 时间戳 → 因子 artifact 在 `available_at` 前不可见。同跑全量套件退出码 0（该 live 测试在未注入凭证的常规套件中保持跳过）。
- `tests/test_document_governance.py` 的 24 个失败为本目标开始前的既有状态（用户文档迁移：v4.4 旧稿与两个 docx 已删、治理测试尚在重指向 v1.1 设计稿的过程中）。【已于 2026-07-18 关闭】：经用户确认后，治理测试按 v1.1 设计稿重写（28 个用例全过），被删旧 md 从 git 原样恢复（v1.1 §14.1 链接所需），全量套件 `pytest tests/ -q` 复核退出码 0（655 tests / 0 failures / 1 skipped=credential-gated live e2e）。
