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
| P1 | 行级 `ingested_at` 与原始响应留存 | `storage.write_unit` 落行级 UTC 时间；`ParquetStore` 与 `Settings` 默认 `keep_raw=True`，仍可显式关闭 | S | 【已关闭并强化】2026-07-28 |
| P2 | 无 `available_at` 抽象：财务用 `ann_date` 代替，但行情衍生字段（daily_basic）、指数权重、行业成员都默认"生效日即可知"，无可用滞后建模 | `qlib_builder.py:795-799`（`same_trade_date_after_close` 仅是约定）；`strategy_backtest.py:222-229` | M | 【已关闭】2026-07-18 |
| P3 | 财务修订无处理路径：按 period 重拉 + `SELECT DISTINCT` 合并导致修订行共存，构建期去重对同 `(ann_date, end_date)` 冲突行无 tie-break | `supplemental_data.py:572-593`、`storage.py:329-333`、`qlib_builder.py:819-822` | M | 【已关闭】2026-07-18 |
| P4 | reference 类数据（stock_basic、index_classify 等 29+ 个）实质 current_only：快照只取最新代，历史状态不可复原，且无字段级等级标注 | `reference_data.py:18-54,174-241` | M | 【已关闭】2026-07-18 |
| P5 | `disclosure_date`（披露日历）已下载但无任何消费方，未用于校准 `ann_date` 或生成 available_at | `planner.py:234-245`；grep 仅注册点 | S | 【已关闭】2026-07-18（只标记不阻断） |
| P6 | 读取侧无统一栅栏与回归测试：PIT 靠"构建好的数据集+自律"，新代码直接读 snapshot/parquet 可绕过全部规则；无财务重述/泄漏回归测试 | tests 中 PIT 测试集中在构建期，无读取侧守卫测试 | M | 【已关闭】2026-07-18 |

关闭证据（2026-07-18，测试文件均带 `pytestmark = pytest.mark.no_database`）：

- **P1**：`storage.write_unit` 为每行写 tz-aware UTC `ingested_at`（可注入 clock）；`build_snapshot` 经 `union_by_name` 保留该列、旧单元填 NULL，manifest 记录 `ingested_at_min/max`。2026-07-28 再强化为 `Settings.keep_raw=True`、`ParquetStore(..., keep_raw=True)` 和 `.env.example KEEP_RAW_RESPONSES=true`，测试同时验证原始响应文件默认生成；只有操作者显式关闭时才不留存。
- **P2**：政策注册表 `AVAILABILITY_POLICIES`（`src/quant_data/availability.py:71-103`）；政策写入 qlib `research_feature_contract.availability_policy` + `availability_policy_version`（`qlib_builder.py:924-941`）；消费侧 `_snapshot`/`_industries_at` 应用带版本保守滞后（`AVAILABILITY_LAG_CONFIG_VERSION=1`、默认 5 个自然日，`availability.py:47-48`；`strategy_backtest.py:57,102,143,243-274`；注释写明"真实公布滞后无源数据，属近似"）。测试：`tests/test_pit_semantics.py` 的 `test_availability_registry_declares_all_policy_families`、`test_index_weight_applies_the_conservative_publication_lag`、`test_snapshot_helper_applies_publication_lag`、`test_industries_at_helper_applies_publication_lag`；`tests/test_qlib_builder.py` 泄漏测试已更新为新契约。
- **P3**：`_fundamental_revision_order`（`src/quant_data/qlib_builder.py:1108-1128`）：`end_date` 之后依次按 `f_ann_date`（列存在时）、`update_flag`（列存在时）、`ingested_at`（列存在时）取较新者，最终以投影列内容哈希 `md5(concat_ws(...))` 保证全序。测试：`tests/test_qlib_builder.py::test_fundamental_revision_conflict_prefers_newest_f_ann_date`、`test_fundamental_revision_conflict_uses_latest_ingested_at`、`test_fundamental_revision_dedup_is_deterministic_across_row_order`（换序+重跑逐帧相等）。
- **P4**：`RECOVERABILITY_LEVELS`（`availability.py:105-164`）逐组标注并在注释说明依据；进 snapshot manifest（`storage.py:275`）与 qlib 契约/provenance（`qlib_builder.py:938-941`，经 `research_feature_contract` 进入 identity 哈希）。强制测试：`tests/test_qlib_builder.py::test_research_contract_admits_only_evidence_grade_recoverability`（契约内数据集只允许 `native_history`/`reconstructed`）、`tests/test_pit_semantics.py::test_recoverability_levels_cover_the_audited_groups`。
- **P5**：`verify._verify_disclosure_reconciliation`（`src/quant_data/verify.py:406-498`）：财务行 `ann_date` 与披露日历 `coalesce(actual_date, ann_date)` 对账，只进 `warnings` 与 `disclosure_checks`，不阻断。测试：`tests/test_pit_semantics.py` 的 `test_disclosure_reconciliation_*` 三个用例。
- **P6**：读取守卫 `filter_available`（`availability.py:196-240`）：按注册表政策过滤、无政策数据集抛 `AvailabilityPolicyError`、日期列缺失抛错、不可解析日期 fail-closed 丢弃；已接入 `scripts/run_multifactor_backtest.py:276,282` 与 `scripts/run_recommendation_refresh.py:390,395` 的临时元数据读取路径。测试：(a) 重述确定性 `tests/test_qlib_builder.py::test_financial_restatement_applies_only_after_the_new_announcement`；(b) available_at 之前读不到 `tests/test_pit_semantics.py::test_financial_fields_are_invisible_before_the_announcement_date`；(c) 无政策被拒 `test_guard_fails_closed_for_unregistered_dataset`。rdagent/research 入口排查：`research_automation.py:196` 只读因子 artifact（自带 datetime 轴与 sha256 lineage），不直接读 snapshot parquet；`market_overview.py` 为盘中观察视图、按 `trade_date` 过滤展示，不作为研究证据读取路径。

## 三、NLP 链路现状

| 环节 | 评级 | 关键证据 |
| --- | --- | --- |
| 公告全文下载 | 已实现；生产范围收敛（2026-08-08） | `src/quant_data/cninfo_announcements.py`：以 `anns_d` 为发现源抓 PDF 正文，内容寻址不可变文件 + 每轮下载日志 + 限速 + 幂等跳过；元数据索引行级 `available_at`（公告次一交易日，fail-closed 依赖 trade_cal）与 `ingested_at`。生产清单复核为 2,526,585 篇，按首批 1,470 份约 1.05GB 的实测均值推算全量超过 1TB，而生产卷当时仅余约 303GB；故生产任务显式限定为监管类高信号正文，不伪装全量公告已覆盖。通用 CLI 仍保留 `--all-announcements` 供具备独立大容量存储的环境使用。 |
| 问询函/监管函 | 已实现（2026-07-18） | 标题关键词分类 `REGULATORY_TITLE_PATTERN`/`categorize_title`（问询函/关注函/监管函/警示函/纪律处分 → `regulatory_letter`，`cninfo_announcements.py:20-81`） |
| LLM/NLP 抽取 | 已实现（2026-07-18） | `src/quant_platform/announcement_nlp.py`：PDF 文本抽取 → LLM 抽取（事件类型、语气分数、关键数值）→ 结构化字段索引，行级 `available_at`/`ingested_at`，原子写、按 sha256+prompt_version+model 幂等、fail-closed |
| 信号字段落地 | 已实现（2026-07-18） | 因子库机制完整（`factor_candidates`/`factor_evaluations`，`database.py:143-256`）；announcement_nlp 因子值 artifact 按 datetime/instrument 单值序列 parquet + sha256 落盘（`announcement_nlp.py:513-549`）；入库通道已接：`announcement_factor_registry.register_announcement_factor`（`announcement_factor_registry.py:181`）校验 manifest sha256 后经 `ResearchStore.add_candidate` 落库，幂等键 (name, values_sha256)（`research_store.py:437`），CLI `quant-db register-announcement-factor`（`db_cli.py:58`） |
| 文本语料消费 | 部分实现（2026-07-18） | 公告正文语料已被 `announcement_nlp.py` 消费；`major_news` 的 `content` 全文（`supplemental_data.py:1086-1098`）与 `research_corpus` bundle 仍无消费方 |

信号落地关闭证据（2026-07-18）：`src/quant_platform/announcement_factor_registry.py` — `_verified_artifact`（:70）fail-closed 校验 manifest（schema、availability_policy、source 身份、parquet 实测 sha256 与 manifest 一致，任何不符在写库前抛错）；`register_announcement_factor`（:181）创建 kind=`announcement_nlp_factor_import` 研究运行（成功 mark `succeeded`、失败 mark `failed`，不占住 active-kind 唯一槽位），生成确定性谱系 code artifact（`_code_artifact_source` :127，内容可由 fields 索引重算因子值），availability_policy/prompt_version/model 写入候选 description+variables_json 现有字段（不加表字段）；幂等键 (name, values_sha256)（`research_store.py:437` `find_candidate`）：同 sha256 重注册返回既有候选不建行，新 sha256 建新候选（候选行不可变，新行即既有版本机制）。测试：`tests/test_announcement_factor_registry.py`（15 个：注册成功、幂等重注册、新版本、缺 artifact/校验和不符 fail-closed、CLI，其中 10 个带 `no_database`）。

## 四、可复用机制清单（开发约束，禁止另起炉灶）

- **待抓取清单**：`anns_d` 已按日全量落盘且含 PDF URL，直接作为公告正文下载的发现源，不另建公告发现机制。生产下载仅选择标题命中问询函、关注函、监管函、警示函或纪律处分的记录；全量 2,526,585 篇是明确记录的存储边界，不计为已下载覆盖。
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

## 七、report_rc 结构化消费（2026-07-20 追加）

Tushare `report_rc`（券商研报盈利预测与评级，cn_institutional 任务按月 3,000 行分页下载，`supplemental_data.py:1060-1070`）此前无消费方；本节记录其结构化因子产出与注册通道的落地：

- **PIT 注册**：`report_date` 为研报发布日期（提供商当日 19-22 点刷新），按 `strictly_after_announcement_date` 同族保守处理（发布次日可用），政策注册于 `availability.py`（`AVAILABILITY_POLICIES["report_rc"]`），recoverability 标注 `native_history`（按 report_date 可回拉 2010 年以来历史）。
- **生产者** `src/quant_platform/report_rc_factors.py`（无 LLM，纯结构化）：读 units/snapshots 双层 parquet（duckdb union_by_name，缺必需列/缺 trade_cal fail-closed），行级 `available_at` = report_date 之后首个 trade_cal 交易日；周频观察网格（ISO 周最后一个交易日）聚合，只计入 grid 日及之前可用的行。产出 `fields.parquet`（含 eps/np/max_price/min_price/quarter 原始预测列，作为预期差类因子的扩展位）、评级变化/EPS 修正事件帧、覆盖度帧，以及三个带 sha256 manifest 的因子 artifact：`report_rc_rating_change`（机构内评级档位变化事件均值，5 档精确映射表 `rating-ladder.v1`，未知评级 fail-closed 不进事件但计入覆盖）、`report_rc_coverage_20d`（近 20 个开放日覆盖研报数，分析师关注度）、`report_rc_eps_revision`（最近预测季度 EPS 相对上修幅度，机构内链，分母下限 0.01 元）。周频网格同时满足外部评估通道 sparse_event 形态门槛（日覆盖 ≤ 50%）。
- **注册通道**：`announcement_factor_registry.register_announcement_factor` 重构为通用 `register_external_factor`（manifest sha256 fail-closed 校验、确定性谱系 code artifact、研究运行谱系、`ResearchStore.add_candidate`、幂等键 name+values_sha256，原公告 NLP 行为不变），report_rc 以 `source.dataset=report_rc` + `producer_version` 身份接入同一通道；CLI `quant-data report-rc-factors`（生产）与 `quant-db register-report-rc-factor`（注册，`--factor-name all` 默认三个全注册）。
- **评估通道**：因子 artifact 直接适配 `external_factor_evaluation` 的 sparse_event 形态（事件日 IC/RankIC、HAC、BH 校正），经 `scripts/evaluate_external_factor_batch.py` 评估落 `factor_evaluations`。
- **测试**：`tests/test_report_rc_factors.py` 29 个（评级映射正/反例、机构内链与匿名/未知评级排除、PIT 发布日不可见、周末顺延、周网格不早于可用日、覆盖度窗口计数与过期、EPS 最近季度与分母下限、端到端 artifact+manifest+确定性重跑、sparse_event 形态兼容与评估、注册 fail-closed/幂等/新版本、code artifact 重算一致、CLI 幂等），全部通过；回归 `test_announcement_factor_registry.py`/`test_external_factor_evaluation.py`/`test_pit_semantics.py`/`test_corpus_nlp.py`/`test_announcement_nlp.py`/`test_supplemental_data.py` 90 个通过，`ruff check src tests` 通过。
- **覆盖度限制**：`report_rc` 仅覆盖有卖方跟踪的个股（A 股约 5,400 只中通常 2,000-3,000 只，小盘/冷门股无覆盖）；历史深度自 2010 年起；评级口径因券商而异（已用机构内比较 + 精确映射表缓解，跨机构绝对档位仍不可比）；本机无已下载数据，真实覆盖统计待凭证就位跑 `quant-data report-rc-factors` 后核实。

## 八、其余语料 NLP 消费与 major_news 个股映射（2026-07-20 追加）

继 corpus_nlp（major_news 市场级 + irm_qa 个股级）与 report_rc 结构化消费之后，本节记录剩余已下载语料的处置决定与落地。原则：复用 corpus_nlp 的 LLM 调用/PIT 时间戳/artifact/注册通道，不发明第二条通道；宁保守勿滥造。

### 各语料处置决定

| 语料 | 决定 | 理由 |
| --- | --- | --- |
| `major_news`（个股级映射） | **接入**（确定性方法，零新增 LLM 调用） | content 全文里识别被提及的上市公司，复用 corpus_nlp fields 已有的逐条 LLM 情绪，映射到个股形成稀疏事件因子 |
| `news`（9 源快讯） | **接入，但只做聚合特征** | 快讯与 major_news 报道同一批市场事件，再做逐条 LLM 情绪与市场级 `news_sentiment_daily` 高度重复且成本高出一个数量级；快讯流真正的增量是其**量能动态**（major_news 是编辑筛选过的薄信息流，无法度量），故只产确定性的快讯强度因子，不产第二个情绪分 |
| `npr`（国家政策法规库） | **接入**（LLM，市场级） | 一手政策文本，与新闻报道不重复（是新闻的上游源头）；`pubtime` 精确发布时间。注意 `content_html` 非接口默认字段，下载通常只有标题+发文字号+发文机关，故实为标题级处理——法规/批复标题本身是信息主体，可辩护 |
| `cctv_news`（新闻联播） | **接入**（LLM，市场级，与 npr 合并为一个政策因子） | 政策信号语料，date-only 保守顺延次一交易日；与 major_news 有部分重叠但作为独立政策通道仍有增量，优先级低于前两项 |
| `wc_list` / `wc_cnt`（微信） | **跳过；全历史 unavailable（2026-08-08 复核）** | 接口 schema 决定不可用：`wc_list` 为公众号文章列表（id/title/pub_time/url 等，无正文），`wc_cnt` 为阅读计数（sn/account/publish_time/title + 计数，无文本）；两者均无个股键，需单独权限。2026-08-08 复核时，官方当前数据接口目录不再提供这两个入口，生产网关对既有请求返回上游参数错误；因此不伪造覆盖、不再作为 `research_corpus` 必需下载项，旧工作单元仅保留审计。标题级处理与快讯/要闻重复且 PIT 依据弱。 |
| `research_report`（研报文本） | **跳过 LLM** | 同一批研报的评级/EPS/目标价已由 report_rc 结构化消费（第七节）；`research_report` 只有标题+摘要（`abstr`）无正文，摘要情绪与评级强相关、增量低，而 LLM 成本为每日数百条。记录结论，不硬做 |

### 新因子清单

| 因子 | 形态 | 来源 | 说明 |
| --- | --- | --- | --- |
| `major_news_mention_sentiment_daily` | sparse_event | major_news × stock_basic/namechange | 提及该股的长篇新闻 LLM 情绪均值（复用 corpus_nlp fields，无新增 LLM 调用） |
| `major_news_mention_count_daily` | sparse_event | 同上 | 当日提及该股的distinct新闻条数（新闻关注度变量） |
| `news_flash_intensity_daily` | market_timeseries（MARKET） | news（9 源快讯） | 当日 15:00 截点前快讯数 / 此前 20 个开放日均值（min 5 天历史，严格只用历史日，无 look-ahead） |
| `policy_sentiment_daily` | market_timeseries（MARKET） | npr + cctv_news | 政策语料 LLM 情绪日均值（corpus_nlp 同一 pipeline 产出） |

### 个股映射消歧策略（`mention-rules.v1`，`src/quant_platform/major_news_mentions.py`）

确定性规则，不用 LLM 抽取（成本与可复现性考虑）：

1. **别名候选**：`stock_basic.name`（简称，自 `list_date` 起有效）+ `stock_basic.fullname`（全称，下载含该列时）+ `namechange` 历史名及其 `[start_date, end_date]` 有效区间。**按新闻 pub_time 当日的有效别名匹配**——用 PIT 的名字而非今天的名字，避免改名股的前视/幸存者偏差。
2. **高歧义直接放弃**：别名长度 < 3 字符一律不用。"平安"类两字简称正是高歧义情形，漏掉好于猜错；匹配字符串是完整简称为子串（"中国平安"四字命中才映射 601318.SH，单独"平安"二字不会误命中）。
3. **跨股冲突丢弃**：同一别名串映射到多个 ts_code 且有效区间重叠 → 对所有涉事股票整体丢弃该别名；区间不重叠的序贯复用保留。
4. **匹配规则**：从左到右扫描，每个位置取最长匹配别名，命中区间消费不重叠；同一股票在一条新闻里只记一次提及。

### PIT 注册（`availability.py`）

- 精确时间戳语料 `major_news`（pub_time）/`news`（datetime）/`npr`（pubtime，兼容 pub_time 列名）：`same_trade_date_after_close`（注册表为整日粒度，与平台"决策日在收盘后"的既有约定一致；生产者在因子日期上另施加 15:00 细粒度截点规则）。
- date-only 语料 `cctv_news`（date）/`irm_qa_sh`/`irm_qa_sz`（trade_date）与评估后跳过的 `research_report`（trade_date）：`strictly_after_announcement_date`（保守顺延）。
- recoverability 均标注 `native_history`（按发布日期/时间可回拉历史，provider 不改写旧行）。

### 落地与接线

- **corpus_nlp 扩展**（`src/quant_platform/corpus_nlp.py`）：新增 `_npr_items`/`_cctv_news_items` 归一化（pubtime/pub_time、content_html/content 列名按行 coalesce 容错；npr 无正文时标题级回退），`available_at_for` 按精确/仅日期两类分派，新增 `build_policy_sentiment_series` 与 `policy_sentiment_daily` artifact；新增通用 `register_corpus_factor`（三个 corpus 因子统一走 `register_external_factor`，补上了 news_sentiment_daily/irm_qa_sentiment_daily 此前没有注册入口的缺口）。
- **major_news 个股映射**（`src/quant_platform/major_news_mentions.py`，无 LLM）：aliases.parquet（审计用别名表）+ events.parquet（含 matched_alias、行级 available_at/factor_date）+ 两个因子 artifact + `register_major_news_mentions_factor`。
- **快讯强度**（`src/quant_platform/news_flash_factors.py`，无 LLM）：daily_counts.parquet（范围内每个开放日零填充，作为谱系重算的精确输入）+ `news_flash_intensity_daily` artifact + `register_news_flash_factor`。
- **CLI**：`quant-data major-news-mentions`、`quant-data news-flash-factors`（生产）；`quant-db register-corpus-factor`、`register-major-news-mentions-factor`、`register-news-flash-factor`（注册，`--factor-name all` 默认全量，幂等）。

### 测试

- `tests/test_corpus_nlp.py` 扩展至 45 个：npr/cctv 加载正反例（列名变体、坏行丢弃、缺列 fail-closed）、`available_at_for` 新语料规则、端到端 7 条语料 + policy 因子 artifact/manifest、注册 fail-closed（未知名/缺 manifest/篡改/来源身份）与 code artifact 重算一致。
- `tests/test_major_news_mentions.py` 20 个（16 no_database + 4 真实 PG）：别名表规则（全称/历史名/最短长度/跨股冲突/序贯复用）、最长匹配与单次提及、PIT 有效区间正反例（过期别名不匹配）、端到端事件与因子值、确定性重跑 sha256 一致、无成功抽取行不产生事件、缺输入 fail-closed、sparse_event 形态兼容、注册成功/幂等/篡改不落库/CLI 幂等。
- `tests/test_news_flash_factors.py` 19 个（16 no_database + 3 真实 PG）：15:00 截点映射、零填充网格、严格历史窗口均值、最小历史天数、零快讯日不出值、端到端 artifact+确定性、market_timeseries 形态兼容、availability 注册与 `filter_available` 执行、注册成功/幂等/CLI。
- `tests/test_corpus_factor_registration.py` 4 个（真实 PG）：corpus 注册成功/幂等/篡改不落库/CLI 幂等。
- 回归：`test_report_rc_factors.py`/`test_announcement_factor_registry.py`/`test_external_factor_evaluation.py`/`test_pit_semantics.py` 等 no_database 子集通过，`ruff check src tests` 通过。

### 限制与说明

- 提及情绪是**整篇新闻的情绪**按提及传播（一篇提到三只股票则三者同分），不是针对每只股票的目标级情绪；目标级抽取需要新一轮 LLM 调用，本轮刻意不做。
- `stock_basic` 为 current_only 快照：未改过名的股票其当前名按 `list_date` 起有效处理（近似，依据是未改名则名称自上市起有效）；已改名股票由 `namechange`（native_history）区间覆盖。若 `namechange` 未下载则回退仅用 stock_basic。
- 快讯强度因子的"零快讯日不出值"把缺数据与真零区分不开，按保守处理（不出值而非填零）；零仍进入后续分母。
- 本机无已下载真实语料，真实覆盖率/别名命中率统计待凭证就位后跑 `quant-data corpus-nlp && quant-data major-news-mentions && quant-data news-flash-factors` 核实。

## 九、Known limitations（经评估后明确不修复）

以下两项于 2026-07-20 评估后确认对 A 股中低频主线无实际影响，记录为已知限制而非待办缺口：

1. **指数权重/行业成员可用时间 = 生效日 + 5 自然日近似**（`availability.py` 的 `effective_date_with_lag(days=5)` 政策）。指数公司与中证/申万没有机器可读的真实发布时间源，5 日滞后是保守近似；权重调整对中低频组合的影响远小于这个时滞误差，除非接入付费数据源，否则无修复路径。
2. **分钟线覆盖与非复权**：A 股分钟数据只有 5min 全市场（`stk_mins`），1min 仅限流动性股票子集；分钟线不做复权（`minute_qlib_builder.py`）。分钟线只服务执行细节研究，不在"日线中低频因子 → 回测 → 模拟"主线上，不补。

若未来策略频率提高到日内或接入付费指数数据源，再重新评估这两项。

## 十、Barra 风格结构化风险模型（2026-07-21 追加）

平台原有风险能力只有 qlib metadata 中 size（log 市值）单风格暴露 + qlib `ShrinkCovEstimator` 收缩协方差。本节记录 Barra CNE5/6 简化版结构化风险模型的落地，目标是服务因子评估中性化与（将来的）组合优化；定位中低频月度/周度再平衡，不追求高频精度。

- **风格暴露扩展**：新增 `src/quant_data/style_exposure_panel.py`（原始描述子面板，纯快照输入、无平台依赖）与 `src/quant_platform/style_exposures.py`（截面标准化，纯 pandas/numpy）。风格集共 9 个：size（log 市值）、nonlinear_size（size³ 对 size 正交化，Barra NLSIZE）、value（BP=1/PB 与 EP=1/PE_TTM 等权合成）、momentum（过去 252 交易日收益剔除近 21 日）、volatility（120 日收益波动率，最少 60 个观测）、liquidity（21 日平均换手率的 log）、growth（fund 通道营收/净利同比等权合成）、profitability（ROE）、leverage（资产负债率）。标准化管线为 Barra 经典三段：MAD 去极值（±3 个 1.4826 缩放 MAD）→ 流通市值加权 z-score → 对 size 做市值加权回归取残差正交化后再 z-score；行业正交刻意不做（结构化风险模型自带行业因子，避免双重中性化）。横截面不足 2 个有效观测或加权方差为零时输出 0.0（加权均值），保证单行 metadata 有限。
- **PIT 纪律**：量价/估值/换手描述子来自 daily_basic 与复权收盘，沿用 `same_trade_date_after_close` 语义（t 日暴露服务 t 日收盘后决策）；基本面描述子走 `fina_indicator` 公告日 ASOF 通道（`trade_date > ann_date` 严格晚于公告日，`end_date` 永不参与可用性）；行业暴露用申万 L1 `industry_memberships` 的 in/out_date 区间，经 `filter_available` 平台可用性守卫（生效日 + 保守发布滞后）消费。
- **qlib metadata 集成**：`qlib_builder.py` 的 `_build_style_exposures` 在保留历史 `log_market_cap` 原始列的同时追加全部标准化风格列（向后兼容的 schema 扩展），同次构建顺带产出 `full_market_weights.parquet`（全市场流通市值权重，供因子收益回归的市值加权用）。消费方（portfolio_policy / portfolio_optimizer / strategy_backtest）读取原列不受影响。
- **结构化风险模型**（`src/quant_platform/risk_math.py`，`barra-lite-cne6-v1`）：因子收益用逐日截面市值加权最小二乘（市场因子 + 风格 + 行业 dummy，最大行业为参照类；等权回退）估计；因子协方差为因子收益序列的指数加权协方差（半衰期 90 交易日），特征值裁剪到 PSD；特质风险为各股回归残差的同半衰期 EWMA 方差，历史不足 20 个观测（新股/长期停牌）收缩到截面中位数并设下限 `1e-8`。产出 `StructuredRiskModel` artifact（exposures/factor_covariance/specific_variance 三件分开可取，组合优化器直接消费因子结构），`covariance()` 给出稠密 `B F B' + D`（`estimate_covariance` 的 drop-in），`portfolio_risk()` 做组合波动率与市场/风格/行业/特质四段风险分解，`save/load` 持久化（parquet + JSON manifest，版本校验）。行业 dummy 缺失的个股降级为"风格 + 特质"风险而非失败。
- **与 ShrinkCov 路径的关系**：`estimate_covariance`（qlib `ShrinkCovEstimator`，alpha=0.10/const_var，钉住 qlib commit 校验）原样保留，作为样本协方差对照与回退；结构化模型是并行新增路径，两者接口（输入收益率帧 → 协方差 DataFrame）对齐，可互换。
- **测试**：`tests/test_style_exposure_panel.py`、`tests/test_style_exposures.py`、`tests/test_structured_risk_model.py` 共 29 个全部通过（小手算数据集验证暴露值、PIT 未来数据不进暴露/行业成员区间正反例、z-score 加权均值方差与 size 正交性、因子协方差正定性、组合风险端到端、ShrinkCov 回退共存、artifact 往返）。回归 `test_qlib_builder.py`/`test_portfolio_policy.py`/`test_portfolio_optimizer.py`/`test_strategy_allocation_math.py` 45 个通过；`ruff check src tests` 通过。`test_strategy_allocation_recommendations.py` 的唯一用例需真实 PG（127.0.0.1:55432），本次运行环境 PG 不可达（连接超时），与本次改动无关。
- **已知局限**：① 新股/长期停牌股特质风险历史不足时收缩到截面中位数（不区分行业/风格特征，粗糙但保守）；② 行业因子用静态 L1 dummy，未做行业因子收益的多重共线性处理（参照类 + 满秩检查跳过退化日）；③ EWMA 半衰期 90 日为固定参数，未按因子类型分层；④ 组合风险接口当前不做交易成本/约束优化，优化器为后续独立开发项；⑤ 本机未对全量真实数据跑过结构化模型构建，真实覆盖率/因子收益平稳性待数据就位后验证。

## 十一、逻辑面与市场认可度闭环（2026-08-08 追加）

- **逻辑面不是自由文本猜测**：`announcement_nlp` 升级为 `announcement-nlp.v2`，在原有事件类型、语气、关键数值之外，增加固定枚举的 `impact_direction`、`impact_horizon`、`impact_channels` 与最长 240 字的证据约束摘要。未知、混合或材料不支持时必须输出 `uncertain`/空渠道/空摘要，不允许补写公告没有的事实。
- **可解释逻辑因子**：新增 `announcement_logic_score`，公式固定为 `direction_enum_score × horizon_enum_weight × confidence`，再按 `(available_at, ts_code)` 求均值。方向与期限权重是代码内冻结映射，不直接采用 LLM 自报的任意数值；因子严格只读公告当时可见字段，完全不读事件后的价格。
- **市场认可度只作标签**：新增 `event_market_response.py`，基于已通过阻断质量门的不可变日线快照，计算公告首个可用交易日起 1/3/5/20 日的个股收益、沪深300收益、超额收益、成交额异常和“公告方向 × 超额收益”的一致性。每个期限均记录 `outcome_end` 与 `label_available_at`；停牌、缺价或样本尾部不足时保持 NULL，不顺延猜测。
- **防泄漏硬隔离**：标签 manifest 固定 `role=training_label_only`，并明确禁止 `factor_candidates`、Qlib 实时推理特征和 live signal 消费。标签生产器没有因子注册入口；只有公告逻辑因子可通过现有 sha256 治理通道注册。
- **可运行任务**：新增 `announcement_nlp`、`corpus_nlp`、`event_market_response` 三类 durable worker 命令、API 入队端点和数据目录任务映射。事件标签任务要求指定快照且 `verification.json.ok=true`，公告 NLP 在公告下载仍运行时拒绝重复启动。
- **Qlib 评估接线**：新增 `external_factor_evaluate` durable job/API。任务只接受带外部来源身份、代码/数值 artifact 与 sha256 的候选，绑定可复现日线 Qlib 数据集身份、训练/验证/最终测试窗口及 purge/embargo 间隔；复用 `evaluate_external_factor_batch.py` 的 sparse-event/market-timeseries、HAC、BH-FDR、成本和泄漏哨兵评估，并把通过、证据不足和运行失败全部写入 `factor_evaluations` 审计账本。
- **限流与恢复**：Tushare 中转站按运营方最新确认改为 99 次/分钟上限（100 次限制保留 1 次余量）；巨潮正文长任务每 100 条原子刷新 index/download-log 并更新 job progress，避免只在任务结束时才出现 checkpoint。公告/语料 NLP 同样每 100 条写入不可变 field unit、合并 fields/state 并刷新 durable job progress；进程中断后按 process key 跳过已成功调用，不重复消耗 LLM。无可用文本层的公告 PDF 记录为 `source_unavailable` 来源缺口，LLM 调用/解析失败仍使任务非零退出，禁止把部分成功伪装成完整成功。
- **覆盖边界**：逻辑与市场认可度只能覆盖真实落盘且有文本层的公告。2026-08-08 全量扫描确认当前 `anns_d` 可追溯的监管公告最早为 2016-01-01，并不存在 2008–2015 的正文发现清单；因此信息面的可信边界记为 2016-01-01，不向前伪造。扫描件无文本层时 fail-closed，不用模型虚构字段。
- **旧近年范围下载闭环（2026-08-08）**：此前 2024 年以来监管类子范围清单共 20,687 条，成功请求 17,594、已有文件跳过 2,602、源站不可用 491；491 条逐项核对均为 HTTP 404，未发现 429、5xx、网络错误或非 PDF 响应。该数字只证明旧近年子范围，不代表 2016 年以来全历史。`cninfo_announcements` 将 404/410 记入 `source_unavailable.parquet` 审计墓碑而不是伪装成功，30 天后才重新探测；重复运行在冷却期内不再浪费请求。非 404/410、日历缺口、内容校验失败仍计入 `failed`，CLI 返回非零退出码使 durable job 真实失败，避免“有错误但任务显示成功”。
- **公告正式质量门（2026-08-08）**：公告下载命令结束后自动按同一 discovery 范围生成 `announcements/quality/quality_*.json` 与原子更新的 `latest.json`。验收逐项覆盖 URL（正文或明确 404/410 墓碑）、严格复算公告次一交易日 `available_at`、核对来源元数据和内容寻址路径、PDF magic、字节数及完整 SHA-256，并记录 index/tombstone artifact 哈希、真实发现边界、缺口样本和告警。元数据存在但文件缺失/篡改、PIT 漂移、重复 URL、无墓碑缺口均阻断 durable job 成功；合法源站缺失只作为明确告警，绝不伪造正文。
- **历史扩展任务（2026-08-08）**：新增存储扩容后，以同一监管标题治理规则扫描 2016-01-01–2026-08-03，共识别 164,633 条监管类 URL；已有索引覆盖约 20,196 条，尚缺约 144,437 条，按已落盘 PDF 的 1.94MB 实测均值估算新增约 261GB。生产 durable job `fc9536fbbbc44c8ea3534ed36c67bf24` 已从 checkpoint 启动补齐；完成前不把该区间标为可用。
- **历史正文兼容边界**：部分 2016 年 `static.cninfo.com.cn`/`dataclouds.cninfo.com.cn` 链接以非标准 multipart form-data 包裹真实 PDF。下载器仅在响应中同时找到完整 `%PDF-` 至 `%%EOF` 载荷时剥离传输包装，否则继续 fail-closed；detail URL 转成静态 PDF 后再次去重，避免同一正文重复请求。该兼容修复已推送，当前长任务结束前不热部署，失败 URL 在安全窗口部署后幂等补跑。
- **语料真实边界（2026-08-08 生产审计）**：`major_news` 共 3,543,392 行，真实时间范围 2018-11-20–2026-07-29；`cctv_news` 13,230 行，范围 2024-01-01–2026-08-02；`irm_qa_sh` 274,022 行、`irm_qa_sz` 570,009 行，均从 2024 年开始；`npr` 当前无落盘行。故 `cn_corpus_nlp` 的目录边界改为 2018-11-20，并明确依赖承载 `major_news` 的 A 股全量数据任务；不存在的 `npr` 不伪造覆盖。有限 `limit` 试跑或起点晚于上述边界的任务保留在 jobs 审计表，但不得把数据目录能力标成全量成功。
- **语料默认范围（2026-08-08）**：`corpus_nlp` 的生产默认集为 `major_news`、`cctv_news`、`irm_qa_sh`、`irm_qa_sz`；`npr` 仍是显式支持的可选来源，但在真实数据落盘前不作为默认必需项。因子 manifest 只记录本轮实际请求的来源，禁止把未参与的 `npr` 写进谱系。日期窗口直接下推到 DuckDB/Parquet 扫描，避免先把 440 万级全文载入内存后再过滤。
- **NLP 因子发布范围治理（2026-08-08）**：`fields.parquet`/`state.parquet` 继续保留不同 prompt、模型和历史试跑范围的行用于审计与幂等恢复，但公告与语料因子发布只读取“当前 prompt + 当前模型 + 本轮请求 item/process key 集合”。manifest 写入日期、证券、类别、limit、计划条数及 process-key 集合哈希；有限试跑、类别子集或旧 prompt 结果不得混入本轮因子，也不得通过跨版本平均悄然改变信号。
- **语料选择覆盖审计（2026-08-08）**：语料任务在 DuckDB 内对日期/证券范围中的去重有效文本计数，不把未抽样的 440 万级全文载入 Python；结果与每个因子 manifest 分来源记录 `eligible_unique_items`、真实日期边界、治理抽样前后条数、全局 limit 后条数及选择率。稳定哈希排序只使用来源文本与时间，不读取事件后收益；因此抽样预算透明、可复现，且不会将经过抽样的数据声称为全量 NLP 覆盖。
- **语料批处理（2026-08-08）**：生产 CLI/API 默认每次把 40 条语料合并为一次 LLM 请求，每条携带内容哈希 `item_id`，返回必须逐条、无遗漏、无重复地匹配全部 ID；任何错配整批 fail-closed，重跑仍按 `item_id+prompt_version+model` 幂等恢复。批内单条文本上限 1,000 字，避免上下文无界增长；需要诊断时可显式设 `batch_size=1` 回到逐条模式。这样 440 万级语料的理论请求数由约 440 万降至约 11 万，同时不伪装未处理行。
- **高密度语料选样（2026-08-08）**：为避免把同日上千条转载新闻全部当成独立信息，`major_news` 按发布日期用稳定内容哈希排序，默认每日最多 40 条；互动易/上证 e 互动按 `(来源, 日期, 股票)` 默认最多 2 条。选样在 DuckDB 扫描阶段完成，规则与上限写入每个因子 manifest；`0` 可显式关闭上限。低于生产上限的试跑不得认证完整目录能力。该规则控制的是情绪估计的代表性样本，不删除原始语料，也不把未选文本伪装成已做 LLM 抽取。

## 十二、资金面进入 Qlib / RD-Agent（2026-08-08 追加）

- **真实数据边界**：资金面使用 Tushare 官方 `moneyflow` 个股逐日主动买卖单数据；官方文档声明接口从 2010 年开始、金额字段单位为万元。2026-08-08 对生产不可变快照 `cn-20080101-20260803-20260808T033243Z` 的实测为 10,630,216 行、5,649 只证券、最早 2016-01-04、最晚 2026-07-29，关键金额列均有值且 `(ts_code, trade_date)` 无重复。生产覆盖只认 2016-01-04 起的真实落盘范围，不把官方理论起点或 2008 年补成伪数据。
- **紧凑资金面通道**：`qlib_builder.py` 研究字段契约升为 v5，新增三个 Qlib 字段：`mf_net_inflow_amount = net_mf_amount × 10000`（元）、`mf_net_inflow_ratio = net_mf_amount × 10 / daily.amount`（净流入/当日成交额）及 `mf_large_order_imbalance = (大单+特大单主动买入额 - 主动卖出额) / 主动买卖总额`。不把九个高度共线的原始分档列全部灌入模型。
- **PIT 与质量门**：`moneyflow` 注册为 `same_trade_date_after_close`、`native_history`；只与同股票同交易日行情连接，供当日收盘后的下一决策/下一 Bar 使用。缺少数据集、主键、关键金额列或任何可用行时 Qlib 构建直接失败；来源列缺失和全 NULL 分开记录在 provenance，金额统一换算为元并写入字段单位。
- **RD-Agent 约束**：全市场行业中性配方升级为 `qlib-rdagent-single-mainline-2026-08-08-v4`，明确允许研究上述资金面字段；公告、研报、新闻、情绪与逻辑因子仍只能走受治理外部因子评估/晋级 artifact，事件后市场认可度只能作训练标签，禁止写回实时特征。
