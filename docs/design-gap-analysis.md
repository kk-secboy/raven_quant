# 设计稿 × 代码对表（2026-07-21）

基准：《个人量化投资与模拟盘系统设计稿》全部 14 节；代码基线 `259549a`
（worktree 分支 `design-gap-analysis`）。状态词：✅已实现 / 🟡部分实现 /
❌缺失 / ⛔冲突 / ❓无法静态判定。原则：补缺 > 修正 > 重构；测试绿不算
豁免理由。

## 0. 总览

- 四路并行对表覆盖 §1-§14，产出约 180 条合同核对记录。
- **⛔ 冲突 8 项**（其中 2 项需用户决策，见第 1 节）。
- **❌ 缺失约 30 项**，集中在：§6.10/§8 账户层抽象、§4.5/§6.11 paper
  阶段与前向证据门、§4.3/§6.8 反过拟合工具链、§10 运行日历、§8.6 影子
  账户、§11.3 safe_mode、§4.4 TWR/外部现金流。
- 数据层（§3）与成本/模拟账本（§5/§9 主体）质量最高，绝大多数合同
  ✅；这与此前审计结论一致：问题主要是"没写"，不是"写错"。

## 1. 需用户决策的冲突（本 goal 不自行处理）

### 1.1 ⛔ QMT 券商写网关 vs "永久不授权"

- 合同：§1.3:49「不连接券商写接口」；§13:1161「券商写入、自动下单、
  自动撤单和资金控制永久不在本设计授权内」；§11.3「系统代码不提供券商
  写入入口」；§8.4「永远不替用户撤销券商客户端真实委托」。
- 代码：`src/quant_broker_gateway/` 实现完整 QMT 写链路——
  `qmt.py:57-94` 真实下单、`qmt.py:168-187` 撤单、`app.py:106`
  `POST /v1/orders`、`store.py:122-146` 超时自动撤单重报。
- 缓解：`config.py:63` `broker_feature_enabled=False` 默认关闭、HMAC
  鉴权、sandbox 环境断言、Fake adapter 测试。但写入口的存在本身即违反
  字面合同，且无测试扫描证明该路径不会向真实账户发单。
- 决策选项：A) 修订设计稿把"默认关闭的 sandbox 网关"纳入合同；B) 按
  设计稿移除网关代码。README 称 QMT 为"默认关闭的可选插件"，说明现状是
  有意为之，倾向 A，但须用户定夺。

### 1.2 ⛔ 股票配对策略全链路上线 vs "永久 research_only"

- 合同：§13:1157「`stock_pair_stat_arb` 只保留离线关系研究；即使取得
  借券数据也不能进入正式双腿回测、持久模拟或推荐。未来若确实要做，
  必须另立完整卖空账户与双腿执行专项设计」。
- 代码：全链路已上线——`pair_trading.py:347` 原生双腿回测（含借券费
  /可空性强制）、`strategy_store.py:1014-1167` 双人批准门、
  `simulation_store.py:71,459-505` 持久双腿模拟入账、
  `allocation_store.py:162-166` 进入卫星分配。
- 注：设计稿 v1.1（2026-07-19）可能晚于配对代码；代码侧已有相当投入。
- 决策选项：A) 修订设计稿承认配对为已实现的专项能力（需补齐召回/受限
  回款等卖空语义）；B) 按设计稿降级为仅离线研究（下线批准门/模拟/分配
  入口，保留研究代码）。须用户定夺。

## 2. 冲突（可直接修正，本期处理或登记）

### 2.1 ⛔ 唯一 ExecutionCore 不存在，回测/模拟两套成交语义（§2.3、§9.3、§12.3）

- 合同：单一轻量纯规则 ExecutionCore 统一成交判定，两条适配链黄金案例
  +差分测试。
- 现状：正式回测由 Qlib 原生 Exchange 撮合（`qlib_exchange.py:16-21`
  自述），模拟用 `simulation_engine.execute_simulation_day` 自研下一
  Bar 逻辑；仅共享成本模型、整手、公司行动原语。差分测试完全缺失（仅
  `test_document_governance.py:673` 校验设计稿文本含此要求）。
- 影响：「回测成绩=模拟成交」无任何测试保障，回测→模拟差异无法归因。

### 2.2 ⛔ 最终样本外一次性消费存在跨 campaign 重开漏洞（§4.1、§12.1）

- 合同：OOS 一经打开，`first_opened_at` 后形成的候选不得冒充；改名研究
  活动/谱系不可规避；`oos_vintage_id/sealed_candidate_set` 联合密封。
- 现状：消费标记挂在 `factor_evaluations` 行（`strategy_store.py:
  879-893`），新 campaign 的新评估行可对同一日历窗口再次预约消费；
  `oos_vintage/sealed_candidate_set/candidate_formation_ts` 全部缺位。
  缓解：同 program+dataset 只许一个 campaign、final 失败禁重试。
- 已修正（2026-07-21）：迁移 `0040_oos_vintage_sealing` 新增
  `oos_vintages` 表，vintage 键 =（scope=不可变 research_program_id
  （无 program 谱系时退化为数据集身份）+ dataset_identity + 最终测试
  日历窗口），不含 campaign/谱系/策略名等可换皮字段；首次消费时密封
  `sealed_candidate_set` 并落 `sealed_at/first_opened_at/consumed_at`。
  `StrategyStore.create_backtest` fail-closed：vintage 已密封而候选不在
  密封集合 → 拒绝；已 consumed → 任何候选（含新 campaign 的新评估行、
  同候选的新版本）再次消费一律拒绝。同一候选按现行规则首次消费不变。

### 2.3 ⛔ RD-Agent 沙箱物理可见最终样本外（§4.1）

- 合同：RD-Agent 环境物理上不能读最终样本外。
- 现状：整个数据集（含 test 段）只读挂载进沙箱
  （`rdagent_runtime.py:230-236`），test 边界仅作环境变量。缓解：Docker
  隔离+只读+评估证据绑定窗口。另：`worker.py:131` 以
  `env={**os.environ,**extra_env}` 启动包装进程，宿主环境变量全量继承。

### 2.4 ⛔ Qlib 评估失败的候选被静默丢弃（§4.2、§6.6）

- 合同：失败/超时/放弃试验全部入账不能删。
- 现状：`worker.py:1708-1710` `if item.get("status") != "ok": continue`
  不产生任何 evaluation 记录。
- 已修正（2026-07-21）：新增 `ResearchStore.record_failed_evaluation`，
  失败/超时/中止的评估落 `factor_evaluations` 行（`gate_status=
  evaluation_failed`，错误摘要 + 运行上下文哈希入
  `recompute_evidence_json`），候选状态置 `evaluation_failed`，与
  `gate_failed`（评估完成但未过门）语义区分；晋升门（promote 要求
  latest gate passed）与确定性排名（`gate_status == "passed"`）天然把
  它当不合格而非无记录；失败行永久留痕，候选可再评估恢复。
  `worker._import_factor_evaluations` 不再跳过失败项，部分失败批次在
  标记 run 失败前也会把 ok/failed 全部入账；
  `external_factor_evaluation.import_external_evaluations` 同步修正，
  无法归属到已入账候选的失败项直接抛错（fail-closed）。

### 2.5 ⛔ paper 阶段与前向证据门整体缺失（§4.5、§6.11、§7.4）

- 合同：`candidate→paper` 硬门后自动发生；`paper→recommendation_enabled`
  需预注册前向证据门（最少自然时间/独立决策/完整持有周期/数据完整率/
  对账率/成本偏差）+人工批准。
- 现状：无 `paper`/`recommendation_enabled` 状态；最终回测硬门→
  `awaiting_approval`→人工 approve→直接进推荐。唯一前向证据（5 天人工
  复核 NAV）只挂在 allocation 审批上。

### 2.6 ⛔ 股息税负债缺失导致 NAV 阶段性虚高（§5.6）

- 合同：税额未定须记保守税费负债或估值不确定性，后续只确认差额。
- 现状：at_sale 税制应收按税前全额入账（`corporate_actions.py:381-393`），
  不记应付税负债、不标 valuation_uncertain，卖出时才扣税
  （`simulation_engine.py:404-427`）→ 除权至卖出间 NAV 最高虚高股息 20%。
- ✅ 已修正（2026-07-22）：除权确认应收（税前）同时按批次以除权时点
  持有期档位计提 `liability_per_share`（保守上界：持有期只增、税率只降），
  NAV = 现金 + 市值 + 应收 − 未结算负债（`pending_dividend_tax_liability`
  按 untaxed_quantity×每股负债实时重算）；卖出按消耗量同比例释放负债，
  只确认实际税额与已提的差额（多提冲回/旧账本少提补扣），到账重分类不
  动负债；迁移 `0041_dividend_tax_liability`（entitlements.liability_
  per_share / actions.tax_liability_amount / nav.corporate_tax_liabilities，
  旧行默认 0 向后兼容）。

### 2.7 ⛔ 多 allocation 可同时 active；推荐发送者无唯一约束（§6.10、§9.1、§8.1）

- 合同：同一时刻唯一 `active_account_allocation_policy_id`；
  唯一 `active_recommendation_account_id`。
- 现状：`allocation_store.py:519-595` set_status 无全局唯一约束；
  `recommendation_store.py:80-96` 无单发送者指针。

### 2.8 ✅ 成本压力只有"全项×2"（§7.3）（2026-07-22 已解决）

- 合同：佣金/价差/滑点/冲击/成交率分别恶化；×2 不能替代逐项压力。
- 解决：`CostModelConfig.scaled`/`CostScheduleBook.scaled` 逐项缩放；
  `qlib_backtest.py` 新增 component_cost_stress 补充情景块
  （commission_2x / slippage_2x / impact_2x / fill_rate_75pct，
  double_cost 保留为聚合情景）；审批门在原有四情景之外另要求逐项块
  全过且 artifact 完整（strategy_store.py）。

### 2.9 ⛔ 盘中分钟股票池按全市场流动性扫描（§3.1:130）

- 合同：盘中只拉持仓/待执行目标+冻结白名单，不扫描全市场。
- 现状：`universe.py:134-151` 按近 20 日成交额全市场排名前 N 选股；
  `forward_paper`/`recommendation_enabled` 概念在 src 零命中。

## 3. 缺失清单（按 12.6 阶段组织）

### 阶段 3 正式验证（本期优先级 2）

- ❌ Walk-forward 外层 Fold 每折重跑内层选择（全库无 outer-fold 概念）
- ✅ 负对照（随机/打乱标签）与泄漏哨兵（未来字段/标签平移/同 Bar 成交）
  （2026-07-22：`external_factor_evaluation.py` placebo-leakage-sentinel-v1，
  固定种子打乱标签对照分布 + 因子 ±1 日平移坍塌检测，诊断标记入评估记录不
  硬拒；同 Bar 成交侧由执行层 open 成交价 + 涨跌停阈值既有约束覆盖）
- ❌ 消融实验（移除组件的增量证据）
- ❌ 信号衰减曲线（延迟重执行实验，决定提醒有效期）
- ❌ Holm 校正、PBO、区块 Bootstrap、同区间成对比较（BH/DSR 已有）
- ❌ 延迟（latency）压力情景；退市/数据延迟专项情景
- ❌ OOS 密封全套（见 2.2）；解冻密封记录
- 🟡 embargo 天数为固定下限 max(5,horizon)，未按信息传播论证
- ✅ HAC 零方差时输出显式"未定义"（2026-07-22：
  `statistical_validation.py` undefined_zero_hac_variance，statistic/p_value
  为 None 而非 inf；下游门按证据不足处理，BH 校正将 None 垫为 1.0）
- 🟡 event_stress 按已实现收益最差窗口选取，字面违反"市场阶段不得事后
  挑选"（确定性偏保守，建议登记为已知偏差或改为预注册窗口）
- ❌ 「重跑一致」可复现实测（五种结论区分之一）
- 🟡 现金收益参数（cash_yield）与 XIRR 仍缺；恢复期指标已完成
  （2026-07-22：`qlib_backtest.py` max_drawdown_recovery_days /
  max_drawdown_recovery_status，recovered/ongoing/no_drawdown 三态）

### 阶段 5 账户政策（本期优先级 1）

- ✅ `AccountAllocationPolicy` 对象：冻结决策日才产 `AllocationArtifact`
  （decision_frequency/valid_until/输入 as_of），其他任务复用不临时重估
  ——已完成（2026-07-21）：迁移 `0039_allocation_policy_guards`（
  `strategy_allocation_artifacts` 表 + `decision_frequency` 列）+
  allocation_store.py refresh 冻结决策日门（未到期复用、到期重估、重估
  失败沿用旧预算）
- ✅ 唯一 active 政策约束（见 2.7）——已完成：迁移 `0039` 部分唯一索引
  + 激活路径 fail-closed；推荐组合按 `recommendation_scope` 区分
  （standalone 发送者唯一，allocation_member 合法多 active）
- ✅ 账户级证券净额合并层——已完成（2026-07-22）：新模块
  `src/quant_platform/account_netting.py` + 迁移 `0042_account_netting_plans`。
  固定顺序：预算应用一次→各策略目标→证券级净额（同证券代数合并、反向
  部分/完全抵消只交易净额）→账户硬约束→ExecutionPolicy 引用；
  `strategy_contributions` 保留毛/净两侧归因；幂等键=账户+artifact 版本+
  决策日+输入 as_of+政策版本，重试同结果。遗留：simulation_engine 消费
  净额计划的执行层接线为后续工作
- ✅ `implementation_tier`/`catalog_role` 机制与条件状态——已完成
  （2026-07-21）：`strategy_catalog.py` 机器可读注册表，§6.4 全部 26 个
  模板登记在册，conditional 项带 `blocked_reason` 可见，
  `list_recipe_catalog()`/`validate_recipe_catalog()` fail-closed
- ✅ 成员暂停自动预算重归一——已完成（2026-07-22）：决策日重估时活跃
  成员按政策方法重新归一，暂停成员预算转 cash_reserve，预算之和≤可投资
  资本，非决策日不临时重估

### 阶段 4/9 模拟与账本（补强）

- ❌ 拆并股/配股/换股/代码变更/ETF 折算/基金清盘公司行动类型
- ❌ 公告阶段信息事件（现公告行直接丢弃）
- ❌ 需持有人选择的复杂事件提醒+人工处理
- ❌ 订单 planned/open/cancelled 持久态；价格保护/执行窗口/策略贡献字段
- ❌ 外部现金流（出入金）与单位化 TWR/XIRR（§4.4、§8.3、§12.1）
- ❌ 现金批次/冻结桶/可取现金视图（现为单一 cash 浮点）
- ❌ 日终最终费用导入与差额确认；日终归因存档
- ❌ 离散化后约束重检（现金/单票/行业/TE/换手/容量）
- ❌ 手动/CSV 影子账户（manual_shadow）全链路（§8.6）
- ❌ 个人市场权限模型（buy_sell/sell_only/disabled/unknown + as_of/有效期
  + 只能收紧 + simulation_only 降级）（§8.7）
- ❌ ETF 子类型白名单门禁（现为注释预留，任何基金前缀均可模拟交易；
  方向保守但与合同不符）

### 阶段 6-8 模板与研究体系

- ❌ 6 种 research_mode 中 5 种（template_extension/component_discovery/
  model_challenger/portfolio_execution_challenger/new_strategy_proposal/
  falsification）；NewStrategyProposal 10 项内容与人工确认点
- ❌ ModelArtifact 与例行 refit 日历；回滚只切同 spec 合格历史制品
- ❌ `economic_hypothesis_group` 统一试验计数与资本上限
- ❌ 8 插槽形式化模板结构与空槽行为声明
- ❌ ExecutionPolicy 缺 3 个（wait_cancel_replace、multi_day_transition、
  participation_capped_slicing 完整版）；按证券/流动性路由
- 🟡 ResearchBrief 缺禁止项/失败条件/LLM 段；台账缺 LLM 提供商/模型/
  提示词版本记录；无读审计

### 阶段 9-10 推荐与运行日历

- ❌ BUY/SELL/EXIT/HOLD/NO_ACTION × READY/WAIT/PARTIAL/CANCELLED/EXPIRED/
  BLOCKED 两维模型；`projected_position`；keep/cancel/replace/new
- ❌ 两路径（继续旧目标/切换候选）换仓成本比较；成本感知 no-trade band
- ❌ 周报、月度决策日、盘前检查、盘中执行检查调度；对账通过才生成建议
  的顺序门（现 recommendation_refresh 与模拟对账独立）
- ❌ `safe_mode`（严重异常停新建议新订单但保留查看/对账/恢复）
- ❌ 风险状态三档 normal/caution/risk_off 与 `risk_scope` 标记
- ❌ 数据过期硬阻断新开仓/新建议的显式门（现为 degraded 标记，阻断链不完整）

### 运维与验收

- 🟡 备份恢复演练只有校验和，无订单/成交/NAV 对账（§11.4）
- ❌ 日线↔分钟聚合一致性检查（§3.5）
- 🟡 原始响应体默认不保留（`keep_raw=False`，偏离 §3.2 字面要求；
  登记为已知偏差或改默认）
- 🟡 披露日历对账只 warning 不阻断（已知，此前评估接受）
- ❌ 回测链 vs 模拟链黄金案例差分测试（见 2.1）

## 4. 无法静态判定（需运行时/凭证验证）

- Tushare 数据水位"等待"行为（现为 verify 报错而非延迟等待）
- 数据过期是否实际硬阻断新建议（推荐链运行时行为）
- Token/密钥不进入 RD-Agent 提示词与日志（需运行时扫描）
- RD-Agent Docker 内环境隔离实效（worker env 全量继承问题）
- 长期停牌持仓估值处理；页面展示层文案与指标呈现

## 5. 本期开发清单（按 goal 优先级）

1. **阶段 5**：AccountAllocationPolicy/AllocationArtifact 对象与冻结决策
   日语义；唯一 active 约束（DB 部分唯一索引）；`implementation_tier`/
   `catalog_role` 字段与条件状态机。
   - ✅ 已完成（2026-07-21）：迁移 `0039_allocation_policy_guards`（部分
     唯一索引 + `strategy_allocation_artifacts` 表 + `decision_frequency`/
     `recommendation_scope` 列）；`strategy_catalog.py` 全目录注册表；
     allocation/recommendation 激活路径 fail-closed；refresh 冻结决策日门。
     净额合并层与成员暂停预算重归一亦已完成（2026-07-22，见第 3 节阶段 5）。
2. **阶段 3**：OOS 密封（oos_vintage/sealed_candidate_set）；失败候选
   入账修正（worker.py:1708）；负对照/泄漏哨兵；逐项成本压力；HAC 零
   方差"未定义"；恢复期指标。
   - ✅ 部分完成（2026-07-21）：OOS vintage 密封与失败候选入账已完成
     （见 2.2/2.4）。
   - ✅ 追加完成（2026-07-22）：负对照/泄漏哨兵（placebo 对照 +
     标签平移哨兵，诊断标记不硬拒）、逐项成本压力（见 2.8）、HAC 零方差
     "未定义"、恢复期指标均已完成；latency/退市专项情景、消融、衰减曲线、
     Holm/PBO/区块 Bootstrap、现金收益与 XIRR 仍在清单。
3. **冲突修正（无需用户决策的）**：股息税保守负债（2.6）；披露/成本
   等登记类偏差入档。
4. **其余缺失项**：按清单顺序量力推进；本 goal 内未完成的项在文档中
   保留状态，最终报告如实说明。

## 6. 明确不做（本 goal）

- QMT 网关与配对策略的处置（第 1 节，待用户决策）
- web 前端（迁移在飞）；生产服务器操作；git push/merge
