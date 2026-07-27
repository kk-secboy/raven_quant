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
- 已决议（2026-07-22）：按选项 A 执行——QMT 保留为默认关闭的可选沙箱
  插件，设计稿 §1.3/§2.2/§8.6/§11.3/§13/§14 已修订纳入合同。

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
- 已决议（2026-07-22）：按选项 B 执行——配对策略降为 research-only。
  机制：`strategy_catalog.require_capital_eligible_strategy_type` 以目录
  `catalog_role` 为单一事实源；`strategy_store._approve_pair`、
  `allocation_store` 成员校验、`simulation_store.create`（pair 适配器）
  三处 fail-closed；promotion/recommendation 链因要求 approved/multifactor
  被传递覆盖；离线 `run_pair_backtest` 与脚本保留；迁移
  `0045_research_only_pair` 将存量 approved pair 版本降级 retired 并留
  审计事件。

## 2. 冲突（可直接修正，本期处理或登记）

### 2.1 ⛔ 唯一 ExecutionCore 不存在，回测/模拟两套成交语义（§2.3、§9.3、§12.3）

- 合同：单一轻量纯规则 ExecutionCore 统一成交判定，两条适配链黄金案例
  +差分测试。
- 现状：正式回测由 Qlib 原生 Exchange 撮合（`qlib_exchange.py:16-21`
  自述），模拟用 `simulation_engine.execute_simulation_day` 自研下一
  Bar 逻辑；仅共享成本模型、整手、公司行动原语。差分测试完全缺失（仅
  `test_document_governance.py:673` 校验设计稿文本含此要求）。
- 影响：「回测成绩=模拟成交」无任何测试保障，回测→模拟差异无法归因。
- 已修正（2026-07-26）：黄金案例差分套件上线
  （`tests/test_execution_core_differential.py`，16 案例；Qlib 链由
  `tests/execution_core_harness.py` 经 shim 加载 pinned qlib
  `backtest/exchange.py` 真实源码驱动，仅数据访问层替换）。收敛一处板级
  bug：模拟链非清仓卖出取整硬编码 100 股，科创板/北交所（步长 1）被错误
  压到主板整手——`simulation_engine.py` 改用
  `lot_rules[instrument].lot_increment`，回归测试
  `test_simulation_engine.py::test_non_liquidating_sell_uses_board_lot_increment`。
  差分确认完全一致面：正常下一 Bar 成交（数量/价格/费用/现金影响逐项
  一致）、停牌不成交、一字涨停买拒/跌停卖拒、参与率部分成交且余量当日
  过期、买入整手取整、成本六分项（佣金/最低佣金/印花税/过户费/滑点/
  冲击）逐组件一致、零股清仓卖出、T+1 解锁后卖出、现金充足买入。
- 已登记口径差异（差分套件以声明行为固定，编号 D1–D6 与测试一一对应）：
  - D1 涨跌停判定价位：Qlib 分钟链按 bar `vwap` 判定
    （`qlib_backtest.py` limit_threshold on `$vwap`），模拟链按 bar
    `close` 判定（`simulation_engine.py:_bar_rejection_reason`）。方向：
    close 触板而 vwap 未触板时 Qlib 成交、模拟拒单（模拟更保守）。
  - D2 日线正式链按日 `$open` 判定涨跌停与成交价，模拟链按分钟 bar；
    开盘一字板时 Qlib 日线拒单而模拟分钟链成交（结构性：两条链的数据
    频率不同，无法低成本统一）。
  - D3 T+1 执法层不同：Qlib Exchange 层为 T+0（T+1 由策略层
    `qlib_policy_strategy.apply_t1_target_floor` 在生成目标时拦截），
    模拟链在引擎层锁当日买入批次（`_sellable_quantity`）。订单层输入
    相同时结果一致；绕过策略层直接下单时 Qlib 链会成交当日买入——登记
    为分层差异而非行为错误。
  - D4 买入现金约束模型：Qlib 适配层现金检查用扁平保守费率
    （`qlib_exchange.py:34-38` 已自述）且 qlib 取整含 +0.1 epsilon，
    边界现金下可成交（如现金 1004 买 100 股 @10）；模拟链用精确共享
    成本模型迭代缩减，同例拒单（模拟更保守，方向有利于可信回测）。
  - D5 多 bar 窗口聚合语义：Qlib 对订单窗口用 `all` 判定停牌/涨跌停
    （窗口内全部 bar 受限才拒单），模拟链逐 slice bar 判定；模拟侧无
    多 bar 窗口概念，属结构性不可比，测试中固定 Qlib 侧行为防漂移。
  - D6 非清仓卖出取整：Qlib 按 1 股取整（零股减持全成交），模拟链按
    板块步长保守下取整（主板 150→100）；修正板级步长后该差异仅存在于
    主板/创业板/基金非清仓减持场景，方向为模拟少卖（更保守）。

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
- ✅ 已修正（2026-07-26）：迁移 `0043_promotion_stages`
  （`strategy_versions.promotion_stage` + `strategy_forward_gates` 预注册
  门槛表 + `strategy_promotion_stages` 阶段表）；`StrategyStore.approve`
  硬门通过自动置 `paper` 并打开隔离阶段（审批 artifact 带 `datasets.json`
  时同事务链自动绑定独立 paper 模拟账户，缺失则 `awaiting_simulation`
  不阻断审批）；`PromotionStore` 提供门槛注册（仅 paper 前）、证据评估
  （NAV 跨度/成功批次/完整往返/数据完整率/对账率/成本偏差六子项，
  不足 fail-closed 标 insufficient_evidence）与四人眼人工晋升；来源合同
  漂移按 §9.5 冻结旧阶段只读、新阶段证据从零不拼接；standalone 推荐组合
  创建拒绝 paper 阶段版本（NULL 阶段为迁移前旧行/夹具的祖父语义）。

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
- 🟡 现金收益参数（cash_yield）仍缺；XIRR 已完成（2026-07-27：模拟账户
  `performance_summary` 资金加权补充指标，退化输出状态而非伪精确）；
  恢复期指标已完成（2026-07-22：`qlib_backtest.py`
  max_drawdown_recovery_days / max_drawdown_recovery_status，
  recovered/ongoing/no_drawdown 三态；2026-07-27 模拟链
  `unitized_performance.py` 基于单位化曲线的恢复期）

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

- ✅ 拆并股/配股/换股/代码变更/ETF 折算/基金清盘公司行动类型
  - 已完成（2026-07-27）：`corporate_actions.py` 新增 `CorporateEvent`
    信封（唯一事件键 + effective_date + 载荷哈希）；拆股/并股/ETF 份额
    折算经 `apply_share_split` 在经济生效日按比例调整数量与单位成本
    （总成本不变、不产生现金、不动 NAV 口径，防虚假亏损；并股不足 1 股
    的尾仓不臆造份额，critical 留痕人工处理）；检测走 adj_factor/
    fund_adj 跳变（`detect_split_events`，dividend 行已解释的跳变剔除）
    或显式录入；代码变更经 `apply_code_change` 迁移持仓身份（数量/批次/
    成本原样、应收跟随、无经济损益，双活持仓 fail-closed），namechange
    行驱动（`normalize_namechange_rows`，仅名称变更时为信息事件）；
    迁移 `0050_corporate_event_types` 新增 `simulation_corporate_events`
    append-only 台账（portfolio_id+event_key 唯一，只应用一次）；
    引擎 `corporate_events`/`applied_event_keys` 入口，迟到事件只标记
    复核不回写；store 证据哈希绑定 `corporate_events_sha256`。
  - 有据跳过：配股（无配股发行表）、换股（无要约结构化工单）、基金清盘
    （fund_basic 仅摘牌日期、无清算回收金额，设计 §5.6 禁止统一假设现金
    结算）→ `UNSUPPORTED_EVENT_TYPES` fail-closed：事件流 critical 记
    `unsupported_event_type:<type>` 原因并入台账幂等留痕，永不自动入账；
    清盘公告经 anns_d 关键词（`detect_liquidation_events`）接入该路径
- ✅ 公告阶段信息事件（原公告行直接丢弃）
  - 已完成（2026-07-27）：`normalize_announcement_rows` 把 dividend 预案行
    （无 ex_date）转为 plan 阶段 informational 事件（不改现金/持仓/NAV），
    实施公告行携带 `linked_ex_event_key`/`related_event_key_prefix` 与
    后续除权事件唯一键关联；事件键 `corporate_action:announcement:
    {instrument}:{ann_date}` 幂等，重放不重复入账
- ✅ 需持有人选择的复杂事件提醒+人工处理
  - 已完成（2026-07-27）：通用 `choice_required` 事件类型——anns_d 标题
    关键词（配股/换股/要约/现金选择权/吸收合并，
    `detect_choice_required_events`）或显式录入识别；产生 warning 提醒
    事件并在持仓上置 `choice_pending` 待人工处理标记（store 从台账重推导）；
    处置前卖出不硬阻断但产生 `corporate_action_choice_pending_sale` 可见
    提示；系统永不代客提交选择。遗留：人工处置回写（解除标记）与
    alert_store 外发接线待后续队列项
- ✅ 订单 planned/open/cancelled 持久态；价格保护/执行窗口/策略贡献字段
  （2026-07-23 已完成：迁移 `0046_simulation_order_states` 给
  `simulation_orders` 增加 portfolio_id/limit_price/not_before/not_after/
  target_version/account_netting_plan_id/strategy_contributions_json/
  plan_op/cancel_reason/updated_at 与状态检查约束；`simulation_order_state.py`
  纯状态机消费 keep/cancel/replace/new；`SimulationStore.
  create_order_plan_batch` 单事务提交计划（取消只释放未用余量一次、事件入
  流、幂等键防重），`process_batch` 计划模式执行工作单（跨日窗口留存
  open、过期 fail-closed、风险门阻断新增买单），引擎支持价格保护与执行
  窗口；净额计划绑定校验账户归属并落 strategy_contributions）
- ✅ 外部现金流（出入金）与单位化 TWR/XIRR（§4.4、§8.3、§12.1）
  - 已完成（2026-07-27）：迁移 `0049_external_cash_flows` 新增
    `simulation_external_flows`（flow_key 幂等、open/close 时点、事件流
    落账、已结算交易日 fail-closed）与 `simulation_nav`/`simulation_portfolios`
    单位化列；`unitized_performance.py` 实现 F_t_open/F_t_close TWR 链
    （r_t=(V_t−F_close)/(V_{t−1}+F_open)−1，分母非正/断链标记
    undefined/broken_chain 且不跳过续乘）、单位化回撤与恢复期（recovered/
    ongoing）、XIRR（网格夹逼+二分，undefined_single_sign/no_root/
    multiple_roots 状态）；引擎 open 流盘前可投资、close 流盘后入 NAV
    不可交易，守恒校验含现金流；人民币 NAV 口径并存不替换；
    `SimulationStore.record_external_flow`/`performance_summary` 对外输出
- ❌ 现金批次/冻结桶/可取现金视图（现为单一 cash 浮点）
- ❌ 日终最终费用导入与差额确认；日终归因存档
- ❌ 离散化后约束重检（现金/单票/行业/TE/换手/容量）
- ✅ 手动/CSV 影子账户（manual_shadow）全链路（§8.6）
  - 已完成（2026-07-27）：迁移 `0048_permission_shadow_account` 新增
    `shadow_account_snapshots`（持仓/现金/可卖/未完成订单快照，含
    import_source/imported_by/imported_at/content_sha256）；
    `shadow_account.py` 提供手工/CSV 导入（schema fail-closed）、
    自然日新鲜度判定与 `account_state_for_actions`；推荐链接入
    `attach_account_actions(account_context=...)`，影子/模型/模拟账户
    以 `account_context.account_type` 分离展示，陈旧/缺失状态降级为
    simulation_only（WAIT + 原因），不静默回退模拟账本。
- ✅ 个人市场权限模型（buy_sell/sell_only/disabled/unknown + as_of/有效期
  + 只能收紧 + simulation_only 降级）（§8.7）
  - 已完成（2026-07-27）：同迁移新增 `market_permission_versions`
    （scope=exchange/board/risk_warning/etf_subtype + 权限 +
    confirmation_source + as_of/valid_until + relaxation_confirmed）；
    `market_permission.py` 收紧纪律（buy_sell→sell_only→disabled→unknown
    方向放行，放宽需显式确认）、过期/未确认判 unknown；动作层接入
    `attach_account_actions(permission_store=...)`：disabled/sell_only
    阻断买入（BLOCKED 记原因，SELL/EXIT 放行），unknown/过期标
    simulation_only（WAIT，建议保留）。
- ✅ ETF 子类型白名单门禁（2026-07-27 已完成：
  `etf_subtypes.py` 版本化注册表（子类型+验收状态+验收日期/依据，参考
  strategy_catalog 模式；仅股票型 equity 标注已验收，跨境/债券/黄金/
  商品/货币登记待验收项，未验收默认拒绝；代码段启发式+显式代码覆盖
  分类，LOF/封基等无法归类者 unclassified 一并拒绝）；与
  market_permission 刻意分建——后者是个人授权 unknown→simulation_only
  降级语义，不满足平台验收硬拒单语义；门禁挂在 `simulation_engine.
  execute_simulation_day` 成交判定处，未验收子类型订单 fail-closed 拒单
  记原因（etf_subtype_not_accepted:<subtype>），买卖双向、权重推导与
  持久订单两路径同门；研究/回测（含配对台账）不受限）

### 阶段 6-8 模板与研究体系

- ❌ 6 种 research_mode 中 5 种（template_extension/component_discovery/
  model_challenger/portfolio_execution_challenger/new_strategy_proposal/
  falsification）；NewStrategyProposal 10 项内容与人工确认点
- ❌ ModelArtifact 与例行 refit 日历；回滚只切同 spec 合格历史制品
- ❌ `economic_hypothesis_group` 统一试验计数与资本上限
- ❌ 8 插槽形式化模板结构与空槽行为声明
- ✅ ExecutionPolicy 缺 3 个（wait_cancel_replace、multi_day_transition、
  participation_capped_slicing 完整版）——已完成（2026-07-27）：
  `execution_algorithms.py` 补齐 6 政策目录并输出版本化
  `execution_policy_id`（next_bar_baseline/twap_execution/vwap_execution 为
  既有三者的规范名映射）；`plan_participation_capped_slices` 显式政策形态
  （slot_volumes 量能证据 × max_participation 整手容量，切片参与率恒
  ≤上限，容量不足的余量显式报 `unallocated_quantity` 过期/重报，绝不当
  已成交）；`plan_wait_cancel_replace` 限价等待→按 wait_checks（1min/
  5min 检查节奏）未成交取消→按 replace_step_bps 逐档激进改价重报
  （买上取/卖下取 0.01 tick），max_replaces 后余量过期，语义与
  simulation_order_state 对齐（取消只释放未成交余量一次、replace=取消+
  新价新单且永不放大订单）；`plan_multi_day_transition` 跨日切片（每日
  基数=剩余量/剩余天数向上整手，daily_volumes 证据下按参与率封顶，每日
  not_before 10:00/not_after 15:00 与订单窗口对齐，Σ切片+未分配=总量
  守恒，可选 intraday_algorithm 日内复切）。全部纯函数确定性。
  遗留：按证券/流动性路由（多政策组合路由）仍未实现
- 🟡 ResearchBrief 缺禁止项/失败条件/LLM 段；台账缺 LLM 提供商/模型/
  提示词版本记录；无读审计

### 阶段 9-10 推荐与运行日历

- ✅ BUY/SELL/EXIT/HOLD/NO_ACTION × READY/WAIT/PARTIAL/CANCELLED/EXPIRED/
  BLOCKED 两维模型；`projected_position`；keep/cancel/replace/new
  （2026-07-26 已完成：`recommendation_actions.py` 纯规则计划层 +
  迁移 `0044_recommendation_actions` 的 `account_actions_json` 快照集成；
  模拟订单当日 15:00 过期的未成交余量按 EXPIRED 重报而非当作仍有效）
- ✅ 两路径（继续旧目标/切换候选）换仓成本比较；成本感知 no-trade band
  ——已完成（2026-07-27）：新模块 `transition_decision.py`
  （`transition-decision-v1`）。`estimate_transition_cost` 用共享
  CostModelConfig 对每条权重变动腿按买/卖分别计佣金+印花税（股票卖出）
  +过户费+滑点+冲击（ADV 证据参与率，缺省保守 max_volume_participation）
  得完整人民币换仓成本；`compare_transition_paths` 输出 hold_path（成本 0
  保旧目标）与 switch_path 全量对照、增量收益、模式与决策原因入证据。
  增量预期收益只走显式冻结映射 `expected_returns_from_scores`（截面
  z-score × 冻结 slope × horizon，硬封顶），再乘 conservative haircut 与
  完整成本比较（≤成本→HOLD）；直接把 rank score 当人民币传入触发
  fail-closed 校验。无校准收益时退化为预冻结漂移带（min_weight_change/
  min_turnover/min_order_value，不重复计成本）。现金缺口/失效/权限收紧/
  风险退出等 hard_constraints 绕过 band（决策 forced 并记原因）。
  遗留：与 PortfolioPolicy.decide/推荐链的接线（消费决策入快照）为后续
  队列项
- 🟡 周报、月度决策日、盘前检查、盘中执行检查调度；对账通过才生成建议
  的顺序门（2026-07-27 部分完成：新调度 kind `weekly_report`（周六本地
  时间，汇总模拟 NAV/批次/成交费用/风险事件/数据健康，落
  `artifacts/ops-reports/` + 去重告警）、`monthly_decision_day`（每月首个
  交易日，到期 allocation artifact 走 0039 decision_frequency 语义触发
  `AllocationStore.refresh` 重估 + 月度健康汇总）、`preopen_check`（交易日
  盘前：日历确认/数据就绪/账户状态/未完成订单/停牌与风险摘要）；
  顺序门 `ops_calendar.evaluate_recommendation_gate`：`recommendation_refresh`
  入队前校验关联模拟账户最近批次对账通过 + NAV healthy/certified/无
  stale/不滞后，阻断则 fail-closed 跳过并产 critical 告警，旧快照保留并
  标注 `retained_stale`；均遵守 trading_days_only 日历门、misfire
  fail-closed、幂等与 worker 分发纪律。遗留：盘中执行检查调度）
- ✅ `safe_mode`（严重异常停新建议新订单但保留查看/对账/恢复；
  2026-07-27 完成：迁移 `0047_safe_mode` 单例状态行 + `safe_mode.py`
  SafeModeStore（自动/人工开启、人工 actor+reason 恢复、可选健康门、幂等
  告警）；自动触发=模拟批次账本守恒失败（mark_batch_failed 识别
  ledger-integrity 错误）/数据质量门 verify 失败（cli verify+snapshot）/
  任一活跃模拟账户连续 3 期 NAV degraded 或未认证（scheduler
  project_alerts）；阻断面=顺序门首检 + create_snapshot +
  模拟账户/批次/订单计划/配对批次创建 fail-closed；放行=读路径、
  process_batch 既有批次对账、mark_batch_failed 恢复登记；健康组件
  safe_mode 入 OperationalHealthStore，API GET/engage/release）
- ❌ 风险状态三档 normal/caution/risk_off 与 `risk_scope` 标记
- ❌ 数据过期硬阻断新开仓/新建议的显式门（现为 degraded 标记，阻断链不完整）

### 运维与验收

- 🟡 备份恢复演练只有校验和，无订单/成交/NAV 对账（§11.4）
- ✅ 日线↔分钟聚合一致性检查（§3.5）（2026-07-27 已完成：
  `verify._verify_minute_daily_consistency` 对同一 (ts_code, trade_date)
  将分钟 bar 聚合（开=首 bar 开、高/低=max/min、收=末 bar 收、量额=Σ）
  与日线对比；价格相对容差 1e-4，量额按换算契约折算（分钟股/CNY →
  日线手/千元）后同容差；仅覆盖契约明确的股票/ETF 分钟数据集
  （SIMULATION_MINUTE_SOURCE_DATASETS，指数/期货/期权不在换算契约内）；
  公共键取值矛盾记 error（与 OHLC 矛盾同级），分钟覆盖缺失与反向键只
  记 minute_daily_checks coverage 计数不硬失败）
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
     Holm/PBO/区块 Bootstrap、现金收益仍在清单。
   - ✅ 追加完成（2026-07-27）：外部现金流（出入金）与模拟链单位化
     TWR/XIRR（见第 3 节阶段 4/9 对应行）。
3. **冲突修正（无需用户决策的）**：股息税保守负债（2.6）；披露/成本
   等登记类偏差入档。
4. **其余缺失项**：按清单顺序量力推进；本 goal 内未完成的项在文档中
   保留状态，最终报告如实说明。

## 6. 明确不做（本 goal）

- QMT 网关与配对策略的处置（第 1 节，待用户决策）
- web 前端（迁移在飞）；生产服务器操作；git push/merge
