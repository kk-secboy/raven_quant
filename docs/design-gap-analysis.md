# 冻结设计稿实施状态（2026-07-29）

基准：`个人量化投资与模拟盘系统设计稿.md` v1.1。本文只描述当前代码
状态，不代表任何策略具有投资价值，也不把宿主机测试冒充生产部署。

## 结论

冻结设计稿要求的大部分代码骨架和安全合同已经存在，但产品主线**尚未形成
可实际使用的生产闭环**：

`Tushare/PIT 数据 → RD-Agent 研究 → Qlib 探索 → 候选冻结 → 独立复算与
正式验证 → 隔离 paper → 人工批准 → 唯一账户推荐 → 主模拟账本 →
月/周/日/分时运行与 Web 观察`

当前代码已经覆盖现金批次、订单冻结、费用差额、离散后硬约束、统计工具、
模型制品、研究模板、推荐调度、Web 前端和日终归因；这证明能力入口存在，
不证明已有 Alpha、可投资策略或可运行推荐。2026-07-29 审计确认：

- 线上既有日线 Qlib 制品主要覆盖 2024—2026；2018—2026 原始全量回补仍在
  处理 `share_float` 分页恢复，2008—2017 尚未开始，因而没有 2008 至今的
  冻结日线研究数据集。
- 研究入口默认窗口已改为训练 2008—2017、验证 2018—2020、最终锁箱
  2021 至最近完整交易日；新闻类接口按真实历史起点裁剪，分钟历史不冒充
  2008 至今覆盖。但这些是计划和门槛，尚不是已完成的数据或回测成绩。
- `run_multifactor_backtest.py` 的外层 walk-forward 当前使用正式 backtest
  payload 的 `periods`，而自动研究链把该 payload 固定为最终测试窗口。
  因此“外层 walk-forward 覆盖 2008 至今多轮市场状态”尚未被代码与制品
  共同证明，必须拆分完整历史选择证据和一次性最终锁箱证据后重新验收。
- 线上尚无通过新门槛的正式策略、10 万元隔离 paper、真实前向批次或唯一
  账户推荐；分时提醒虽有执行检查代码，也还没有可消费的批准订单计划。

## 实施对表

| 设计域 | 当前状态 | 主要实现证据 |
| --- | --- | --- |
| 系统边界 | 已实现 | 主线不接券商写接口；QMT 仅为默认关闭、沙箱限定的可选插件；pair 策略只能离线研究 |
| 数据与 PIT | 代码已实现、线上数据未完成 | availability/recoverability 合同、行级 `ingested_at`、数据身份与 lineage、原始响应默认留存；2008—2017 回补和全历史质量门待完成 |
| 正式验证 | 部分实现 | Holm、PBO、区块 bootstrap、配对比较、消融、信号衰减和一次性最终 OOS 已有代码；完整历史外层 walk-forward 与最终锁箱的区间分离仍需修复和制品验收 |
| 研究治理 | 已实现 | `ResearchBrief`、六种研究模式、八个组件槽、假设组上限、失败试验入账、预注册准入与联合候选密封 |
| 策略与模型 | 已实现 | 设计稿策略目录、标准/conditional 能力分级、不可变 `StrategySpec`、独立 `ModelArtifact` refit 生命周期 |
| 账户分配 | 代码已实现、无线上批准制品 | 唯一启用政策、冻结决策日、`AllocationArtifact`、预算只应用一次、账户级净额与同向比例归因 |
| 推荐 | 代码已实现、无线上推荐 | 唯一推荐账户、分时执行检查、账户三级风险态、硬门、有效期和显式阻断原因 |
| 执行语义 | 已实现 | 受管 next-bar/TWAP/VWAP、整手/最低委托/参与率/涨跌停/T+1、历史与模拟差分测试 |
| 模拟账本 | 已实现 | 现金批次、可交易/可取时间、买单现金冻结、卖单证券冻结、部分成交消费、撤单/替换幂等释放 |
| 费用与交收 | 已实现 | 成交时确认费用、最终费用只计差额、卖出回款批次、公司行动应收与税负债、交收只重分类 |
| 风险与对账 | 已实现 | 现金/持仓/NAV 守恒、重复事件键、负值检查、陈旧数据降级、严重账本异常触发 safe mode |
| 日终归因 | 已实现 | 每批保存策略、行业、资产、成本与执行偏差归因；缺少绑定行业快照时明确标为 partial |
| 运行日历 | 已实现 | 日/周/月/分时任务、交易日和补跑规则、冻结推荐与研究调度解耦 |
| Web 产品 | 已实现 | 首页、研究、组合、策略详情、推荐、运行、登录；桌面/移动布局、加载/空/错状态、API 代理 |
| 部署制品 | 代码已实现、生产版本待更新 | API/worker/Web Docker 构建定义、Web 健康检查、Node 版本约束和本地构建路径；当前线上版本落后于本审计提交 |

## 关键闭环

### 1. 正式验证不是探索分数

- `formal_validation.py` 提供 Holm 多重检验、PBO、moving-block bootstrap、
  同区间配对比较、外层 walk-forward 聚合、消融和延迟衰减门。
- `run_multifactor_backtest.py` 把正式验证写入冻结 artifact；正式批准读取
  这些结果，缺字段或失败状态均不能晋升。
- 当前自动链仍把外层 walk-forward 限制在最终 backtest 区间，不能据此声称
  已覆盖 2008 至今。修复必须让嵌套选择只消费训练/验证历史，让最终锁箱保持
  一次性、物理隔离，并在批准门校验两段不可变证据的日期、数据身份和哈希。
- `StrategyConfigRequest` 冻结现金收益口径。没有治理过的可投资现金工具时
  强制使用零收益，不能把研究无风险利率冒充可获得收益。
- 最终 OOS 由不可换皮的 vintage 和密封候选集合约束；看过结果后不能补入
  候选或重新领取同一段最终样本。

### 2. 研究框架不是一个示例策略

- `research_contracts.py` 固化六种研究模式与八个组件槽，分别约束
  baseline、标准 alpha、conditional alpha、模型挑战者、账户政策挑战者和
  执行政策挑战者。
- 四类依赖专用 PIT 数据或分钟权限的 conditional 模板会展示缺失条件，
  不会进入 capital/recommendation 生命周期。
- `model_artifact_store.py` 将例行 refit 与 `StrategySpec` 变更分开；输入、
  配方、预测、有效期和安全门都有不可变哈希。
- `economic_hypothesis_group` 与组上限进入策略和分配成员，避免换名字绕过
  同一经济假设的集中度限制。

### 3. 推荐只消费冻结制品

- 推荐链只读取已批准版本、当前有效 `ModelArtifact` 和
  `AllocationArtifact`；普通推荐不会临时运行 RD-Agent 或重算账户预算。
- 连续目标先经过申报单位离散化，再重新检查现金、单票、行业、资产类别、
  tracking error、换手、容量和风险硬门。
- 账户风险态为 `normal / restricted / blocked`；风险证据陈旧或缺失时
  fail-closed，不会沿用看似正常的旧状态。
- 分时任务类型独立于日度推荐，负责检查已批准订单计划的执行窗口，而不是
  盘中扫描未批准策略。

### 4. 模拟账本不会重复计算资产

- 迁移 `0054_simulation_cash_lots` 建立互斥现金批次、追加式现金事件和
  订单预留。买单创建只把自由现金重分类为冻结，NAV 不变；成交按实际金额
  与已确认费用消费，终态只释放剩余头寸一次。
- 迁移 `0055_security_reservations` 建立卖单证券预留和追加式证券事件。
  同一份可卖数量不能被两张订单重复冻结；成交、撤单、替换和过期都按唯一
  事件键消费或释放。
- 卖出回款是单一现金资产，携带 `tradable_at` 和 `withdrawable_at`；
  系统使用绑定 Qlib 日历给出的下一交易日，不再用工作日近似。
- 迁移 `0051_simulation_fee_adjustments` 只保存最终费用与已确认费用的
  差额；交收事件除新增确定差额外不改变 NAV。
- `simulation_engine.py` 仍是纯计算核心；`simulation_store.py` 在一个
  PostgreSQL 事务内只应用一次结果并做现金、证券与 NAV 对账。

### 5. 日终归因不制造数据

- 迁移 `0056_day_attributions` 为每个模拟批次保存不可变日终归因。
- 策略执行归因优先按账户净额计划中冻结的同向净需求比例分配；没有成员级
  冻结贡献时只给出来源级汇总，并明确标注 fallback。
- 资产、成本和执行偏差从实际持仓、成交与费用明细计算。
- 模拟回放支持带 SHA-256 证据的 PIT 行业快照；快照完整时保存行业暴露与
  成交活动归因。未提供或覆盖不全时分别标为
  `blocked_missing_bound_industry_snapshot` /
  `blocked_incomplete_bound_industry_snapshot`，整份归因为 `partial`；
  不能把证券代码或当前行业标签猜成历史行业。

## 外部条件与阻断状态

以下不是继续隐藏的代码待办，而是运行时条件：

1. **真实数据与正式运行结果**：需要可用的 Tushare 权限、冻结 Qlib
   数据集和足够历史区间。没有这些条件时，系统能验证合同和算法，但不能
   声称已有可投资正式成绩。
2. **PIT 行业日终归因**：需要在模拟回放证据中绑定不可变行业快照。缺失时
   归因保留 partial 状态，不阻塞账本入账，也不输出伪行业贡献。
3. **conditional 策略**：事件驱动、分钟 ETF、期货 CTA、ML ensemble
   等模板必须取得各自的数据、权限、成本和前向证据后才能晋升；默认停在
   research。
4. **生产部署**：本地代码和镜像定义可交付，但未获授权时不推送、不重启
   服务器，也不把本地 build 当成已部署。
5. **投资结论**：代码完成只证明系统合同闭合；任何策略仍需真实冻结数据、
   正式样本外结果、隔离前向证据和人工批准。

## 最终验收命令

```powershell
$env:PYTHONPATH='src'
$env:TEST_DATABASE_URL='postgresql+psycopg://quantlab:quantlab@127.0.0.1:55433/quantlab_test'
.\.venv\Scripts\python.exe -m ruff check . --exclude .worktrees,quant,web,node_modules
.\.venv\Scripts\python.exe -m pytest --basetemp=E:\pytest-codex\goal-full-final -q

$env:Path='C:\Users\joejoe\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin;' + $env:Path
pnpm --dir web run lint
pnpm --dir web run test
```

验收还应执行 `git diff --check`，并确认 `.worktrees/`、`quant/`、
`web/node_modules/` 与 `web/.wrangler/` 等本地/生成目录不属于交付源文件。

## 2026-07-28 最终验收结果

- Python 全量 `pytest -q`：100% 通过，退出码 0。
- Ruff 全库检查：通过；`git diff --check` 无空白错误。
- Web：TypeScript `--noEmit`、ESLint、Vinext 生产构建与 2 个 SSR 契约
  测试全部通过；生产构建本地启动后 `/` 返回 HTTP 200。
- Alembic：单一 head 为 `0056_day_attributions`。
- Docker Compose：在提供必需部署变量后 `config --quiet` 通过，Web 与
  gateway 使用健康依赖。
- 本地 Web 镜像构建未宣称成功：Docker Hub 网络无法取得
  `node:22-alpine` 元数据；这是外部镜像源阻断，未重启或推送任何服务。
