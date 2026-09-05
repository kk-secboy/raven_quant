# QuantLab

QuantLab 是面向 A 股中低频量化研究与模拟交易的本地优先平台。主数据源是
Tushare；2008–2015 年仅允许使用通过重叠校验的 BaoStock 行情补档。Qlib 和
RD-Agent 组成唯一研究技术主线。

> **权威关系：**[根目录 Markdown](%E4%B8%AA%E4%BA%BA%E9%87%8F%E5%8C%96%E6%8A%95%E8%B5%84%E4%B8%8E%E6%A8%A1%E6%8B%9F%E7%9B%98%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1%E7%A8%BF.md)
> 是产品、策略和风险基准，Qlib/RD-Agent 是技术基准。本 README 只提供项目入口和
> 使用方法，不定义另一套产品方案。

根目录同名 DOCX 仅保留为 3.0 定稿发布快照，不再作为日常修改源。
根目录 `如何搭建自己的量化交易系统-GitHub原版副本.docx` 仅保留为冻结的历史原稿，
不属于现行产品基准，也不得作为第二套方案继续修订。

唯一生产主线是：

`受治理数据 → 因子/模型/交易规则研究 → 独立复算与样本外验证 → 隔离模拟盘 → 严格前向门 → 自动晋级活动策略 → 每日长中短选股 → 三周期账户净额 → 用户建议与统一模拟账本`

项目不做实盘、Tick、Level-2、逐笔或毫秒高频。历史 QMT 沙箱源码不进入生产镜像，
也没有启动脚本、配置入口或 Web 开关；页面、调度和模拟任务不得向任何券商网关发单。

## 代码入口

- `src/quant_data`：Tushare 下载、不可变快照、血缘和 Qlib 数据转换。
- `src/quant_platform`：研究编排、准入、Qlib 回测、自动晋级、分配、模拟和运维 API。
- 生产发布只到不可变模拟盘账本；真实券商网关不打包、不配置、也不提供开启开关。
- `web`：数据、RD-Agent、因子准入、Qlib 回测、晋级、分配和模拟控制台。
- `migrations`：PostgreSQL/Alembic 版本化迁移。
- `deploy`：Docker Compose、镜像和系统服务模板。

Python 包由 `pyproject.toml` 管理并通过 Hatchling 构建；可执行入口包括
`quant-data`、`quant-db`、`quant-worker`、`quant-scheduler` 和 `quant-web`。
Docker 镜像分别位于 `deploy/Dockerfile.api`、`deploy/Dockerfile.worker`、
`deploy/Dockerfile.rdagent` 和 `deploy/Dockerfile.web`。

## 本地安装

需要 Python 3.11+、PostgreSQL、Node.js 22.15+，以及运行 RD-Agent 沙箱时
所需的 Docker。

```powershell
cd E:\projects\rdagent-python
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
.\.venv\Scripts\python.exe scripts\configure_tushare.py --env-file .env
```

在 `.env` 中配置 PostgreSQL 连接；Tushare Token 应通过上面的隐藏输入脚本写入，
不要放入命令行、提交记录或聊天内容。

初始化数据库并启动本地 API：

```powershell
.\.venv\Scripts\quant-db.exe upgrade
.\.venv\Scripts\quant-web.exe --reload
```

另一个终端启动前端：

```powershell
cd E:\projects\rdagent-python\web
corepack enable
pnpm install --frozen-lockfile
pnpm run dev
```

## 数据与研究起点

```powershell
.\.venv\Scripts\quant-data.exe probe
.\.venv\Scripts\quant-data.exe bootstrap --profile full --start 2016-01-01 --snapshot-start 2008-01-01 --end latest
.\.venv\Scripts\quant-data.exe status
.\.venv\Scripts\quant-data.exe verify
```

主数据网关只从 2016 年开始；2008–2015 年行情必须先通过 BaoStock 2016
重叠校验，再运行独立的历史回补。只有两段真实数据合并且达到研究窗口要求后，
系统才允许生成可用于 RD-Agent/Qlib 正式研究的快照。正式研究收口必须使用
`full` profile；`core` 和 `research` 只用于分阶段下载与审计。

Bootstrap、Qlib 转换、RD-Agent 研究和回测也可以从 Web 控制台创建为持久任务。
任务关闭浏览器后不会丢失。日线与分钟数据使用隔离的数据集和执行契约；15/30/60
分钟研究由 Qlib 从 1 分钟或 5 分钟数据重采样，不增加下载线路。
日频接口升级显式字段合同时，原始旧单元继续不可变保留。若增量回看产生同一
基金/交易日的旧、新合同记录，质量门会逐业务字段复核：完全相同才由快照做语义
去重，价格或复权因子存在任何冲突都会阻断发布，未完成的新单元也不得被旧成功
记录掩盖。
旧财务单元缺少行级 `ingested_at` 时，不允许直接修改原快照。运维人员只能在旧快照
manifest、成功 work-unit ledger、原始单元 SHA/行数与封印投影全部吻合后运行
`quant-data snapshot-ingested-at-successor --source <旧快照> --name <新快照>`；命令仅修复
已审计的 `fina_indicator`，把 ledger `updated_at` 作为保守 acquisition 上界，并验证修复
前后除 `ingested_at` 外的 provider 行集完全一致。Daily Qlib v8 才具有现行正式准入权限，
旧 v7 及更早制品只保留审计用途。
全 A 股 5 分钟增量任务按成功 checkpoint 的实际交易日覆盖复用旧单元；每个尚未覆盖的
连续区间按自然季度合并且单次不超过 150 个交易日。这样补齐一个 14 交易日缺口时，
每只股票只请求一个区间，下一交易日仍只新增后缀，既不重下历史也不改变旧成功单元键。

## RD-Agent 研究中心

平台在同一套不可变数据、作业、制品和审计框架中登记八个场景；其中三个保留场景照常
自动调度、创建和执行，五个冻结场景不再调度、不能通过 API 创建、worker 拒绝执行，
历史运行与制品全部保留只读。

保留场景：

- `fin_quant`：联合研究完整的因子集与模型 bundle；每次运行先保证因子、模型两臂各有一次真实尝试，再交回官方 bandit。只有两臂都被接受并通过因子单独、模型单独、联合及联合相对 incumbent 四项独立消融，才形成可准入的联合候选；否则记录为无资本权限的研究负结果。
- `fin_strategy`：研究股票资格、市场状态、排名、入场、退出、持有期、组合和风险规则；输出 `StrategyProposal` 与确定性编译的规则 IR，仅有研究权限。
- `fin_factor_report`：只读取已验签且满足可用时点的研报 PDF，提取因子后复用因子门禁。

冻结场景（不再产生新运行，历史只读）：

- `fin_factor`：因子提出、实现和迭代；候选仍须独立复算、PIT 检查和三窗口准入。
- `fin_model`：在固定特征集上研究预测模型；使用固定随机种子重新训练并独立验证。
- `general_model`：从论文实现模型，初始状态仅为 `implementation_ready`；兼容实现可人工送入模型门禁。
- `data_science`：隔离运行通用数据科学任务，制品只进入实验室。
- `llm_finetune`：只在密封模型/数据、固定镜像和合格 GPU 能力同时满足时运行，制品不属于交易模型。

官方 RD-Agent 运行成绩和 Trace SOTA 只作为研究反馈，不能直接晋级。因子、模型和
策略候选必须通过 QuantLab 的独立 Qlib 评价、密封最终 OOS 和正式回测后，才会进入
隔离模拟盘；满足各周期真实前向门后由系统原子自动晋级，人工只能紧急暂停或回滚。研究与回测阶段的统计指标（bootstrap、PBO、DSR、显著性校正）照常计算并密封入档但不参与否决（宽进严出）；模型锦标赛、外部 NLP 因子评估和因子 SOTA 增量适用同一口径，统计显著性一律只入档；整条链唯一的生死门是模拟盘前向表现。
Alpha20/158/360 只是基础候选特征；每周期只冻结一个由基础特征与已证明增量价值的
RD-Agent 因子组成的冠军因子包，并限制为一个活动策略和最多两个隔离影子挑战者。
首次冷启动或没有合格旧策略时，`fin_strategy` 仍按既定周期用透明公开配方作为
研究对照组，但父策略保持为空；这不会给候选模拟、推荐或生产权限。
若同一数据身份和周期已经存在独立准入的模型、集成或联合冠军，`fin_strategy` 会冻结
它的完整信号身份：`policy_only` 只比较买卖和持仓规则，保持同一份分数；通过后才在
`full_stack` 将完整冠军信号加新规则与透明公开基线比较。冠军不存在时才回退透明基线，
任何身份不一致都失败关闭。历史训练上下文可以早于权威成本表，但产生模拟成交的比较
区间不得早于成本制度首个有效交易日。

当前受管策略运行时为 `qlib-rdagent-single-mainline-2026-09-06-v38`：Qlib 持仓的内部数量
乘信号日已知 factor 才是等价原始股数，持仓估值必须与 NAV 使用同一价格。当日行情缺失时，
数量换算只复用信号时点之前最后有效 factor；不填充报价、成交量或可交易状态，也不读取未来值。
完全没有历史因子证据时仍阻断。原价分钟执行
采用 factor=1，日线历史代理仍采用日线复权口径，不能根据信号频率猜测账户单位。申报单位约束
作用于本次买卖增量，合法持仓保持不动时不重新取整。策略校验后的数量直接生成订单，
禁止再按另一个价格或默认风险比例缩放；容量按等价股数变化乘已知原始参考价复检。
仅保留 Qlib 原有的完整清仓特例，T+1、容量、现金、持仓等硬约束继续复检。这项修复
针对 Qlib 回测执行边界；推荐目标历史与模拟账本的真实持仓仍须分别核验，不能相互冒充。
该执行修复会改变交易结果，必须重新计算受影响回测，不能沿用旧结果宣称等价。
v36 的 PIT 证券资格和 Qlib 截面查找
准备机制继续复用固定输入。可转债评级 `cb_rating` 继续按日刷新请求桶，同一桶复用检查点，
新桶追加请求，保留原查询参数和历史快照。
风格数据门仍仅适用于策略实际消费的输入；模拟订单仍按统一账户时钟共享同证券同分钟容量。
本次发布新增数据库迁移 `0109_strategy_runtime_v38`，绑定 runner、完整源码闭包和
精确 worker 镜像身份。v37 及更早身份和证据保持只读，不得改写旧版本或拼接模拟阶段，
不得让旧冻结任务改用新运行时，也不得重用已经消费的正式 OOS。

Autopilot 的现行权限到 `fin_factor/fin_model/fin_quant` 研究、独立准入和只读冠军选择
为止；它不会创建 `StrategyVersion`、正式 OOS、批准或模拟账户。旧
`AutopilotCapitalPipeline` 代码已物理删除；数据库中已有的 `capital_pipeline` 状态统一
标记为 `legacy_readonly`，只保留历史查询和审计，API、scheduler 和 Autopilot 不存在
实例化、推进或补跑入口。唯一自动资本入口是受管 `fin_strategy` 的 settlement：规则候选依次完成
`policy_only → full_stack → 正式 OOS → paper_validating`，任一步失败都保持现金且不会
退回旧资本链。旧的 standalone transparent-baseline bootstrap 已随退役线物理删除；
三套公开基线只在同一个 `fin_strategy` 竞赛里充当对照组。
简单模式先输出冲突净额后的唯一账户操作清单，三周期结果仅作为可折叠来源解释。
平衡型个人账户的默认周期预算为短线 20%、中线 50%、长线 30%，再统一应用
账户现金、单票、行业和总风险上限；缺失周期的预算保留现金，不向其他周期重分配。
平台不公开官方 `server_ui`，高级页面从不可变、验签后的制品只读展示真实 RDLoop 的
Loop、Hypothesis、Feedback 和 Trace 摘要；原始 pickle、路径、代码和凭据不对 Web
开放。诊断按钮也不替代生产 readiness。项目不连接真实券商，模拟结果不会自动触发实盘。

研究与任务页面将当前运行与历史记录分开显示；当前阶段只关联同一研究的真实执行任务，
子任务成功不代表整条研究完成。最近结束的研究结果保持可见；运行中已有试验失败会单独说明，
进度数量只来自当前任务绑定的证据。刷新失败时显示“上次状态”与上次成功读取时间，
不会把缓存记录冒充最新进展。公开失败信息采用固定安全原因；有明确取消证据的
维护中止保留原始审计状态，页面只解释为“已中止”。这些展示修复
不改变金融权限、运行时身份或 OOS 边界，也不授予模拟或荐股资格。

旧 `/api/research-programs` 与 `/api/research-campaigns` 端点已物理删除，历史记录只保留
在数据库表中只读可查；上游因子/模型/联合研究只由 `/api/autopilot` 编排，
受管 `fin_strategy` 调度在同一主线中承接唯一策略资本结算，不另建控制面。配对交易代码已
物理删除，生产 API、scheduler 和 worker 不存在配对回测、影子账户或订单入口，数据库
历史记录保留只读。

## Docker 部署

```powershell
Copy-Item deploy\.env.example deploy\.env
.\.venv\Scripts\python.exe scripts\configure_tushare.py --env-file deploy\.env
docker compose --env-file deploy\.env -f deploy\compose.yaml up -d --build
docker compose --env-file deploy\.env -f deploy\compose.yaml ps
```

默认入口为 `http://127.0.0.1:38080`。首次访问时创建唯一初始管理员；仓库和镜像
都不包含默认密码。完整迁移、健康检查、备份恢复和安全升级步骤见
[部署手册](docs/DEPLOYMENT.md)。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
pnpm --dir web run lint
pnpm --dir web run test
```

产品语义变更必须直接更新根目录权威 Markdown，并通过文档治理测试。
仅在明确需要发布新版 Word 快照时才导出 DOCX；日常修订不依赖 LibreOffice。
命令、接口或部署方式变化时才更新本 README 和部署手册。

## 文档

- [产品、策略和风险基准 Markdown](%E4%B8%AA%E4%BA%BA%E9%87%8F%E5%8C%96%E6%8A%95%E8%B5%84%E4%B8%8E%E6%A8%A1%E6%8B%9F%E7%9B%98%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1%E7%A8%BF.md)
- [3.0 定稿 DOCX 快照](%E5%A6%82%E4%BD%95%E6%90%AD%E5%BB%BA%E8%87%AA%E5%B7%B1%E7%9A%84%E9%87%8F%E5%8C%96%E4%BA%A4%E6%98%93%E7%B3%BB%E7%BB%9F.docx)
- [部署手册](docs/DEPLOYMENT.md)
