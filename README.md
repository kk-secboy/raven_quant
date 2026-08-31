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
全 A 股 5 分钟增量任务按成功 checkpoint 的实际交易日覆盖复用旧单元；每个尚未覆盖的
连续区间按自然季度合并且单次不超过 150 个交易日。这样补齐一个 14 交易日缺口时，
每只股票只请求一个区间，下一交易日仍只新增后缀，既不重下历史也不改变旧成功单元键。

## RD-Agent 研究中心

平台在同一套不可变数据、作业、制品和审计框架中接入八个场景：

- `fin_factor`：因子提出、实现和迭代；候选仍须独立复算、PIT 检查和三窗口准入。
- `fin_model`：在固定特征集上研究预测模型；使用固定随机种子重新训练并独立验证。
- `fin_quant`：联合研究完整的因子集与模型 bundle，并强制因子、模型、联合三组消融。
- `fin_strategy`：研究股票资格、市场状态、排名、入场、退出、持有期、组合和风险规则；输出 `StrategyProposal` 与确定性编译的规则 IR，仅有研究权限。
- `fin_factor_report`：只读取已验签且满足可用时点的研报 PDF，提取因子后复用因子门禁。
- `general_model`：从论文实现模型，初始状态仅为 `implementation_ready`；兼容实现可人工送入模型门禁。
- `data_science`：隔离运行通用数据科学任务，制品只进入实验室。
- `llm_finetune`：只在密封模型/数据、固定镜像和合格 GPU 能力同时满足时运行，制品不属于交易模型。

官方 RD-Agent 运行成绩和 Trace SOTA 只作为研究反馈，不能直接晋级。因子、模型和
策略候选必须通过 QuantLab 的独立 Qlib 评价、密封最终 OOS 和正式回测后，才会进入
隔离模拟盘；满足各周期真实前向门后由系统原子自动晋级，人工只能紧急暂停或回滚。
Alpha20/158/360 只是基础候选特征；每周期只冻结一个由基础特征与已证明增量价值的
RD-Agent 因子组成的冠军因子包，并限制为一个活动策略和最多两个隔离影子挑战者。
简单模式先输出冲突净额后的唯一账户操作清单，三周期结果仅作为可折叠来源解释。
平衡型个人账户的默认周期预算为短线 20%、中线 50%、长线 30%，再统一应用
账户现金、单票、行业和总风险上限；缺失周期的预算保留现金，不向其他周期重分配。
平台不公开官方 `server_ui`，高级页面从不可变、验签后的制品只读展示真实 RDLoop 的
Loop、Hypothesis、Feedback 和 Trace 摘要；原始 pickle、路径、代码和凭据不对 Web
开放。诊断按钮也不替代生产 readiness。项目不连接真实券商，模拟结果不会自动触发实盘。

旧 `/api/research-programs` 与 `/api/research-campaigns` 只保留历史 GET 查询，全部写入、
状态修改、重试和调度入口返回 410；唯一自动研究入口是 `/api/autopilot`。配对交易算法
只保留离线研究代码，生产 API、scheduler 和 worker 均不能创建配对回测、影子账户或订单。

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
