# QuantLab

QuantLab 是面向 A 股中低频量化研究与模拟交易的本地优先平台。项目只使用
Tushare 数据，并以 Qlib 和 RD-Agent 组成唯一技术主线。

> **权威关系：**[根目录 Markdown](%E4%B8%AA%E4%BA%BA%E9%87%8F%E5%8C%96%E6%8A%95%E8%B5%84%E4%B8%8E%E6%A8%A1%E6%8B%9F%E7%9B%98%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1%E7%A8%BF.md)
> 是产品、策略和风险基准，Qlib/RD-Agent 是技术基准。本 README 只提供项目入口和
> 使用方法，不定义另一套产品方案。

根目录同名 DOCX 仅保留为 3.0 定稿发布快照，不再作为日常修改源。
根目录 `如何搭建自己的量化交易系统-GitHub原版副本.docx` 仅保留为冻结的历史原稿，
不属于现行产品基准，也不得作为第二套方案继续修订。

唯一生产主线是：

`Tushare 不可变快照 → Qlib 数据集 → RD-Agent 研究 → 独立复算与准入 → Qlib 正式回测 → 策略审批 → 买卖推荐 → 核心/卫星分配 → 统一模拟交易 → 表现复核与策略生命周期`

项目不做实盘、Tick、Level-2、逐笔或毫秒高频。QMT 仅作为默认关闭的可选插件保留；
页面、调度和模拟任务不得向 QMT 或任何券商网关发单。

## 代码入口

- `src/quant_data`：Tushare 下载、不可变快照、血缘和 Qlib 数据转换。
- `src/quant_platform`：研究编排、准入、Qlib 回测、审批、分配、模拟和运维 API。
- `src/quant_broker_gateway`：默认关闭的可选 QMT 沙箱插件，不在主线运行。
- `web`：数据、RD-Agent、因子准入、Qlib 回测、审批、分配和模拟控制台。
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
.\.venv\Scripts\quant-data.exe bootstrap --profile core --start 2018-01-01 --end latest
.\.venv\Scripts\quant-data.exe status
.\.venv\Scripts\quant-data.exe verify
```

Bootstrap、Qlib 转换、RD-Agent 研究和回测也可以从 Web 控制台创建为持久任务。
任务关闭浏览器后不会丢失。日线与分钟数据使用隔离的数据集和执行契约；15/30/60
分钟研究由 Qlib 从 1 分钟或 5 分钟数据重采样，不增加下载线路。

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
