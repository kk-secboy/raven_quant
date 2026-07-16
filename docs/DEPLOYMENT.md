# QuantLab 部署手册

本文只记录安装、配置、迁移、启动和运维命令。产品、策略与风险要求以
[根目录 Markdown](../%E5%A6%82%E4%BD%95%E6%90%AD%E5%BB%BA%E8%87%AA%E5%B7%B1%E7%9A%84%E9%87%8F%E5%8C%96%E4%BA%A4%E6%98%93%E7%B3%BB%E7%BB%9F.md)
为准，技术实现以 Qlib/RD-Agent 为准。

## 1. 安装

### 本地开发

需要 Python 3.11+、PostgreSQL、Node.js，以及运行 RD-Agent 沙箱时所需的 Docker。

```powershell
cd E:\projects\rdagent-python
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

### Docker Compose

部署需要 Docker Engine、Compose v2 和私有 RD-Agent DinD 沙箱所需的 privileged
container 支持。复制示例配置：

```powershell
Copy-Item deploy\.env.example deploy\.env
```

Compose 将 PostgreSQL、API、调度器、Qlib Worker、RD-Agent Worker、私有沙箱、
Web 和同源网关隔离运行。持久数据保存在 PostgreSQL 与 Compose 数据卷中。

## 2. 配置

必须配置：

- `DATABASE_URL` 或 Compose 中的 PostgreSQL 密码；
- 唯一且长期保存的 `PLATFORM_SECRET_KEY`；
- Tushare API 地址和 Token；
- RD-Agent 使用的模型凭据（只在启用自动研究时需要）。

使用隐藏输入辅助脚本校验并写入 Tushare Token：

```powershell
.\.venv\Scripts\python.exe scripts\configure_tushare.py --env-file .env
.\.venv\Scripts\python.exe scripts\configure_tushare.py --env-file deploy\.env
```

不要在命令行、日志或版本库中暴露密钥。`PLATFORM_SECRET_KEY` 需要另存于受保护的
恢复介质；数据库备份包含密文但不包含该密钥。部署到 HTTPS 反向代理后设置
`AUTH_COOKIE_SECURE=true`，否则入口只绑定 `127.0.0.1`。

## 3. 数据库迁移

本地环境：

```powershell
.\.venv\Scripts\quant-db.exe upgrade
```

Compose 环境启动时会先执行迁移。需要单独执行时使用：

```powershell
docker compose --env-file deploy\.env -f deploy\compose.yaml run --rm api quant-db upgrade
```

迁移是前向版本化操作。升级前必须完成预检和协调备份，不得在持久任务运行时重建
容器或手工修改数据库版本表。

## 4. 启动与停止

本地 API：

```powershell
.\.venv\Scripts\quant-web.exe --host 127.0.0.1 --port 8765
```

完整 Compose：

```powershell
docker compose --env-file deploy\.env -f deploy\compose.yaml up -d --build
docker compose --env-file deploy\.env -f deploy\compose.yaml ps
```

默认入口为 `http://127.0.0.1:38080`。首次访问创建初始管理员，系统没有默认账户
或默认密码。

安全停止前先确认任务队列和工作单元为空：

```powershell
.\.venv\Scripts\python.exe scripts\release_preflight.py
docker compose --env-file deploy\.env -f deploy\compose.yaml down
```

预检未通过时不得用重启绕过正在运行的下载、研究、回测或模拟任务。

## 5. 健康检查

```powershell
Invoke-RestMethod http://127.0.0.1:38080/api/health
Invoke-RestMethod http://127.0.0.1:38080/api/qlib/status
Invoke-RestMethod http://127.0.0.1:38080/api/rdagent/status
docker compose --env-file deploy\.env -f deploy\compose.yaml ps
```

`/api/health` 必须返回 `status=ok`，并确认 PostgreSQL、密钥解密和外部 Worker 模式；
Qlib 与 RD-Agent 状态端点必须报告固定上游版本和可用运行时。容器存活不能替代应用
健康与运营 readiness。控制台中的 readiness 仍会因数据覆盖、审批、模拟复核、
严重风险或血缘不一致而阻断。

查看关键日志：

```powershell
docker compose --env-file deploy\.env -f deploy\compose.yaml logs -f api scheduler worker rdagent-worker
```

## 6. 备份与恢复

协调备份会在确认持久任务空闲后保存 PostgreSQL 和数据卷，并保留最新 14 份：

```powershell
.\scripts\backup.ps1 -BackupRoot E:\quantlab-backups -RetentionCount 14
```

Linux：

```bash
python scripts/backup.py --backup-root /opt/quantlab-backups --retention-count 14
```

将完成的备份目录复制到独立存储，并单独保存正确的 `PLATFORM_SECRET_KEY`。恢复属于
破坏性操作，必须显式确认；工具在停止写入服务前验证清单、校验和与密钥指纹：

```powershell
.\scripts\restore.ps1 `
  -BackupDirectory E:\quantlab-backups\quantlab-YYYYMMDDTHHMMSSZ `
  -ConfirmRestore
```

Linux：

```bash
python scripts/restore.py \
  --backup-directory /opt/quantlab-backups/quantlab-YYYYMMDDTHHMMSSZ \
  --confirm-restore
```

迁移或存储实现变化后，在隔离 Compose 项目中演练恢复：

```powershell
.\.venv\Scripts\python.exe scripts\restore_drill.py
```

## 7. 升级与回滚

对现有安装先运行只读、失败关闭的预检：

```powershell
.\.venv\Scripts\python.exe scripts\release_preflight.py `
  --report artifacts\release-preflight.json
```

预检检查 Compose 配置、服务健康、数据库迁移路径、持久任务和磁盘空间。通过后使用
受支持的升级工具；它会构建镜像、协调备份、迁移、健康检查，并在失败时恢复之前的
数据和镜像：

```powershell
.\.venv\Scripts\python.exe scripts\release_upgrade.py `
  --backup-root E:\quantlab-backups `
  --confirm-upgrade
```

先在隔离项目中演练同一路径：

```powershell
.\.venv\Scripts\python.exe scripts\release_upgrade_drill.py
```

不要对已有安装直接运行无预检的 `docker compose up --build`，也不要通过删除卷、
回退 Alembic 版本或覆盖数据库来制造“成功”。自动回滚失败时保持服务停止，保存日志
和预检报告，再从已验证备份恢复。

## 8. 常见排障

- **API 503：**检查 PostgreSQL、`PLATFORM_SECRET_KEY` 和所有已保存密文是否可解密。
- **Qlib/RD-Agent 状态不可用：**查看对应 Worker 日志；不要把监听端口当作运行时可用。
- **Tushare 任务失败：**重新运行 Token 校验，检查权限、限流和持久 checkpoint；不得
  切换数据商绕过缺失权限。
- **任务长期运行：**从控制台检查 lease、子任务和幂等键；先诊断再重启服务。
- **升级预检失败：**读取 JSON 报告中的独立 blocker，逐项处理，不要跳过协调备份。
- **恢复后不健康：**核对密钥指纹、数据卷、数据库 revision 和 Worker 固定版本。

## 9. 可选 QMT 附录

QMT 插件默认关闭，不属于研究、回测、审批、分配或模拟主线。不启用 QMT，不删除
插件代码。页面、调度器和模拟任务不得向 QMT 或任何券商网关发单；系统也不提供
实盘模式。

只有隔离的运维验收需要验证插件自身时，才可复制 `deploy/qmt-gateway.env.example`，
保持沙箱环境并在 Windows MiniQMT 主机上手工启动：

```powershell
Copy-Item deploy\qmt-gateway.env.example deploy\qmt-gateway.env
.\scripts\start_qmt_gateway.ps1
```

验收完成后停止插件并恢复默认关闭状态。插件不得被 Web、调度器或模拟任务自动启动，
也不得接收上述主线产生的订单。

返回 [项目入口](../README.md)。
