# QuantLab 部署手册

本文只记录安装、配置、迁移、启动和运维命令。产品、策略与风险要求以
[根目录 Markdown](../%E4%B8%AA%E4%BA%BA%E9%87%8F%E5%8C%96%E6%8A%95%E8%B5%84%E4%B8%8E%E6%A8%A1%E6%8B%9F%E7%9B%98%E7%B3%BB%E7%BB%9F%E8%AE%BE%E8%AE%A1%E7%A8%BF.md)
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

### 评估任务资源

`evaluation-worker` 默认一次执行一个任务，容器限制为 8 CPU、40 GiB 内存和
512 个进程/线程。`memswap_limit=40g` 与内存上限相同，禁止该容器额外使用 swap。
部署环境显式设置 `QUANTLAB_QLIB_KERNELS=1`，限制 baseline 和正式回测的 Qlib
特征预处理进程数；升级前检查旧环境文件，已有的更大值会覆盖 Compose 默认值。
训练日期、市场、Alpha158 特征、模型参数及统计门槛不因资源限制而缩减。

Compose 中 OMP/MKL/OpenBLAS/NumExpr 的默认线程数为 1，仅限制 worker 本身与健康
探针。任务执行器仍按既有资源账本设置最多 8 个数值线程及 Linux CPU affinity，
baseline 的 LightGBM `num_threads=8` 保持不变。跨 worker 的研究预算仍为
24 CPU / 40 GiB；baseline 占满 40 GiB 预算，其他受该账本治理的重研究任务必须等待。
容器上限用于隔离资源耗尽，不保证全量训练一定能在上限内完成；运行时检查
`memory.current`、`memory.peak`、`memory.events` 和 `pids.current`，保留失败证据。

服务器重启不是取消任务。Worker 启动会将尚有尝试次数的中断任务重新排队并等待
120 秒，`safe_mode` 不阻止通用任务认领。恢复前先检查旧任务的状态及
`attempts/max_attempts`；若需禁止继续执行，用现有取消入口令该任务终结，并确认
没有残留进程，再启动受控的新版本。不要通过重试旧任务覆盖需保留的中断制品。

模型执行器的数据准备复用、空间预算、隔离验证与升级边界见
[模型数据准备复用](model-prepared-data.md)。

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

Scheduler 首次调度可能需要校验完整数据目录，Compose 为它保留 5 分钟启动宽限，
覆盖默认 `SCHEDULER_MAX_TICK_SECONDS=300` 的单轮预算。首轮未完成时 `/health`
仍返回 503；宽限不会把未就绪服务当作健康，也不会强制等待 5 分钟才放行。首轮成功后
即可通过检查；宽限结束后仍按每 10 秒检查、连续 12 次失败判为不健康。调整单轮预算时，
应同步审视启动宽限和发布等待时间，避免正常首次校验被发布流程提前判失败。

查看关键日志：

```powershell
docker compose --env-file deploy\.env -f deploy\compose.yaml logs -f api scheduler worker rdagent-worker
```

## 6. 备份与恢复

手工完整备份仍使用 v1：它会协调停止写入服务，保存 PostgreSQL 和完整数据卷，适合
离线迁移及恢复演练：

```powershell
.\scripts\backup.ps1 -BackupRoot E:\quantlab-backups -RetentionCount 14
```

Linux：

```bash
python scripts/backup.py \
  --backup-root /opt/quantlab-backups \
  --retention-count 14 \
  --format-version 1
```

生产每日定时备份使用 v2 控制面格式，只保存 PostgreSQL custom dump、脱敏部署配置和
不可变数据 manifest 清单；它明确不复制或恢复 `/data`。systemd 单元显式传入 `--online`，
因此 PostgreSQL 使用在线一致性 dump，清单作为在线快照采集，不停止 API、scheduler、
worker 或其他 `WRITER_SERVICES`，也不会中断正在运行的研究、回测和模拟任务。首次安装或
发布 systemd 单元后，以 root 从当前受控 release 执行：

```bash
sh /opt/quantlab/scripts/install_backup_service.sh /opt/quantlab
```

安装器创建稳定的 `/opt/quantlab-ops/venv`，按
`deploy/backup-ops-requirements.txt` 安装固定依赖，并启用持久 timer。定时任务使用专用
`backup_preflight.py`，只检查部署配置、Compose/PostgreSQL、备份目录边界、数据库大小及
10 GiB 保留空间；它不依赖业务 readiness、策略是否已进入 paper 或 release 根盘的
20 GiB 门槛。`--online` 只允许用于 v2；full v1 手工备份及发布升级创建的回滚备份仍会
协调停止写入服务，升级成功前也继续保持停写。v2 恢复只替换数据库，现有 `/data` 保持
原样；不可变数据必须由独立存储和 manifest 清单另行保障。

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

若预检报告旧容器来自多个 release/config 合同，不得删除卷、换 Compose project name
或直接覆盖数据库来“重跑”。先从待发布候选代码调用基线收敛器，但显式指向当前
`/opt/quantlab` 的受保护配置。第一次只做 dry-run；只有队列为空、回滚配置等价检查
通过时，才执行确认收敛：

```bash
CURRENT=$(readlink -f /opt/quantlab)
PY=/opt/quantlab-ops/venv/bin/python

$PY scripts/canonicalize_release_baseline.py \
  --project-name quantlab-platform \
  --env-file "$CURRENT/deploy/.env" \
  --compose-file "$CURRENT/deploy/compose.yaml" \
  --receipt-root /opt/quantlab-backups/canonical-baselines \
  --release-id canonical-pre-upgrade \
  --wait-timeout 900

$PY scripts/canonicalize_release_baseline.py \
  --project-name quantlab-platform \
  --env-file "$CURRENT/deploy/.env" \
  --compose-file "$CURRENT/deploy/compose.yaml" \
  --receipt-root /opt/quantlab-backups/canonical-baselines \
  --release-id canonical-pre-upgrade \
  --wait-timeout 900 \
  --confirm-convergence
```

收敛只按当前镜像重建无状态服务并留下回滚 receipt；它不迁移 PostgreSQL、不覆盖
`/data`、不清安全模式。随后把当前 `.env` 以 `0600` 权限复制进由正式提交生成的
不可变候选 release，再执行候选的预检、备份预检和升级。

```powershell
.\.venv\Scripts\python.exe scripts\release_upgrade.py `
  --backup-root E:\quantlab-backups `
  --stable-release-link /opt/quantlab `
  --confirm-upgrade
```

仅控制面修复需要保留当前模型计算环境时，可同时指定
`--preserve-model-sandbox-image <当前 digest 引用>` 和
`--preserve-model-sandbox-image-id <私有 Docker 中的实际 sha256 ID>`。
工具会核对当前 release、候选源码、已运行 worker 和模型沙箱内安装文件的计算依赖，
并在镜像准备阶段再次检查；源码、配置或镜像身份不一致则停止。
默认仍重建模型沙箱。此选项不跳过备份、RD-Agent/Qlib 镜像准备和发布验收，
也不改变任务回执本身的恢复条件。

发布通过后，若旧显式完整活动在模型竞赛成功后，因
`fin_quant incumbent prediction uses another label horizon` 阻断于研究启动前，
可从受保护生产环境运行 `scripts/recover_fin_quant_handoff.py`。
先用 `--source-cycle`、唯一的 `--event-key`、`--actor` 和 `--reason` 生成只读 JSON 计划；
核对源数据、预算、竞赛证据、发布身份和没有下游输出后，将该计划保存为文件，
以 `--execute --plan <文件> --plan-sha256 <文件字节的 SHA256>` 执行。
执行会重新锁定并验证状态，只创建一个带审计的后继活动。
先暂停同周期其他活动并排空发布队列，再正常恢复调度；不得重置旧任务 attempts，
不得修改旧终态活动，也不得直接调用实验脚本冒充受管调度。
该入口拒绝已产生联合研究输出、已有策略调度或已打开 OOS 的活动。

先在隔离项目中演练同一路径：

```powershell
.\.venv\Scripts\python.exe scripts\release_upgrade_drill.py
```

不要对已有安装直接运行无预检的 `docker compose up --build`，也不要通过删除卷、
`docker compose down -v`、更换 production project name、回退 Alembic 版本或覆盖数据库
来制造“成功”。正式升级不得使用 `--pull`、`--reuse-backup` 或 `--skip-stable-link`。
自动回滚失败时保持服务停止，保存日志和预检报告，再从已验证备份恢复。成功验收后
才由工具原子切换 `/opt/quantlab`；旧 release 和回滚镜像先保留，不再运行但也不立即删除。

## 8. 常见排障

- **API 503：**检查 PostgreSQL、`PLATFORM_SECRET_KEY` 和所有已保存密文是否可解密。
- **Qlib/RD-Agent 状态不可用：**查看对应 Worker 日志；不要把监听端口当作运行时可用。
- **Tushare 任务失败：**重新运行 Token 校验，检查权限、限流和持久 checkpoint；不得
  切换数据商绕过缺失权限。
- **任务长期运行：**从控制台检查 lease、子任务和幂等键；先诊断再重启服务。
- **升级预检失败：**读取 JSON 报告中的独立 blocker，逐项处理，不要跳过协调备份。
- **恢复后不健康：**核对密钥指纹、数据卷、数据库 revision 和 Worker 固定版本。

## 9. 真实交易边界

生产发布不打包 QMT 或任何券商网关，也不提供启动脚本、账户配置、Web 开关或自动
实盘模式。历史沙箱源码仅作为仓库审计材料保留，并由发布上下文明确排除。页面、
调度器、回测、推荐和模拟任务只能写入 QuantLab 的隔离模拟账本。

返回 [项目入口](../README.md)。
