# Qlib 与 RD-Agent 官方能力对照（2026-08-25）

## 对照版本

- Qlib 固定提交：`d5379c520f66a39953bad76234a7019a72796fd0`
- RD-Agent 固定提交：`4f9ecb005881cddc08df0124a2e894c018007679`

本表以固定源码、官方论文和官方文档为准，不按前端是否有按钮推断能力。

官方资料：

- [Qlib 论文](https://arxiv.org/abs/2009.11189)
- [Qlib 官方仓库](https://github.com/microsoft/qlib)
- [Qlib 组件目录](https://qlib.readthedocs.io/en/latest/)
- [RD-Agent 官方仓库](https://github.com/microsoft/RD-Agent)
- [RD-Agent(Q) 论文](https://arxiv.org/abs/2505.15155)

## Qlib

| 官方能力 | 当前项目 | 判断与下一步 |
| --- | --- | --- |
| Qlib bin 数据、DataHandler、表达式引擎 | 已接入 | 继续使用平台 PIT 字段和受限表达式编译器 |
| Alpha158 / Alpha360 | 已进入统一因子库 | 前端已明确区分 Alpha158 探索基线与 Alpha158/360/SOTA 受治理特征集 |
| 自定义公式因子 | 已接入 | RD-Agent 公式必须经白名单编译和独立复算 |
| Workflow / Recorder | 已接入 | 正式 Qlib 运行强制绑定 Recorder 和数据身份 |
| 多种监督模型 | 部分接入 | 固定基线只有 LightGBM；`fin_model/fin_quant` 可产生模型挑战者，但未进入默认持续研究 |
| TopkDropoutStrategy | 已接入 | 只适合作基线；正式策略使用受治理策略和执行合同 |
| EnhancedIndexingStrategy / 风险模型 | 部分接入 | 项目已有基准相对优化和 Qlib 风险模型适配，应纳入正式能力验收 |
| Portfolio / Backtest | 已接入 | 官方回测之外再加 PIT、统计门、真实费用和模拟差分 |
| NestedExecutor / 高频联合执行 | 已接入日/分钟路径 | 只在原生分钟数据和执行证据完整时启用 |
| Online Serving / RollingStrategy | 未直接采用 | 项目用自己的不可变 ModelArtifact、调度器和推荐系统替代；需要补齐自动滚动重训对照测试 |
| 市场动态模型 | 未成为默认主线 | 可作为 `fin_model` 候选，不应未经独立 OOS 直接上线 |
| RL 订单执行 | 未接入 | 当前个人日频/分钟数据和模拟目标不需要优先接入；先完成传统执行前向验证 |
| Meta Controller / Meta-learning | 未接入 | 属研究实验室能力，不是当前可靠模拟盘的必要条件 |
| 报告分析与可视化 | 部分接入 | 指标已保存，官方分析图和模型诊断尚未统一到 Web |

结论：Qlib 不是“只用了 Alpha158”，但默认页面给人的确是这个印象。核心数据、
表达式、Recorder、回测、风险和分钟执行已经接入；多模型、滚动训练和部分高级研究
能力没有成为默认自动主线。

## RD-Agent

| 官方入口 | 当前项目 | 主要缺口 |
| --- | --- | --- |
| `fin_factor` | 已接入并进入持续研究 | 当前自动驾驶主线使用它 |
| `fin_model` | 已接入运行、制品和独立三窗口评估 | 尚未进入持续自动研究轮换 |
| `fin_quant` | 已接入完整 bundle 和消融评估 | 尚未进入持续自动研究轮换 |
| `fin_factor_report` | 已接入受管 PDF 和独立因子门 | 需要按研报到达量自动安排预算 |
| `general_model` | 已接入实验室 | 产物必须转入 `fin_model` 门，不能直接投资 |
| `data_science` | 已接入隔离队列 | 与投资链保持隔离是正确的 |
| `llm_finetune` | 代码入口已接入、服务器无 GPU | 不应为了“功能全”用 CPU 慢速兜底 |
| `health_check` | 已接入诊断 | 平台 readiness 仍是生产标准 |
| `ui/server_ui` | 未独立部署 | trace 已并入 QuantLab，避免多一套无鉴权 UI |

结论：七个官方主要 CLI 入口已经封装，但“能手工启动”不等于“自动驾驶会合理轮换”。
下一步是把 `fin_model`、`fin_quant`、`fin_factor_report` 作为有预算、低频率的挑战者
周期接入，而不是每天全部并发烧钱。

## 默认自动驾驶选择

第一版自动驾驶采用一条主线：

`33项数据更新 → 冻结Qlib数据集 → fin_factor持续研究 → 独立复算/去重/SOTA → 参数实验 → 正式回测 → 人工确认 → 模拟盘`

高级能力按计划轮换：

- 每月或数据结构显著变化时运行 `fin_model`；
- 因子与模型都有合格挑战者时运行 `fin_quant`；
- 新研报通过资料治理后运行 `fin_factor_report`；
- RL、Meta-learning、LLM 微调不进入第一版资金主线。

这不是放弃官方能力，而是把研究成本、数据条件和投资权限分层。
