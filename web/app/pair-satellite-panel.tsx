"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

import { apiFetch } from "./api-client";

type PairDefinition = {
  asset_class: string;
  legs: string[];
  shorting_mode: string;
};

type StrategyVersion = {
  id: string;
  status: string;
  created_at: string;
  spec_json?: { pair?: PairDefinition };
};

type Strategy = {
  id: string;
  name: string;
  description?: string | null;
  versions: StrategyVersion[];
};

type Backtest = {
  id: string;
  strategy_version_id: string;
  status: string;
  metrics_json?: Record<string, number | string | boolean | null>;
  error?: string | null;
  created_at: string;
};

type Dataset = {
  id: string;
  name: string;
  frequency: string;
};

type Snapshot = {
  id: string;
  dataset_id: string;
  status: string;
  created_at: string;
};

const PAIR_MINUTE_DATASETS = new Set([
  "a_share_minute_bars",
  "etf_minute_bars",
  "index_minute_bars",
  "futures_minute_bars",
]);

function pct(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
}

function decimal(value: unknown, digits = 3) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

function strategyTitle(strategy: Strategy, version: StrategyVersion) {
  const legs = version.spec_json?.pair?.legs?.join(" / ");
  return legs ? `${strategy.name} · ${legs}` : strategy.name;
}

export function PairSatellitePanel({ api }: { api: string }) {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [backtests, setBacktests] = useState<Backtest[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [selectedVersion, setSelectedVersion] = useState("");
  const [name, setName] = useState("ETF 相对价值研究");
  const [description, setDescription] = useState("双腿价差、协整与容量约束的研究卫星。");
  const [leftLeg, setLeftLeg] = useState("510300");
  const [rightLeg, setRightLeg] = useState("510500");
  const [assetClass, setAssetClass] = useState("etf");
  const [dailyDataset, setDailyDataset] = useState("");
  const [executionSnapshot, setExecutionSnapshot] = useState("");
  const [minuteDataset, setMinuteDataset] = useState("");
  const [shortabilityDataset, setShortabilityDataset] = useState("");
  const [startDate, setStartDate] = useState("2023-01-01");
  const [endDate, setEndDate] = useState("2025-12-31");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  async function load() {
    const [strategyRows, backtestRows, datasetRows, snapshotRows] = await Promise.all([
      apiFetch<Strategy[]>(api, "/api/strategies"),
      apiFetch<Backtest[]>(api, "/api/backtests"),
      apiFetch<Dataset[]>(api, "/api/qlib/datasets"),
      apiFetch<Snapshot[]>(api, "/api/snapshots"),
    ]);
    const pairRows = strategyRows.filter((strategy) =>
      strategy.versions.some((version) => Boolean(version.spec_json?.pair)),
    );
    setStrategies(pairRows);
    setBacktests(backtestRows);
    setDatasets(datasetRows);
    setSnapshots(snapshotRows);
    if (!selectedVersion) {
      setSelectedVersion(pairRows[0]?.versions[0]?.id ?? "");
    }
    if (!dailyDataset) {
      setDailyDataset(datasetRows.find((dataset) => !PAIR_MINUTE_DATASETS.has(dataset.name))?.id ?? "");
    }
    if (!minuteDataset) {
      setMinuteDataset(datasetRows.find((dataset) => PAIR_MINUTE_DATASETS.has(dataset.name))?.id ?? "");
    }
    if (!executionSnapshot) {
      setExecutionSnapshot(snapshotRows.find((snapshot) => snapshot.status === "ready")?.id ?? "");
    }
  }

  useEffect(() => {
    // Catalog state is updated only after the asynchronous API requests resolve.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load().catch((error) => setMessage(error instanceof Error ? error.message : String(error)));
    // The panel owns its refresh lifecycle; changing defaults must not refetch the catalog.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  const versions = useMemo(
    () =>
      strategies.flatMap((strategy) =>
        strategy.versions
          .filter((version) => Boolean(version.spec_json?.pair))
          .map((version) => ({ strategy, version })),
      ),
    [strategies],
  );
  const selectedBacktests = backtests
    .filter((backtest) => backtest.strategy_version_id === selectedVersion)
    .sort((a, b) => b.created_at.localeCompare(a.created_at));
  const latest = selectedBacktests[0];
  const metrics = latest?.metrics_json ?? {};
  const readySnapshots = snapshots.filter((snapshot) => snapshot.status === "ready");

  async function createPair(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    try {
      await apiFetch(api, "/api/pair-strategies", {
        method: "POST",
        body: JSON.stringify({
          name,
          description,
          legs: [leftLeg.trim(), rightLeg.trim()],
          asset_class: assetClass,
          shorting_mode: "research_constraint",
          actor: "personal_researcher",
        }),
      });
      setMessage("研究版本已冻结。它只能进入研究回测，不会形成交易授权。");
      await load();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function runBacktest(event: FormEvent) {
    event.preventDefault();
    if (!selectedVersion) return;
    setBusy(true);
    setMessage("");
    try {
      await apiFetch(api, `/api/strategy-versions/${selectedVersion}/pair-backtests/`, {
        method: "POST",
        body: JSON.stringify({
          start_date: startDate,
          end_date: endDate,
          daily_dataset_id: dailyDataset || null,
          execution_snapshot_id: executionSnapshot || null,
          minute_dataset_id: minuteDataset || null,
          shortability_dataset_id: shortabilityDataset || null,
          actor: "personal_researcher",
        }),
      });
      setMessage("研究回测已提交。结果将保留数据、成本、容量和可做空约束的证据链。");
      await load();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel pair-lab">
      <div className="panel-head">
        <div>
          <p className="eyebrow">PAIR RESEARCH SATELLITE</p>
          <h2>双腿相对价值研究</h2>
          <p className="panel-copy">
            从不可变策略版本到协整、成本、容量和稳健性证据。这里只保存研究结论，
            不得批准、创建持久模拟账户，也不会生成真实交易指令。
          </p>
        </div>
        <span className="boundary-chip">RESEARCH ONLY</span>
      </div>

      <div className="pair-metric-strip">
        <article>
          <span>相关系数</span>
          <strong>{decimal(metrics.correlation)}</strong>
          <small>收益联动</small>
        </article>
        <article>
          <span>协整 P 值</span>
          <strong>{decimal(metrics.cointegration_pvalue)}</strong>
          <small>长期关系</small>
        </article>
        <article>
          <span>年化 Sharpe</span>
          <strong>{decimal(metrics.sharpe_ratio, 2)}</strong>
          <small>成本后</small>
        </article>
        <article>
          <span>最大回撤</span>
          <strong>{pct(metrics.max_drawdown)}</strong>
          <small>研究样本</small>
        </article>
        <article>
          <span>容量占用</span>
          <strong>{pct(metrics.capacity_fill_ratio)}</strong>
          <small>成交约束</small>
        </article>
      </div>

      <div className="pair-workbench">
        <form className="form-card" onSubmit={createPair}>
          <div className="form-card-head">
            <div>
              <span className="step-index">01</span>
              <h3>冻结研究定义</h3>
            </div>
            <span className="status-chip">不可变版本</span>
          </div>
          <label>
            研究名称
            <input value={name} onChange={(event) => setName(event.target.value)} required />
          </label>
          <label>
            研究说明
            <textarea value={description} onChange={(event) => setDescription(event.target.value)} />
          </label>
          <div className="split-fields">
            <label>
              左腿
              <input value={leftLeg} onChange={(event) => setLeftLeg(event.target.value)} required />
            </label>
            <label>
              右腿
              <input value={rightLeg} onChange={(event) => setRightLeg(event.target.value)} required />
            </label>
          </div>
          <label>
            资产类别
            <select value={assetClass} onChange={(event) => setAssetClass(event.target.value)}>
              <option value="etf">ETF</option>
              <option value="a_share">A 股</option>
              <option value="index_futures">股指期货</option>
            </select>
          </label>
          <button type="submit" disabled={busy}>
            冻结研究版本
          </button>
        </form>

        <form className="form-card" onSubmit={runBacktest}>
          <div className="form-card-head">
            <div>
              <span className="step-index">02</span>
              <h3>运行约束回测</h3>
            </div>
            <span className="status-chip">证据优先</span>
          </div>
          <label>
            策略版本
            <select value={selectedVersion} onChange={(event) => setSelectedVersion(event.target.value)} required>
              <option value="">选择研究版本</option>
              {versions.map(({ strategy, version }) => (
                <option key={version.id} value={version.id}>
                  {strategyTitle(strategy, version)}
                </option>
              ))}
            </select>
          </label>
          <div className="split-fields">
            <label>
              开始日期
              <input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} />
            </label>
            <label>
              结束日期
              <input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} />
            </label>
          </div>
          <label>
            日频研究数据
            <select value={dailyDataset} onChange={(event) => setDailyDataset(event.target.value)}>
              <option value="">未选择</option>
              {datasets
                .filter((dataset) => !PAIR_MINUTE_DATASETS.has(dataset.name))
                .map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>{dataset.name} · {dataset.frequency}</option>
                ))}
            </select>
          </label>
          <label>
            执行快照
            <select value={executionSnapshot} onChange={(event) => setExecutionSnapshot(event.target.value)}>
              <option value="">未选择</option>
              {readySnapshots.map((snapshot) => (
                <option key={snapshot.id} value={snapshot.id}>
                  {snapshot.dataset_id.slice(0, 8)} · {snapshot.created_at.slice(0, 10)}
                </option>
              ))}
            </select>
          </label>
          <div className="split-fields">
            <label>
              分钟数据
              <select value={minuteDataset} onChange={(event) => setMinuteDataset(event.target.value)}>
                <option value="">未选择</option>
                {datasets
                  .filter((dataset) => PAIR_MINUTE_DATASETS.has(dataset.name))
                  .map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
                  ))}
              </select>
            </label>
            <label>
              可做空数据
              <select value={shortabilityDataset} onChange={(event) => setShortabilityDataset(event.target.value)}>
                <option value="">无 / 仅约束提示</option>
                {datasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>{dataset.name}</option>
                ))}
              </select>
            </label>
          </div>
          <button type="submit" disabled={busy || !selectedVersion}>
            提交研究回测
          </button>
        </form>
      </div>

      {message ? <p className="notice">{message}</p> : null}

      <div className="pair-results">
        <div className="section-heading">
          <div>
            <p className="eyebrow">AUDITABLE EVIDENCE</p>
            <h3>最近研究记录</h3>
          </div>
          <span>{selectedBacktests.length} 次回测</span>
        </div>
        {selectedBacktests.length ? (
          <div className="table-shell">
            <table>
              <thead>
                <tr>
                  <th>时间</th>
                  <th>状态</th>
                  <th>Sharpe</th>
                  <th>最大回撤</th>
                  <th>换手率</th>
                  <th>结果</th>
                </tr>
              </thead>
              <tbody>
                {selectedBacktests.slice(0, 8).map((backtest) => (
                  <tr key={backtest.id}>
                    <td>{new Date(backtest.created_at).toLocaleString("zh-CN")}</td>
                    <td><span className={`status status-${backtest.status}`}>{backtest.status}</span></td>
                    <td>{decimal(backtest.metrics_json?.sharpe_ratio, 2)}</td>
                    <td>{pct(backtest.metrics_json?.max_drawdown)}</td>
                    <td>{pct(backtest.metrics_json?.turnover)}</td>
                    <td>{backtest.error || "证据已留档"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <strong>还没有研究回测</strong>
            <span>先冻结双腿定义，再绑定数据快照提交约束回测。</span>
          </div>
        )}
      </div>
    </section>
  );
}
