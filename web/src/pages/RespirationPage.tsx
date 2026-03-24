import { useEffect, useState } from "react";

import Callout from "../components/Callout";
import EChart from "../components/EChart";
import SimpleTable from "../components/SimpleTable";
import { numberFormat, useJsonData } from "../lib";
import type { SiteManifest } from "../types";

function comparisonOption(rows: Record<string, any>[]) {
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0 },
    grid: { left: 48, right: 20, top: 42, bottom: 56 },
    xAxis: { type: "category", data: rows.map((row) => `${row.method} / ${row.subset}`), axisLabel: { rotate: 20 } },
    yAxis: [
      { type: "value", name: "MAE / RMSE" },
      { type: "value", name: "Within 3 bpm", min: 0, max: 1 },
    ],
    series: [
      { name: "MAE", type: "bar", data: rows.map((row) => row.resp_mae_bpm) },
      { name: "RMSE", type: "bar", data: rows.map((row) => row.resp_rmse_bpm) },
      { name: "Within 3 bpm", type: "line", yAxisIndex: 1, data: rows.map((row) => row.within_3_bpm_rate) },
    ],
  };
}

function scatterOption(rows: Record<string, any>[]) {
  return {
    tooltip: { trigger: "item" },
    legend: { top: 0 },
    grid: { left: 52, right: 18, top: 36, bottom: 46 },
    xAxis: { type: "value", name: "Reference RR (bpm)" },
    yAxis: { type: "value", name: "Predicted RR (bpm)" },
    series: [
      { name: "Surrogate baseline", type: "scatter", symbolSize: 8, data: rows.map((row) => [row.resp_rate_ref_bpm, row.resp_rate_baseline_bpm]) },
      { name: "CNN", type: "scatter", symbolSize: 8, data: rows.map((row) => [row.resp_rate_ref_bpm, row.resp_rate_pred_bpm]) },
    ],
  };
}

function confidenceOption(rows: Record<string, any>[]) {
  return {
    tooltip: { trigger: "item" },
    grid: { left: 52, right: 18, top: 24, bottom: 46 },
    xAxis: { type: "value", name: "Resp confidence" },
    yAxis: { type: "value", name: "Absolute error (bpm)" },
    series: [{ type: "scatter", symbolSize: 8, data: rows.map((row) => [row.resp_confidence, row.cnn_abs_error]) }],
  };
}

function tuningOption(rows: Record<string, any>[]) {
  const sorted = [...rows].sort((a, b) => Number(a.high_quality_resp_mae_bpm) - Number(b.high_quality_resp_mae_bpm));
  return {
    tooltip: { trigger: "axis" },
    grid: { left: 48, right: 20, top: 18, bottom: 70 },
    xAxis: {
      type: "category",
      data: sorted.slice(0, 16).map((row) => `${row.window_seconds}s / ${row.channel_set}`),
      axisLabel: { rotate: 28 },
    },
    yAxis: { type: "value", name: "HQ RR MAE" },
    series: [{ type: "bar", data: sorted.slice(0, 16).map((row) => row.high_quality_resp_mae_bpm) }],
  };
}

function rrLineOption(rows: Record<string, any>[]) {
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0 },
    grid: { left: 52, right: 18, top: 42, bottom: 42 },
    xAxis: { type: "category", data: rows.map((row) => `t=${Number(row.start_time_s).toFixed(0)}`), axisLabel: { interval: Math.max(1, Math.floor(rows.length / 12)) } },
    yAxis: { type: "value", name: "RR (bpm)" },
    series: [
      { name: "Reference", type: "line", smooth: true, data: rows.map((row) => row.resp_rate_ref_bpm) },
      { name: "Baseline", type: "line", smooth: true, data: rows.map((row) => row.resp_rate_baseline_bpm) },
      { name: "CNN", type: "line", smooth: true, data: rows.map((row) => row.resp_rate_pred_bpm) },
    ],
  };
}

export default function RespirationPage({
  dataset,
  scope,
  manifest,
}: {
  dataset: string;
  scope: string;
  manifest: SiteManifest | null;
}) {
  const stage5 = useJsonData<Record<string, any>>("stage_metrics/stage5.json");
  const timelineOptions = manifest?.timeline_index?.stage5?.[dataset] ?? [];
  const [selectedTimelinePath, setSelectedTimelinePath] = useState<string | null>(timelineOptions[0]?.path ?? null);
  const timeline = useJsonData<Record<string, any>>(selectedTimelinePath);

  useEffect(() => {
    if (timelineOptions.length && !timelineOptions.some((entry) => entry.path === selectedTimelinePath)) {
      setSelectedTimelinePath(timelineOptions[0].path);
    }
  }, [timelineOptions, selectedTimelinePath]);

  if (stage5.loading) return <div className="page-shell">Loading Stage 5 charts…</div>;
  if (stage5.error || !stage5.data) return <div className="page-shell">Unable to load Stage 5 data.</div>;

  const datasetBlock = stage5.data.datasets?.[dataset] ?? {};
  const metricRows = (datasetBlock.metrics ?? []).filter((row: Record<string, any>) => row.split === "eval");

  return (
    <div className="page-shell">
      <Callout title="Stage 5 interpretation constraints" tone="info">
        Stage 5 uses chest Resp references from the public datasets. It extends Stage 4 context into a respiration-window
        multitask interface; it does not replace HR estimation.
      </Callout>

      {scope !== "canonical" ? (
        <Callout title="Canonical-first respiration page" tone="warning">
          Stage 5 charts stay canonical here. Validation and tuning details still appear in Experiments.
        </Callout>
      ) : null}

      <section className="content-card">
        <h3>Baseline vs CNN</h3>
        <EChart option={comparisonOption(metricRows)} />
      </section>

      <section className="two-column-grid">
        <div className="content-card">
          <h3>Reference vs Prediction Scatter</h3>
          <EChart option={scatterOption(datasetBlock.scatter_sample ?? [])} />
        </div>
        <div className="content-card">
          <h3>Confidence vs CNN Error</h3>
          <EChart option={confidenceOption(datasetBlock.scatter_sample ?? [])} />
        </div>
      </section>

      <section className="two-column-grid">
        <div className="content-card">
          <h3>Tuning Snapshot</h3>
          <EChart option={tuningOption(datasetBlock.tuning_rows ?? [])} />
        </div>
        <div className="content-card">
          <h3>Selected CNN Configuration</h3>
          <SimpleTable
            columns={[
              { key: "key", label: "Field" },
              { key: "value", label: "Value" },
            ]}
            rows={Object.entries(datasetBlock.best_config ?? {}).map(([key, value]) => ({ key, value }))}
          />
        </div>
      </section>

      <section className="content-card">
        <div className="section-header-inline">
          <h3>Respiration Timeline</h3>
          <select value={selectedTimelinePath ?? ""} onChange={(event) => setSelectedTimelinePath(event.target.value)}>
            {timelineOptions.map((entry) => (
              <option key={entry.path} value={entry.path}>
                {entry.split} / {entry.subject_id}
              </option>
            ))}
          </select>
        </div>
        {timeline.loading ? (
          <div className="empty-state">Loading respiration timeline…</div>
        ) : timeline.error || !timeline.data ? (
          <div className="empty-state">Timeline unavailable.</div>
        ) : (
          <EChart option={rrLineOption(timeline.data.rows ?? [])} height={420} />
        )}
      </section>

      <section className="content-card">
        <h3>Eval Metric Table</h3>
        <SimpleTable
          columns={[
            { key: "method", label: "Method" },
            { key: "subset", label: "Subset" },
            { key: "resp_mae_bpm", label: "MAE", render: (value) => numberFormat(value) },
            { key: "resp_rmse_bpm", label: "RMSE", render: (value) => numberFormat(value) },
            { key: "resp_pearson_r", label: "Pearson r", render: (value) => numberFormat(value) },
            { key: "within_3_bpm_rate", label: "Within 3 bpm", render: (value) => numberFormat(value) },
          ]}
          rows={metricRows}
        />
      </section>
    </div>
  );
}
