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
    grid: { left: 48, right: 20, top: 42, bottom: 48 },
    xAxis: { type: "category", data: rows.map((row) => row.method), axisLabel: { rotate: 18 } },
    yAxis: [
      { type: "value", name: "AUPRC" },
      { type: "value", name: "AUROC", min: 0, max: 1 },
    ],
    series: [
      { name: "AUPRC", type: "bar", data: rows.map((row) => row.auprc) },
      { name: "AUROC", type: "line", yAxisIndex: 1, data: rows.map((row) => row.auroc) },
    ],
  };
}

function heatmapOption(rows: Record<string, any>[]) {
  const tracks = [
    ["quality_gate_passed", "Gate"],
    ["hr_event_flag", "Event"],
    ["irregular_pulse_flag", "Irregular"],
    ["anomaly_flag", "Anomaly"],
    ["stage4_suspicion_flag", "Suspicion"],
    ["ml_signal_quality_score", "ML quality"],
    ["irregular_pulse_score", "Irregular score"],
    ["anomaly_score", "Anomaly score"],
    ["stage4_suspicion_score", "Suspicion score"],
  ] as const;
  const heatmapRows: Array<[number, number, number]> = [];
  rows.forEach((row, columnIndex) => {
    tracks.forEach(([field], rowIndex) => {
      const value = Number(row[field] ?? 0);
      heatmapRows.push([columnIndex, rowIndex, Number.isFinite(value) ? value : 0]);
    });
  });
  return {
    tooltip: { position: "top" },
    grid: { left: 88, right: 18, top: 16, bottom: 50 },
    xAxis: {
      type: "category",
      data: rows.map((row) => `t=${Number(row.start_time_s).toFixed(0)}`),
      axisLabel: { interval: Math.max(1, Math.floor(rows.length / 12)) },
    },
    yAxis: { type: "category", data: tracks.map((track) => track[1]) },
    visualMap: { min: 0, max: 1, orient: "horizontal", left: "center", bottom: 0 },
    series: [{ type: "heatmap", data: heatmapRows }],
  };
}

export default function SuspiciousSegmentsPage({
  dataset,
  scope,
  manifest,
}: {
  dataset: string;
  scope: string;
  manifest: SiteManifest | null;
}) {
  const stage4 = useJsonData<Record<string, any>>("stage_metrics/stage4.json");
  const timelineOptions = manifest?.timeline_index?.stage4?.[dataset] ?? [];
  const [selectedTimelinePath, setSelectedTimelinePath] = useState<string | null>(timelineOptions[0]?.path ?? null);
  const timeline = useJsonData<Record<string, any>>(selectedTimelinePath);

  useEffect(() => {
    if (timelineOptions.length && !timelineOptions.some((entry) => entry.path === selectedTimelinePath)) {
      setSelectedTimelinePath(timelineOptions[0].path);
    }
  }, [timelineOptions, selectedTimelinePath]);

  if (stage4.loading) return <div className="page-shell">Loading Stage 4 charts…</div>;
  if (stage4.error || !stage4.data) return <div className="page-shell">Unable to load Stage 4 data.</div>;

  const datasetBlock = stage4.data.datasets?.[dataset] ?? {};
  const comparisonRows = (datasetBlock.comparison_rows ?? []).filter((row: Record<string, any>) => row.split === "eval");
  const stratificationRows = (datasetBlock.stratification_rows ?? []).filter((row: Record<string, any>) => row.split === "eval");

  return (
    <div className="page-shell">
      <Callout title="Stage 4 interpretation constraints" tone="warning">
        Proxy labels are repository-specific. Unified suspiciousness is useful for stratification and auditability, not a proven
        better ranking layer than the Stage 3-only baseline.
      </Callout>

      {scope !== "canonical" ? (
        <Callout title="Canonical-first Stage 4 page" tone="info">
          The main Stage 4 views stay canonical. Validation and analysis-only Stage 4 runs appear on the Experiments page.
        </Callout>
      ) : null}

      <section className="content-card">
        <h3>Stage 4 Comparison on Canonical Eval</h3>
        <EChart option={comparisonOption(comparisonRows)} />
      </section>

      <section className="two-column-grid">
        <div className="content-card">
          <h3>Event Metrics</h3>
          <SimpleTable
            columns={[
              { key: "event_type", label: "Event type" },
              { key: "precision", label: "Precision", render: (value) => numberFormat(value) },
              { key: "recall", label: "Recall", render: (value) => numberFormat(value) },
              { key: "f1", label: "F1", render: (value) => numberFormat(value) },
              { key: "valid_event_fraction", label: "Valid event fraction", render: (value) => numberFormat(value) },
            ]}
            rows={(datasetBlock.event_metrics ?? []).filter((row: Record<string, any>) => row.split === "eval")}
          />
        </div>
        <div className="content-card">
          <h3>Stratification View</h3>
          <SimpleTable
            columns={[
              { key: "subgroup", label: "Category" },
              { key: "num_eval_windows", label: "Windows", render: (value) => numberFormat(value, 0) },
              { key: "proxy_abnormal_rate", label: "Proxy abnormal rate", render: (value) => numberFormat(value) },
              { key: "alert_rate", label: "Alert rate", render: (value) => numberFormat(value) },
            ]}
            rows={stratificationRows}
          />
        </div>
      </section>

      <section className="content-card">
        <div className="section-header-inline">
          <h3>Derived Subject Timeline</h3>
          <select value={selectedTimelinePath ?? ""} onChange={(event) => setSelectedTimelinePath(event.target.value)}>
            {timelineOptions.map((entry) => (
              <option key={entry.path} value={entry.path}>
                {entry.split} / {entry.subject_id}
              </option>
            ))}
          </select>
        </div>
        {timeline.loading ? (
          <div className="empty-state">Loading timeline…</div>
        ) : timeline.error || !timeline.data ? (
          <div className="empty-state">Timeline unavailable.</div>
        ) : (
          <EChart option={heatmapOption(timeline.data.rows ?? [])} height={420} />
        )}
      </section>
    </div>
  );
}
