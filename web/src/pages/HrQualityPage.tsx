import EChart from "../components/EChart";
import Callout from "../components/Callout";
import SimpleTable from "../components/SimpleTable";
import { labelizeMethod, numberFormat, useJsonData } from "../lib";

function stage1Option(rows: Record<string, any>[]) {
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0 },
    grid: { left: 48, right: 20, top: 42, bottom: 56 },
    xAxis: { type: "category", data: rows.map((row) => labelizeMethod(String(row.method ?? ""))), axisLabel: { rotate: 20 } },
    yAxis: [
      { type: "value", name: "MAE / RMSE" },
      { type: "value", name: "Pearson r", min: 0, max: 1 },
    ],
    series: [
      { name: "MAE", type: "bar", data: rows.map((row) => row.mae) },
      { name: "RMSE", type: "bar", data: rows.map((row) => row.rmse) },
      { name: "Pearson r", type: "line", yAxisIndex: 1, data: rows.map((row) => row.pearson_r) },
    ],
  };
}

function thresholdOption(rows: Record<string, any>[]) {
  return {
    tooltip: { trigger: "axis" },
    legend: { top: 0 },
    grid: { left: 52, right: 18, top: 42, bottom: 42 },
    xAxis: { type: "value", name: "Threshold" },
    yAxis: [
      { type: "value", name: "Retention" },
      { type: "value", name: "MAE" },
    ],
    series: [
      {
        name: "Retention",
        type: "line",
        smooth: true,
        data: rows.map((row) => [row.threshold, row.retention_ratio]),
      },
      {
        name: "MAE",
        type: "line",
        yAxisIndex: 1,
        smooth: true,
        data: rows.map((row) => [row.threshold, row.mae]),
      },
    ],
  };
}

function policyOption(rows: Record<string, any>[]) {
  return {
    tooltip: { trigger: "item" },
    grid: { left: 52, right: 18, top: 28, bottom: 42 },
    xAxis: { type: "value", name: "Output fraction" },
    yAxis: { type: "value", name: "MAE" },
    series: [
      {
        type: "scatter",
        symbolSize: 14,
        data: rows.map((row) => [row.output_fraction, row.mae, row.profile_name]),
      },
    ],
  };
}

export default function HrQualityPage({ dataset, scope }: { dataset: string; scope: string }) {
  const stage1 = useJsonData<Record<string, any>>("stage_metrics/stage1.json");
  const stage2 = useJsonData<Record<string, any>>("stage_metrics/stage2.json");
  const stage3 = useJsonData<Record<string, any>>("stage_metrics/stage3.json");

  if (stage1.loading || stage2.loading || stage3.loading) return <div className="page-shell">Loading HR and quality views…</div>;
  if (stage1.error || stage2.error || stage3.error) {
    return <div className="page-shell">Unable to load HR and quality data.</div>;
  }

  const stage1Rows = stage1.data?.datasets?.[dataset]?.rows ?? [];
  const stage2BeatRows = stage2.data?.datasets?.[dataset]?.beat_summary ?? [];
  const stage2FeatureRows = stage2.data?.datasets?.[dataset]?.feature_summary ?? [];
  const stage3EnhancedRows = stage3.data?.datasets?.[dataset]?.enhanced_metrics ?? [];
  const thresholdRows = (stage3.data?.datasets?.[dataset]?.threshold_sweep ?? []).filter((row: Record<string, any>) => row.split === "train_select");
  const policyRows = (stage3.data?.datasets?.[dataset]?.policy_sweep ?? []).filter((row: Record<string, any>) => row.split === "eval");

  return (
    <div className="page-shell">
      {scope !== "canonical" ? (
        <Callout title="Canonical-only charts" tone="warning">
          This page stays on canonical outputs so Stage 1–3 comparisons remain source-of-record.
        </Callout>
      ) : null}

      <section className="content-card">
        <h3>Stage 1 Method Comparison</h3>
        <EChart option={stage1Option(stage1Rows)} />
      </section>

      <section className="two-column-grid">
        <div className="content-card">
          <h3>Stage 2 Beat / IBI Summary</h3>
          <SimpleTable
            columns={[
              { key: "variant", label: "Variant" },
              { key: "task", label: "Task" },
              { key: "precision", label: "Precision", render: (value) => numberFormat(value) },
              { key: "recall", label: "Recall", render: (value) => numberFormat(value) },
              { key: "f1", label: "F1", render: (value) => numberFormat(value) },
              { key: "ibi_mae_ms", label: "IBI MAE (ms)", render: (value) => numberFormat(value) },
            ]}
            rows={stage2BeatRows}
          />
        </div>
        <div className="content-card">
          <h3>Stage 2 Feature Accuracy</h3>
          <SimpleTable
            columns={[
              { key: "variant", label: "Variant" },
              { key: "feature", label: "Feature" },
              { key: "mae", label: "MAE", render: (value) => numberFormat(value) },
              { key: "pearson_r", label: "Pearson r", render: (value) => numberFormat(value) },
            ]}
            rows={stage2FeatureRows}
          />
        </div>
      </section>

      <section className="content-card">
        <h3>Stage 3 HR Comparison</h3>
        <SimpleTable
          columns={[
            { key: "method", label: "Method", render: (value) => labelizeMethod(String(value ?? "")) },
            { key: "mae", label: "MAE", render: (value) => numberFormat(value) },
            { key: "rmse", label: "RMSE", render: (value) => numberFormat(value) },
            { key: "pearson_r", label: "Pearson r", render: (value) => numberFormat(value) },
            { key: "retention_ratio", label: "Retention", render: (value) => numberFormat(value) },
          ]}
          rows={stage3EnhancedRows.filter((row: Record<string, any>) => row.task === "hr_comparison")}
        />
      </section>

      <section className="two-column-grid">
        <div className="content-card">
          <h3>Stage 3 Threshold Sweep</h3>
          <EChart option={thresholdOption(thresholdRows)} />
        </div>
        <div className="content-card">
          <h3>Robust Policy Sweep</h3>
          <EChart option={policyOption(policyRows)} />
        </div>
      </section>
    </div>
  );
}
