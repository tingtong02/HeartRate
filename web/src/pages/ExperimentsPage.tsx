import Callout from "../components/Callout";
import SimpleTable from "../components/SimpleTable";
import { numberFormat, useJsonData } from "../lib";

export default function ExperimentsPage({ dataset }: { dataset: string }) {
  const experiments = useJsonData<Record<string, any>>("experiments/experiments.json");

  if (experiments.loading) return <div className="page-shell">Loading experiments…</div>;
  if (experiments.error || !experiments.data) return <div className="page-shell">Unable to load experiment data.</div>;

  const labels = experiments.data.labels ?? [];
  const tuning = experiments.data.stage5_tuning?.[dataset] ?? {};

  return (
    <div className="page-shell">
      <Callout title="Non-canonical experiment scopes" tone="warning">
        Everything on this page is validation-only or analysis-only unless explicitly marked otherwise. These results do not drive
        the headline canonical conclusions on the Overview page.
      </Callout>

      {labels.map((labelBlock: Record<string, any>) => {
        const datasetBlock = labelBlock.datasets?.[dataset];
        if (!datasetBlock) return null;
        return (
          <section className="content-card" key={`${labelBlock.label}-${dataset}`}>
            <div className="section-header-inline">
              <h3>{labelBlock.label}</h3>
              <span className={`scope-badge scope-${labelBlock.scope}`}>{labelBlock.scope}</span>
            </div>
            <SimpleTable
              columns={[
                { key: "method", label: "Method" },
                { key: "auprc", label: "AUPRC", render: (value) => numberFormat(value) },
                { key: "auroc", label: "AUROC", render: (value) => numberFormat(value) },
                { key: "precision", label: "Precision", render: (value) => numberFormat(value) },
                { key: "recall", label: "Recall", render: (value) => numberFormat(value) },
                { key: "alert_rate", label: "Alert rate", render: (value) => numberFormat(value) },
              ]}
              rows={datasetBlock.comparison_rows ?? []}
            />
          </section>
        );
      })}

      <section className="content-card">
        <h3>Stage 5 Tuning Summary</h3>
        <SimpleTable
          columns={[
            { key: "phase", label: "Phase" },
            { key: "window_seconds", label: "Window (s)" },
            { key: "channel_set", label: "Channels" },
            { key: "base_width", label: "Width" },
            { key: "learning_rate", label: "LR", render: (value) => numberFormat(value, 4) },
            { key: "high_quality_resp_mae_bpm", label: "HQ MAE", render: (value) => numberFormat(value) },
            { key: "predicted_valid_coverage", label: "Pred valid coverage", render: (value) => numberFormat(value) },
          ]}
          rows={tuning.top_rows ?? []}
        />
      </section>
    </div>
  );
}
