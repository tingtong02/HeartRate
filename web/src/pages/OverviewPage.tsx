import Callout from "../components/Callout";
import StatCard from "../components/StatCard";
import { numberFormat, percentFormat, useJsonData } from "../lib";

export default function OverviewPage({ dataset, scope }: { dataset: string; scope: string }) {
  const { data, loading, error } = useJsonData<Record<string, any>>("overview_summary.json");

  if (loading) return <div className="page-shell">Loading overview…</div>;
  if (error || !data) return <div className="page-shell">Unable to load overview data: {error}</div>;

  const datasetSummary = data.datasets?.[dataset] ?? {};
  const stage4Conclusion = datasetSummary.stage4_conclusion ?? {};
  const stage5Conclusion = datasetSummary.stage5_conclusion ?? {};

  return (
    <div className="page-shell">
      <section className="hero-card">
        <div className="hero-text">
          <div className="eyebrow">Stage 0–5 Results Site</div>
          <h2>{data.project_title}</h2>
          <p>{data.summary}</p>
        </div>
        <div className="hero-stats">
          <StatCard
            label="Stage 1 Best"
            value={datasetSummary.stage1_best?.method ?? "—"}
            detail={`MAE ${numberFormat(datasetSummary.stage1_best?.mae)}`}
          />
          <StatCard
            label="Stage 4 Standalone"
            value={stage4Conclusion.strongest_stage4_standalone_label ?? "—"}
            detail={`Dataset: ${datasetSummary.label ?? dataset}`}
          />
          <StatCard
            label="Stage 5 MAE Reduction"
            value={percentFormat((stage5Conclusion.mae_reduction_pct ?? 0) / 100, 1)}
            detail="high-quality eval vs surrogate baseline"
          />
        </div>
      </section>

      {scope !== "canonical" ? (
        <Callout title="Canonical-first overview" tone="warning">
          Overview cards stay anchored to canonical source-of-record outputs. Use the Experiments page to inspect
          validation and analysis-only runs.
        </Callout>
      ) : null}

      <section className="section-grid stage-grid">
        {data.stage_cards?.map((stage: Record<string, unknown>) => (
          <div className="stage-card" key={String(stage.stage)}>
            <div className="stage-label">{String(stage.stage)}</div>
            <div className="stage-default">{String(stage.default_path)}</div>
            <p>{String(stage.focus)}</p>
          </div>
        ))}
      </section>

      <section className="content-card">
        <h3>Best-Supported Evidence</h3>
        <div className="banner-list">
          {data.evidence_banners?.map((item: string) => (
            <div className="banner-item" key={item}>
              {item}
            </div>
          ))}
        </div>
      </section>

      <section className="content-card">
        <h3>Current Default Paths</h3>
        <div className="chip-grid">
          {data.default_paths?.map((item: Record<string, unknown>) => (
            <div className="chip-card" key={`${String(item.stage)}-${String(item.name)}`}>
              <span>{String(item.stage)}</span>
              <strong>{String(item.name)}</strong>
            </div>
          ))}
        </div>
      </section>

      <section className="content-card">
        <h3>{datasetSummary.label} Snapshot</h3>
        <div className="metric-pair-grid">
          <StatCard
            label="Stage 3 Default"
            value={datasetSummary.stage3_default?.[0]?.method ?? "gated_stage3_ml_logreg"}
            detail={`MAE ${numberFormat(datasetSummary.stage3_default?.[0]?.mae)}`}
          />
          <StatCard
            label="Stage 3 Retention"
            value={percentFormat(datasetSummary.stage3_default?.[0]?.retention_ratio)}
            detail="canonical eval"
          />
          <StatCard
            label="Stage 4 Unified vs Stage 3"
            value={stage4Conclusion.unified_beats_stage3 ? "Beats baseline" : "Does not beat baseline"}
            detail={`Unified AUPRC ${numberFormat(stage4Conclusion.stage4_full_default?.auprc)}`}
          />
          <StatCard
            label="Stage 5 CNN"
            value={stage5Conclusion.cnn_beats_baseline ? "Beats baseline" : "Mixed"}
            detail={`HQ MAE ${numberFormat(stage5Conclusion.cnn?.resp_mae_bpm)} bpm`}
          />
        </div>
      </section>

      <section className="content-card">
        <h3>Current Conclusions</h3>
        <ul className="clean-list">
          {data.best_supported_conclusions?.map((item: string) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
