import SimpleTable from "../components/SimpleTable";
import { numberFormat, useJsonData } from "../lib";
import type { SiteManifest } from "../types";

export default function ArtifactsPage({ manifest }: { manifest: SiteManifest | null }) {
  const artifactInventory = useJsonData<Record<string, any>>("artifacts/artifact_inventory.json");

  if (!manifest) return <div className="page-shell">Loading manifest…</div>;
  if (artifactInventory.loading) return <div className="page-shell">Loading artifacts…</div>;
  if (artifactInventory.error || !artifactInventory.data) return <div className="page-shell">Unable to load artifact inventory.</div>;

  const artifacts = artifactInventory.data.artifacts ?? [];

  return (
    <div className="page-shell">
      <section className="content-card">
        <h3>Output Scope Rules</h3>
        <ul className="clean-list">
          <li>Canonical source-of-record outputs live in <code>{manifest.output_roots.canonical}</code>.</li>
          <li>Bounded validation and analysis-only outputs live in <code>{manifest.output_roots.validation}</code>.</li>
          <li>Reusable cache artifacts live in <code>{manifest.output_roots.cache}</code>.</li>
        </ul>
      </section>

      <section className="three-column-grid">
        <div className="content-card compact-card">
          <h3>Reference Docs</h3>
          <ul className="path-list">
            {manifest.reference_docs.map((item) => (
              <li key={item.path}><code>{item.path}</code></li>
            ))}
          </ul>
        </div>
        <div className="content-card compact-card">
          <h3>Key Scripts</h3>
          <ul className="path-list">
            {manifest.reference_scripts.map((item) => (
              <li key={item.path}><code>{item.path}</code></li>
            ))}
          </ul>
        </div>
        <div className="content-card compact-card">
          <h3>Key Configs</h3>
          <ul className="path-list">
            {manifest.reference_configs.map((item) => (
              <li key={item.path}><code>{item.path}</code></li>
            ))}
          </ul>
        </div>
      </section>

      <section className="content-card">
        <h3>Artifact Inventory</h3>
        <SimpleTable
          columns={[
            { key: "path", label: "Path" },
            { key: "scope", label: "Scope" },
            { key: "dataset", label: "Dataset" },
            { key: "stage", label: "Stage" },
            { key: "artifact_type", label: "Artifact type" },
            { key: "size_bytes", label: "Size (bytes)", render: (value) => numberFormat(value, 0) },
          ]}
          rows={artifacts}
        />
      </section>
    </div>
  );
}
