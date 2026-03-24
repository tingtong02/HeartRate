import { Suspense, lazy, useEffect, useState } from "react";
import { Route, Routes } from "react-router-dom";

import Layout from "./components/Layout";
import { fetchJson } from "./lib";
import type { ScopeId, SiteManifest } from "./types";

const OverviewPage = lazy(() => import("./pages/OverviewPage"));
const HrQualityPage = lazy(() => import("./pages/HrQualityPage"));
const SuspiciousSegmentsPage = lazy(() => import("./pages/SuspiciousSegmentsPage"));
const RespirationPage = lazy(() => import("./pages/RespirationPage"));
const ExperimentsPage = lazy(() => import("./pages/ExperimentsPage"));
const ArtifactsPage = lazy(() => import("./pages/ArtifactsPage"));

export default function App() {
  const [manifest, setManifest] = useState<SiteManifest | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dataset, setDataset] = useState("ppg_dalia");
  const [scope, setScope] = useState<ScopeId>("canonical");

  useEffect(() => {
    fetchJson<SiteManifest>("site_manifest.json")
      .then((payload) => {
        setManifest(payload);
        setDataset(payload.site.default_dataset);
        setScope(payload.site.default_scope);
        setLoading(false);
      })
      .catch((err: Error) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div className="app-loading">Loading site manifest…</div>;
  }
  if (error || !manifest) {
    return (
      <div className="app-loading">
        <h1>HeartRate_CNN Results Dashboard</h1>
        <p>Unable to load <code>web/public/data/site_manifest.json</code>.</p>
        <p>Run <code>conda run -n HeartRate_env python scripts/build_results_site_data.py</code> first.</p>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <Layout
      title={manifest.site.title}
      datasets={manifest.datasets}
      scopes={manifest.scopes}
      currentDataset={dataset}
      currentScope={scope}
      onDatasetChange={setDataset}
      onScopeChange={setScope}
    >
      <Suspense fallback={<div className="page-shell">Loading page…</div>}>
        <Routes>
          <Route path="/" element={<OverviewPage dataset={dataset} scope={scope} />} />
          <Route path="/hr-quality" element={<HrQualityPage dataset={dataset} scope={scope} />} />
          <Route
            path="/suspicious-segments"
            element={<SuspiciousSegmentsPage dataset={dataset} scope={scope} manifest={manifest} />}
          />
          <Route
            path="/respiration"
            element={<RespirationPage dataset={dataset} scope={scope} manifest={manifest} />}
          />
          <Route path="/experiments" element={<ExperimentsPage dataset={dataset} />} />
          <Route path="/artifacts" element={<ArtifactsPage manifest={manifest} />} />
        </Routes>
      </Suspense>
    </Layout>
  );
}
