import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";

import type { DatasetOption, ScopeId, ScopeOption } from "../types";

const navItems = [
  { to: "/", label: "Overview" },
  { to: "/hr-quality", label: "HR & Quality" },
  { to: "/suspicious-segments", label: "Suspicious Segments" },
  { to: "/respiration", label: "Respiration & Multitask" },
  { to: "/experiments", label: "Experiments" },
  { to: "/artifacts", label: "Artifacts & Reproducibility" },
];

export default function Layout({
  title,
  datasets,
  scopes,
  currentDataset,
  currentScope,
  onDatasetChange,
  onScopeChange,
  children,
}: {
  title: string;
  datasets: DatasetOption[];
  scopes: ScopeOption[];
  currentDataset: string;
  currentScope: ScopeId;
  onDatasetChange: (nextValue: string) => void;
  onScopeChange: (nextValue: ScopeId) => void;
  children: ReactNode;
}) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-eyebrow">Technical Results Dashboard</div>
          <h1>{title}</h1>
          <p>CPU-first public-dataset PPG physiological analysis framework.</p>
        </div>
        <nav className="side-nav">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `side-nav-link${isActive ? " active" : ""}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="control-group">
          <label htmlFor="dataset-select">Dataset</label>
          <select
            id="dataset-select"
            value={currentDataset}
            onChange={(event) => onDatasetChange(event.target.value)}
          >
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.label}
              </option>
            ))}
          </select>
        </div>
        <div className="control-group">
          <label htmlFor="scope-select">Result Scope</label>
          <select
            id="scope-select"
            value={currentScope}
            onChange={(event) => onScopeChange(event.target.value as ScopeId)}
          >
            {scopes.map((scope) => (
              <option key={scope.id} value={scope.id}>
                {scope.label}
              </option>
            ))}
          </select>
        </div>
      </aside>
      <main className="main-content">{children}</main>
    </div>
  );
}
