export type ScopeId = "canonical" | "validation" | "analysis-only";

export interface DatasetOption {
  id: string;
  label: string;
}

export interface ScopeOption {
  id: ScopeId;
  label: string;
}

export interface TimelineIndexEntry {
  subject_id: string;
  split: string;
  path: string;
  num_rows: number;
}

export interface SiteManifest {
  generated_at_utc: string;
  site: {
    title: string;
    default_dataset: string;
    default_scope: ScopeId;
    pages: string[];
  };
  datasets: DatasetOption[];
  scopes: ScopeOption[];
  validation_labels: Array<{ label: string; scope: ScopeId }>;
  timeline_index: Record<string, Record<string, TimelineIndexEntry[]>>;
  artifact_summary?: {
    total_count: number;
    by_scope: Record<string, number>;
  };
  reference_docs: Array<{ label: string; path: string }>;
  reference_scripts: Array<{ label: string; path: string }>;
  reference_configs: Array<{ label: string; path: string }>;
  output_roots: {
    canonical: string;
    validation: string;
    cache: string;
  };
}
