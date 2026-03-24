import { useEffect, useState } from "react";

export function dataUrl(path: string): string {
  const base = import.meta.env.BASE_URL || "/";
  return `${base.replace(/\/$/, "")}/data/${path}`;
}

export async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(dataUrl(path));
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function useJsonData<T>(path: string | null): {
  data: T | null;
  loading: boolean;
  error: string | null;
} {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!path) {
      setData(null);
      setLoading(false);
      setError(null);
      return () => undefined;
    }
    setLoading(true);
    setError(null);
    fetchJson<T>(path)
      .then((payload) => {
        if (!cancelled) {
          setData(payload);
          setLoading(false);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [path]);

  return { data, loading, error };
}

export function numberFormat(value: unknown, digits = 3): string {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "—";
  }
  return numeric.toFixed(digits);
}

export function percentFormat(value: unknown, digits = 1): string {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "—";
  }
  return `${(numeric * 100).toFixed(digits)}%`;
}

export function labelizeMethod(method: string): string {
  return method
    .replace(/_/g, " ")
    .replace(/\bml\b/gi, "ML")
    .replace(/\bhr\b/gi, "HR")
    .replace(/\bcnn\b/gi, "CNN")
    .replace(/\brr\b/gi, "RR");
}

export function mapRows<T extends Record<string, unknown>>(
  rows: T[] | undefined,
  predicate: (row: T) => boolean,
): T[] {
  return Array.isArray(rows) ? rows.filter(predicate) : [];
}
