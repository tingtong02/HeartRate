# HeartRate_CNN Results Dashboard

This directory contains a static, English-first technical dashboard for the repository's Stage 0–5 results.

## Workflow

1. Make sure canonical and validation outputs already exist under `outputs/`.
2. Export compact JSON snapshots for the site:

```bash
conda run -n HeartRate_env python scripts/build_results_site_data.py
```

3. Install frontend dependencies and run the site:

```bash
cd web
npm install
npm run dev
```

4. Build a deployable static bundle:

```bash
npm run build
```

The generated dashboard reads from `web/public/data/`. The raw CSVs in `outputs/` are not loaded by the browser.
