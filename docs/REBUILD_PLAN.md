# Master Plan: HeartRate_CNN Portfolio UI/UX Rebuild

## 0. Global Design System & Aesthetic (The "Clean & Fresh" Vibe)
The goal is to rebuild the React + ECharts frontend into a premium, clean, and highly readable portfolio website for resume showcasing. 
* **Vibe:** "Modern Health-Tech" (similar to Apple Health or modern clinical dashboards). Clean, breathable, and minimalist.
* **Color Palette:**
  * Background: Pure White (`#FFFFFF`) or very soft gray (`#F8FAFC`).
  * Text: Dark Slate (`#1E293B`) for headings, lighter gray (`#475569`) for paragraphs.
  * Primary Accent: Soft Teal (`#14B8A6`) or Ocean Blue (`#0EA5E9`) for active elements, buttons, and primary data lines.
  * Secondary Accent: Coral/Soft Red (`#F43F5E`) for anomalies, heart rate highlights, or error metrics.
* **Typography:** Clean geometric sans-serif (e.g., `Inter`, `Roboto`, or system fonts). Plenty of line height and whitespace.
* **UI Elements:** Soft, subtle drop shadows (`box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1)`), rounded corners (`border-radius: 12px`), and removing harsh borders.

---

## Phase 1: Foundation & Hero Section
**Objective:** Establish the clean design system and build an impressive landing section.
1. **Global CSS Overhaul:** Clear out the old dashboard grid layouts in `styles.css`. Implement the new light, fresh color palette and typography using CSS variables.
2. **Hero Component (`Hero.tsx`):**
   * Build a spacious top section.
   * **Title:** "Robust Vital Sign Estimation via Deep Learning"
   * **Subtitle:** "A 5-Stage Conservative Physiological Pipeline for PPG & ACC signals."
   * **Hero Metrics:** Display the most impressive achievement prominently (e.g., "CNN MAE: 2.37 bpm" vs "Baseline: 9.16 bpm") using clean typography and soft accent colors.
   * **Call to Action:** A sleek, ghost button or soft-colored primary button: "Explore the Pipeline".

---

## Phase 2: The 5-Stage Narrative Skeleton
**Objective:** Replace the cluttered data tables with a guided, scrolling narrative layout.
1. **Layout Structure:** Create a main container that uses a split-screen or alternating-row layout.
   * *Left/Top Side:* Text explanation (The "Why" and "What" of the stage).
   * *Right/Bottom Side:* A dedicated, spacious card with subtle shadow reserved for the ECharts visualization.
2. **Component Creation:** Create placeholder sections for:
   * Stage 1 & 2: HR & Beat Extraction
   * Stage 3: Quality Gate
   * Stage 4: Anomaly Detection (Isolation Forests)
   * Stage 5: The Multitask 1D-CNN (Respiration Rate)
3. **No Data Yet:** Focus purely on HTML/CSS structure, responsive Flexbox/Grid logic, and ensuring the "clean and breathable" aesthetic is maintained.

---

## Phase 3: Data Ingestion & ECharts Integration
**Objective:** Bring the webpage to life by fetching the static JSON data and rendering beautiful, minimalist charts.
1. **Data Fetching:** Wire up React `useEffect` hooks to load the pre-computed JSON/CSV files from `public/data/`.
2. **Chart Styling (ECharts):**
   * Configure ECharts to match the clean UI: Remove heavy grid lines, hide unnecessary axis borders, use soft tooltips.
   * Render actual waveforms (e.g., raw PPG vs filtered PPG in Stage 1).
   * Render anomaly scatter plots for Stage 4.
   * Render the CNN prediction curves vs Ground Truth for Stage 5.
3. **Performance:** Ensure ECharts instances are properly disposed of on unmount to prevent memory leaks, and use data downsampling if the JSON arrays are too massive for the browser.

---

## Phase 4: Final Polish, Interactions & Profile Info
**Objective:** Add the final touches that make it a standout resume portfolio.
1. **Micro-interactions:** Add smooth scrolling between sections, gentle hover effects on metric cards, and a fade-in animation when scrolling down.
2. **Personal Branding Footer:** Add a clean footer containing the author's name, major/role, GitHub repository link, and a link to the original paper/dataset if applicable.
3. **Mobile Responsiveness:** Ensure the split-screen layout gracefully stacks into a single column on mobile devices, and the ECharts resize correctly.