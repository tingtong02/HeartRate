import React, { useMemo } from "react";
import { useJsonData } from "../lib";
import EChartComponent from "./EChartComponent";
import ScrollReveal from "./ScrollReveal";

interface Stage4Row {
  window_index: number;
  start_time_s: number;
  selected_hr_bpm: number | null;
  ml_signal_quality_score: number;
  irregular_pulse_flag: boolean;
  anomaly_flag: boolean;
  anomaly_score: number;
  irregular_pulse_score: number;
}

interface Stage5Row {
  window_index: number;
  start_time_s: number;
  resp_rate_pred_bpm: number | null;
  resp_rate_ref_bpm: number | null;
  resp_rate_ref_valid_flag: boolean;
}

export default function PipelineStory() {
  const { data: s4Data, loading: s4Loading, error: s4Error } = useJsonData<{ rows: Stage4Row[] }>(
    "stage_timelines/stage4/ppg_dalia/ppg_dalia__eval__S1.json"
  );
  
  const { data: s5Data, loading: s5Loading, error: s5Error } = useJsonData<{ rows: Stage5Row[] }>(
    "stage_timelines/stage5/ppg_dalia/ppg_dalia__eval__S1.json"
  );

  const hrOption = useMemo(() => {
    if (!s4Data) return null;
    const hrData = s4Data.rows
      .filter((r) => r.selected_hr_bpm !== null)
      .map((r) => [r.start_time_s, r.selected_hr_bpm]);

    return {
      tooltip: { trigger: "axis", backgroundColor: "rgba(255, 255, 255, 0.95)", borderRadius: 8 },
      grid: { left: "3%", right: "4%", bottom: "15%", top: "5%", containLabel: true },
      xAxis: { type: "value", splitLine: { show: false } },
      yAxis: { type: "value", name: "HR (bpm)", scale: true, splitLine: { lineStyle: { color: "#E2E8F0", type: "dashed" } } },
      dataZoom: [{ type: "inside" }, { type: "slider", height: 20, bottom: 5 }],
      series: [
        {
          name: "Extracted HR",
          type: "line",
          data: hrData,
          smooth: true,
          symbol: "none",
          lineStyle: { width: 3, color: "#0EA5E9" },
          sampling: "lttb",
        },
      ],
    };
  }, [s4Data]);

  const gateOption = useMemo(() => {
    if (!s4Data) return null;
    const sqiData = s4Data.rows.map((r) => [r.start_time_s, r.ml_signal_quality_score]);

    return {
      tooltip: { trigger: "axis", backgroundColor: "rgba(255, 255, 255, 0.95)", borderRadius: 8 },
      grid: { left: "3%", right: "4%", bottom: "15%", top: "5%", containLabel: true },
      xAxis: { type: "value", splitLine: { show: false } },
      yAxis: { type: "value", name: "Signal Quality", max: 1.0, splitLine: { lineStyle: { color: "#E2E8F0", type: "dashed" } } },
      dataZoom: [{ type: "inside" }, { type: "slider", height: 20, bottom: 5 }],
      series: [
        {
          name: "Quality Gate Score",
          type: "line",
          data: sqiData,
          smooth: true,
          symbol: "none",
          areaStyle: { color: "rgba(20, 184, 166, 0.15)" },
          lineStyle: { width: 2, color: "#14B8A6" },
          sampling: "lttb",
        },
      ],
    };
  }, [s4Data]);

  const anomalyOption = useMemo(() => {
    if (!s4Data) return null;
    const normal: number[][] = [];
    const anom: number[][] = [];
    s4Data.rows.forEach((r) => {
      const ts = r.start_time_s;
      const score = Math.max(r.anomaly_score || 0, r.irregular_pulse_score || 0);
      if (r.anomaly_flag || r.irregular_pulse_flag) {
        anom.push([ts, score]);
      } else {
        normal.push([ts, score]);
      }
    });

    return {
      tooltip: { trigger: "item" },
      grid: { left: "3%", right: "4%", bottom: "15%", top: "5%", containLabel: true },
      xAxis: { type: "value", splitLine: { show: false } },
      yAxis: { type: "value", name: "Anomaly Score", splitLine: { lineStyle: { color: "#E2E8F0", type: "dashed" } } },
      dataZoom: [{ type: "inside" }, { type: "slider", height: 20, bottom: 5 }],
      series: [
        { name: "Normal", type: "scatter", data: normal, symbolSize: 4, itemStyle: { color: "#CBD5E1" } },
        { name: "Anomaly/Irregular", type: "scatter", data: anom, symbolSize: 9, itemStyle: { color: "#F43F5E", shadowBlur: 10, shadowColor: "#F43F5E" } },
      ],
    };
  }, [s4Data]);

  const respOption = useMemo(() => {
    if (!s5Data) return null;
    const predData = s5Data.rows
      .filter((r) => r.resp_rate_pred_bpm !== null)
      .map((r) => [r.start_time_s, r.resp_rate_pred_bpm]);
    const refData = s5Data.rows
      .filter((r) => r.resp_rate_ref_valid_flag)
      .map((r) => [r.start_time_s, r.resp_rate_ref_bpm]);

    return {
      tooltip: { trigger: "axis", backgroundColor: "rgba(255, 255, 255, 0.95)" },
      legend: { top: 0, icon: "circle", textStyle: { fontFamily: "Inter", color: "#475569" } },
      grid: { left: "3%", right: "4%", bottom: "15%", top: "15%", containLabel: true },
      xAxis: { type: "value", splitLine: { show: false } },
      yAxis: { type: "value", name: "RR (bpm)", scale: true, splitLine: { lineStyle: { color: "#E2E8F0", type: "dashed" } } },
      dataZoom: [{ type: "inside" }, { type: "slider", height: 20, bottom: 5 }],
      series: [
        { name: "CNN Prediction", type: "line", data: predData, smooth: true, symbol: "none", lineStyle: { width: 3, color: "#14B8A6" } },
        { name: "Ground Truth", type: "line", data: refData, smooth: true, symbol: "none", lineStyle: { width: 2, color: "#94A3B8", type: "dashed" } },
      ],
    };
  }, [s5Data]);

  const renderPlaceholderContent = (loading: boolean, error: string | null, chartOption: any) => {
    if (loading) {
      return (
        <div className="chart-loader">
          <div className="loader-spinner" />
          <span>Fetching sequence data...</span>
        </div>
      );
    }
    if (error) {
      return <div className="chart-error">Failed to load payload: {error}</div>;
    }
    if (chartOption) {
      return <EChartComponent option={chartOption} />;
    }
    return <span>No data available.</span>;
  };

  return (
    <section className="pipeline-story">
      <div className="story-container">
        
        {/* Block 1 */}
        <ScrollReveal className="story-row">
          <div className="story-text">
            <div className="stage-badge">Stages 1 & 2</div>
            <h2>Signal Processing & Beat Extraction</h2>
            <p>
              To maintain an ultra-fast web portfolio, multi-megabyte 64Hz raw waveforms are condensed via analytical extraction. Below is the continuous <strong>Extracted Heart Rate (bpm)</strong> sequence cleanly derived from the raw signals by the Stage 1 & 2 Bandpass and Dynamic Peak detection algorithms.
            </p>
            <ul className="feature-list">
              <li>Bandpass frequency filtering</li>
              <li>Dynamic threshold peak detection</li>
              <li>Inter-beat interval (IBI) extraction</li>
            </ul>
          </div>
          <div className="story-visual">
            <div className="chart-placeholder">
              {renderPlaceholderContent(s4Loading, s4Error, hrOption)}
            </div>
          </div>
        </ScrollReveal>

        {/* Block 2 */}
        <ScrollReveal className="story-row alternate">
          <div className="story-text">
            <div className="stage-badge">Stage 3</div>
            <h2>The Quality Gate & Signal Filtering</h2>
            <p>
              Motion artifacts degrade PPG accuracy. The plotted <strong>Machine Learning Quality Gate Score</strong> dictates whether a given time window is clean enough to trust. If the score drops below the acceptance threshold, the pipeline activates a conservative hold-and-fallback policy.
            </p>
            <ul className="feature-list">
              <li>Machine Learning (LogReg) signal gating</li>
              <li>ECG-backed pseudo-target evaluation</li>
              <li>Robust HR hold-and-fallback policy</li>
            </ul>
          </div>
          <div className="story-visual">
            <div className="chart-placeholder">
              {renderPlaceholderContent(s4Loading, s4Error, gateOption)}
            </div>
          </div>
        </ScrollReveal>

        {/* Block 3 */}
        <ScrollReveal className="story-row">
          <div className="story-text">
            <div className="stage-badge">Stage 4</div>
            <h2>Anomaly Detection via Isolation Forests</h2>
            <p>
              Applying unsupervised learning (Isolation Forests), the pipeline flags severely corrupted or physiologically impossible anomalies (highlighted in <strong>Coral Red</strong>). The anomaly score protects downstream tasks from extreme false readings.
            </p>
            <ul className="feature-list">
              <li>Irregular pulse screening (HistGBDT)</li>
              <li>Unsupervised anomaly scoring (Isolation Forest)</li>
              <li>Unified suspiciousness routing</li>
            </ul>
          </div>
          <div className="story-visual">
            <div className="chart-placeholder">
              {renderPlaceholderContent(s4Loading, s4Error, anomalyOption)}
            </div>
          </div>
        </ScrollReveal>

        {/* Block 4 */}
        <ScrollReveal className="story-row alternate prominent-row">
          <div className="story-text">
            <div className="stage-badge crown-jewel">Stage 5: The Crown Jewel</div>
            <h2>Multitask 1D-CNN for Respiration Rate</h2>
            <p>
              The CNN processes the multi-modal time series to concurrently estimate Respiration Rate. Observe the tight alignment between the CNN's Prediction (Teal) and the Ground Truth (Dashed Gray), representing a drastic MAE reduction over classical surrogate methods.
            </p>
            <ul className="feature-list">
              <li>Multitask 1D-CNN architecture</li>
              <li>Smooth L1 Loss (Regression) & BCE (Validity)</li>
              <li>Significant MAE reduction vs baseline (e.g., 2.37 bpm vs ~9.16 bpm)</li>
            </ul>
          </div>
          <div className="story-visual">
            <div className="chart-placeholder large-placeholder">
              {renderPlaceholderContent(s5Loading, s5Error, respOption)}
            </div>
          </div>
        </ScrollReveal>

      </div>
    </section>
  );
}
