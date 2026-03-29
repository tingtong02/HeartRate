import React from "react";

export default function PipelineStory() {
  return (
    <section className="pipeline-story">
      <div className="story-container">
        
        {/* Block 1: Stages 1 & 2 */}
        <div className="story-row">
          <div className="story-text">
            <div className="stage-badge">Stages 1 & 2</div>
            <h2>Signal Processing & Beat Extraction</h2>
            <p>
              The foundation of the pipeline begins by loading raw photoplethysmography (PPG) and accelerometer (ACC) signals. We apply robust bandpass filtering and dynamic peak detection to extract inter-beat intervals (IBI) and establish a baseline Heart Rate (HR).
            </p>
            <ul className="feature-list">
              <li>Bandpass frequency filtering</li>
              <li>Dynamic threshold peak detection</li>
              <li>Inter-beat interval (IBI) extraction</li>
            </ul>
          </div>
          <div className="story-visual">
            <div className="chart-placeholder">
              <span>[Placeholder: ECharts Multi-line Waveform of Raw vs Filtered PPG]</span>
            </div>
          </div>
        </div>

        {/* Block 2: Stage 3 (Alternating) */}
        <div className="story-row alternate">
          <div className="story-text">
            <div className="stage-badge">Stage 3</div>
            <h2>The Quality Gate & Signal Filtering</h2>
            <p>
              Motion artifacts and signal noise can severely degrade HR estimation. We introduce an ML-backed Quality Gate (Logistic Regression) that evaluates signal integrity, rejecting unreliable segments and implementing a conservative hold policy for sustained accuracy.
            </p>
            <ul className="feature-list">
              <li>Machine Learning (LogReg) signal gating</li>
              <li>ECG-backed pseudo-target evaluation</li>
              <li>Robust HR hold-and-fallback policy</li>
            </ul>
          </div>
          <div className="story-visual">
            <div className="chart-placeholder">
              <span>[Placeholder: ECharts Scatter/Line showing HR vs Accepted/Rejected segments]</span>
            </div>
          </div>
        </div>

        {/* Block 3: Stage 4 */}
        <div className="story-row">
          <div className="story-text">
            <div className="stage-badge">Stage 4</div>
            <h2>Anomaly Detection via Isolation Forests</h2>
            <p>
              Beyond basic heart rate monitoring, Stage 4 applies advanced unsupervised learning to detect physiological anomalies. Using an Isolation Forest model alongside HistGradientBoosting for irregular pulse screening, the pipeline identifies highly suspicious data windows.
            </p>
            <ul className="feature-list">
              <li>Irregular pulse screening (HistGBDT)</li>
              <li>Unsupervised anomaly scoring (Isolation Forest)</li>
              <li>Unified suspiciousness routing</li>
            </ul>
          </div>
          <div className="story-visual">
            <div className="chart-placeholder">
              <span>[Placeholder: ECharts Scatter Plot of Anomalies / Irregularities]</span>
            </div>
          </div>
        </div>

        {/* Block 4: Stage 5 (Alternating & Prominent) */}
        <div className="story-row alternate prominent-row">
          <div className="story-text">
            <div className="stage-badge crown-jewel">Stage 5: The Crown Jewel</div>
            <h2>Multitask 1D-CNN for Respiration Rate</h2>
            <p>
              The pinnacle of the pipeline is a deep learning model designed to concurrently estimate Respiration Rate and predict signal validity. A custom 1D Convolutional Neural Network fuses time-series data (PPG, ACC, respiratory-induced variations) with scalar features to drastically outperform classical baselines.
            </p>
            <ul className="feature-list">
              <li>Multitask 1D-CNN architecture</li>
              <li>Smooth L1 Loss (Regression) & BCE (Validity)</li>
              <li>Significant MAE reduction vs baseline (e.g., 2.37 bpm vs 9.16 bpm)</li>
            </ul>
          </div>
          <div className="story-visual">
            <div className="chart-placeholder large-placeholder">
              <span>[Placeholder: ECharts Comparison of CNN Predict vs Ground Truth RR]</span>
            </div>
          </div>
        </div>

      </div>
    </section>
  );
}
