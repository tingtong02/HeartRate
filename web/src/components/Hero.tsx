import React from 'react';

export default function Hero() {
  return (
    <section className="hero-section">
      <div className="hero-container">
        <div className="author-branding">
          <span className="author-name">lisihang</span>
          <span className="author-major">Electronic Information Engineering</span>
        </div>
        
        <h1 className="hero-title">Robust Vital Sign Estimation via Deep Learning</h1>
        <p className="hero-subtitle">A 5-Stage Conservative Physiological Pipeline for PPG & ACC signals.</p>
        
        <div className="hero-metric">
          <span className="metric-highlight">CNN MAE: 2.37 bpm</span>
          <span className="metric-compare">vs Baseline: 9.16 bpm</span>
        </div>
        
        <button className="cta-button">Explore the Pipeline</button>
      </div>
    </section>
  );
}
