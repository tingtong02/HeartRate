import React from "react";
import Hero from "./components/Hero";
import PipelineStory from "./components/PipelineStory";
import Footer from "./components/Footer";

export default function App() {
  return (
    <div className="portfolio-app">
      <Hero />
      <PipelineStory />
      <Footer />
    </div>
  );
}
