import React, { useEffect, useRef } from "react";
import * as echarts from "echarts";

export interface EChartComponentProps {
  option: echarts.EChartsCoreOption;
  style?: React.CSSProperties;
  className?: string;
}

export default function EChartComponent({ option, style, className }: EChartComponentProps) {
  const chartRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let chartInstance: echarts.ECharts | undefined;

    if (chartRef.current) {
      // Initialize ECharts once the element is successfully mounted
      chartInstance = echarts.init(chartRef.current);
      chartInstance.setOption(option);
    }

    const resizeHandler = () => {
      chartInstance?.resize();
    };

    window.addEventListener("resize", resizeHandler);

    return () => {
      window.removeEventListener("resize", resizeHandler);
      // CRITICAL: Dispose of the chart instance to prevent memory leaks on unmount
      if (chartInstance) {
        chartInstance.dispose();
      }
    };
  }, [option]);

  return (
    <div
      ref={chartRef}
      style={{ width: "100%", height: "100%", ...style }}
      className={className}
    />
  );
}
