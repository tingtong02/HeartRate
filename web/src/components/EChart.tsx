import { useEffect, useRef } from "react";
import * as echarts from "echarts";

export default function EChart({
  option,
  height = 360,
}: {
  option: echarts.EChartsCoreOption;
  height?: number;
}) {
  const hostRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!hostRef.current) {
      return undefined;
    }
    const chart = echarts.init(hostRef.current, undefined, { renderer: "canvas" });
    chart.setOption(option, true);
    const observer = new ResizeObserver(() => {
      chart.resize();
    });
    observer.observe(hostRef.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [option]);

  return <div className="chart-host" ref={hostRef} style={{ height }} />;
}
