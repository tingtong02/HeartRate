import type { ReactNode } from "react";

export default function Callout({
  title,
  tone = "info",
  children,
}: {
  title: string;
  tone?: "info" | "warning" | "success";
  children: ReactNode;
}) {
  return (
    <div className={`callout callout-${tone}`}>
      <div className="callout-title">{title}</div>
      <div className="callout-body">{children}</div>
    </div>
  );
}
