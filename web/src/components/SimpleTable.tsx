import type { ReactNode } from "react";

export default function SimpleTable({
  columns,
  rows,
}: {
  columns: Array<{ key: string; label: string; render?: (value: unknown, row: Record<string, unknown>) => ReactNode }>;
  rows: Record<string, unknown>[];
}) {
  if (!rows.length) {
    return <div className="empty-state">No rows available.</div>;
  }
  return (
    <div className="table-shell">
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={`${rowIndex}-${String(row[columns[0].key] ?? rowIndex)}`}>
              {columns.map((column) => (
                <td key={column.key}>
                  {column.render ? column.render(row[column.key], row) : String(row[column.key] ?? "—")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
