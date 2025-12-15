import React from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface HistoryEntry {
  epoch: number;
  loss: number;
  accuracy: number;
  timestamp: string;
  samples: number;
}

interface TrainingChartProps {
  history: HistoryEntry[];
}

type ChartDataPoint = {
  epoch: number;
  loss: number;
  accuracy: number;
  samples: number;
  timestamp: string;
};

interface RechartsCustomTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: ChartDataPoint }> | undefined;
  label?: string | number;
}

const CustomTooltip: React.FC<RechartsCustomTooltipProps> = ({
  active,
  payload,
  label,
}) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div
        style={{
          background: "rgba(255, 255, 255, 0.95)",
          border: "1px solid #ccc",
          borderRadius: "8px",
          padding: "10px",
          boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
        }}
      >
        <p style={{ margin: "0", fontWeight: "bold" }}>Эпоха {label}</p>
        <p style={{ margin: "5px 0", color: "#ef4444" }}>Loss: {data.loss}</p>
        <p style={{ margin: "5px 0", color: "#10b981" }}>
          Точность: {data.accuracy}%
        </p>
        <p style={{ margin: "5px 0", color: "#666", fontSize: "0.8em" }}>
          Образцов: {data.samples}
        </p>
        <p style={{ margin: "5px 0", color: "#666", fontSize: "0.8em" }}>
          Время: {data.timestamp}
        </p>
      </div>
    );
  }
  return null;
};

const TrainingChart: React.FC<TrainingChartProps> = ({ history }) => {
  if (!history || history.length === 0) {
    return (
      <div className="dashboard-card">
        <h3 className="card-title">📈 Графики обучения</h3>
        <div className="loading">Нет данных для отображения</div>
      </div>
    );
  }

  // Подготавливаем данные для графика
  const chartData = history.map((entry) => ({
    epoch: entry.epoch,
    loss: Number(entry.loss.toFixed(4)),
    accuracy: Number((entry.accuracy * 100).toFixed(2)), // Конвертируем в проценты
    samples: entry.samples,
    timestamp: new Date(entry.timestamp).toLocaleTimeString(),
  }));

  return (
    <div>
      <h3 className="card-title">📈 Графики обучения в реальном времени</h3>

      {/* Summary Stats */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
          gap: "15px",
          marginBottom: "20px",
          padding: "15px",
          background: "#f8fafc",
          borderRadius: "8px",
        }}
      >
        <div style={{ textAlign: "center" }}>
          <div
            style={{ fontSize: "1.5em", fontWeight: "bold", color: "#667eea" }}
          >
            {history.length}
          </div>
          <div style={{ fontSize: "0.8em", color: "#666" }}>Всего эпох</div>
        </div>

        <div style={{ textAlign: "center" }}>
          <div
            style={{ fontSize: "1.5em", fontWeight: "bold", color: "#10b981" }}
          >
            {Math.max(...chartData.map((d) => d.accuracy)).toFixed(1)}%
          </div>
          <div style={{ fontSize: "0.8em", color: "#666" }}>
            Лучшая точность
          </div>
        </div>

        <div style={{ textAlign: "center" }}>
          <div
            style={{ fontSize: "1.5em", fontWeight: "bold", color: "#ef4444" }}
          >
            {Math.min(...chartData.map((d) => d.loss)).toFixed(4)}
          </div>
          <div style={{ fontSize: "0.8em", color: "#666" }}>Лучший Loss</div>
        </div>

        <div style={{ textAlign: "center" }}>
          <div
            style={{ fontSize: "1.5em", fontWeight: "bold", color: "#f59e0b" }}
          >
            {history[history.length - 1]?.samples || 0}
          </div>
          <div style={{ fontSize: "0.8em", color: "#666" }}>Образцов/эпоха</div>
        </div>
      </div>

      {/* Chart */}
      <div className="chart-container">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{
              top: 20,
              right: 30,
              left: 20,
              bottom: 20,
            }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="epoch"
              stroke="#6b7280"
              tick={{ fontSize: 12 }}
              label={{ value: "Эпоха", position: "insideBottom", offset: -10 }}
            />
            <YAxis
              yAxisId="left"
              stroke="#ef4444"
              tick={{ fontSize: 12 }}
              label={{ value: "Loss", angle: -90, position: "insideLeft" }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#10b981"
              tick={{ fontSize: 12 }}
              label={{
                value: "Точность (%)",
                angle: 90,
                position: "insideRight",
              }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />

            <Line
              yAxisId="left"
              type="monotone"
              dataKey="loss"
              stroke="#ef4444"
              strokeWidth={3}
              dot={{ fill: "#ef4444", strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, fill: "#dc2626" }}
              name="Loss"
            />

            <Line
              yAxisId="right"
              type="monotone"
              dataKey="accuracy"
              stroke="#10b981"
              strokeWidth={3}
              dot={{ fill: "#10b981", strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, fill: "#059669" }}
              name="Точность (%)"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Recent Epochs Table */}
      <div style={{ marginTop: "20px" }}>
        <h4 style={{ marginBottom: "10px" }}>📊 Последние эпохи</h4>
        <div style={{ overflowX: "auto" }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Эпоха</th>
                <th>Loss</th>
                <th>Точность</th>
                <th>Образцов</th>
                <th>Время</th>
              </tr>
            </thead>
            <tbody>
              {history
                .slice(-5)
                .reverse()
                .map((entry, index) => (
                  <tr key={index}>
                    <td style={{ fontWeight: "bold" }}>#{entry.epoch}</td>
                    <td style={{ color: "#ef4444" }}>
                      {entry.loss.toFixed(4)}
                    </td>
                    <td style={{ color: "#10b981" }}>
                      {(entry.accuracy * 100).toFixed(2)}%
                    </td>
                    <td>{entry.samples}</td>
                    <td style={{ fontSize: "0.8em" }}>
                      {new Date(entry.timestamp).toLocaleTimeString()}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default TrainingChart;
