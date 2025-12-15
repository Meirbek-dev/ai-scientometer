import React from "react";
import {
  TrendingUp,
  TrendingDown,
  Minus,
  Target,
  Activity,
} from "lucide-react";

interface TrainingStatus {
  is_training: boolean;
  current_stats: {
    epoch: number;
    loss: number;
    accuracy: number;
    learning_rate: number;
    samples_processed: number;
    start_time: string;
    last_update: string;
    improvements: any[];
  };
  recent_history: any[];
  total_epochs: number;
  improvements_count: number;
  data_samples: number;
  training_duration_seconds: number;
  training_duration_formatted: string;
  progress_indicators: {
    loss_trend: string;
    accuracy_trend: string;
    is_improving: boolean;
  };
}

interface TrainingMetricsProps {
  trainingStatus: TrainingStatus | null;
}

const TrainingMetrics: React.FC<TrainingMetricsProps> = ({
  trainingStatus,
}) => {
  if (!trainingStatus) {
    return (
      <div className="space-y-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
            <Target className="w-5 h-5 text-white" />
          </div>
          <h3 className="text-xl font-semibold text-white">Метрики обучения</h3>
        </div>
        <div className="flex flex-col items-center justify-center py-8 space-y-4">
          <Activity className="w-8 h-8 text-slate-400 animate-pulse" />
          <p className="text-slate-400">Загрузка метрик...</p>
        </div>
      </div>
    );
  }

  const {
    current_stats,
    progress_indicators,
    training_duration_formatted,
    data_samples,
  } = trainingStatus;

  const getTrendIcon = (trend: string, isLoss: boolean = false) => {
    switch (trend) {
      case "increasing":
        return isLoss ? (
          <TrendingUp className="w-4 h-4 text-red-400" />
        ) : (
          <TrendingUp className="w-4 h-4 text-green-400" />
        );
      case "decreasing":
        return isLoss ? (
          <TrendingDown className="w-4 h-4 text-green-400" />
        ) : (
          <TrendingDown className="w-4 h-4 text-red-400" />
        );
      default:
        return <Minus className="w-4 h-4 text-slate-400" />;
    }
  };

  const getTrendClass = (trend: string, isLoss: boolean = false) => {
    switch (trend) {
      case "increasing":
        return isLoss ? "trend-down" : "trend-up"; // Для loss рост - плохо
      case "decreasing":
        return isLoss ? "trend-up" : "trend-down"; // Для loss снижение - хорошо
      default:
        return "trend-stable";
    }
  };

  const getTrendText = (trend: string, isLoss: boolean = false) => {
    if (isLoss) {
      switch (trend) {
        case "increasing":
          return "Растет (плохо)";
        case "decreasing":
          return "Снижается (хорошо!)";
        default:
          return "Стабильно";
      }
    } else {
      switch (trend) {
        case "increasing":
          return "Растет (хорошо!)";
        case "decreasing":
          return "Снижается (плохо)";
        default:
          return "Стабильно";
      }
    }
  };

  const getAccuracyColor = (accuracy: number) => {
    if (accuracy >= 0.9) return "#10b981"; // Отлично
    if (accuracy >= 0.8) return "#f59e0b"; // Хорошо
    if (accuracy >= 0.7) return "#ef4444"; // Удовлетворительно
    return "#6b7280"; // Плохо
  };

  const getLossColor = (loss: number) => {
    if (loss <= 0.3) return "#10b981"; // Отлично
    if (loss <= 0.7) return "#f59e0b"; // Хорошо
    if (loss <= 1.0) return "#ef4444"; // Удовлетворительно
    return "#6b7280"; // Плохо
  };

  return (
    <div className="dashboard-card">
      <h3 className="card-title">📊 Метрики обучения</h3>

      <div className="metrics-grid">
        {/* Current Epoch */}
        <div className="metric-item fade-in">
          <div className="metric-value" style={{ color: "#667eea" }}>
            {current_stats.epoch}
          </div>
          <div className="metric-label">Эпоха</div>
          {trainingStatus.is_training && (
            <div className="metric-trend trend-up">⏫ Активно</div>
          )}
        </div>

        {/* Loss */}
        <div className="metric-item fade-in">
          <div
            className="metric-value"
            style={{ color: getLossColor(current_stats.loss) }}
          >
            {current_stats.loss.toFixed(4)}
          </div>
          <div className="metric-label">Loss</div>
          <div
            className={`metric-trend ${getTrendClass(
              progress_indicators.loss_trend,
              true
            )}`}
          >
            {getTrendIcon(progress_indicators.loss_trend, true)}{" "}
            {getTrendText(progress_indicators.loss_trend, true)}
          </div>
        </div>

        {/* Accuracy */}
        <div className="metric-item fade-in">
          <div
            className="metric-value"
            style={{ color: getAccuracyColor(current_stats.accuracy) }}
          >
            {(current_stats.accuracy * 100).toFixed(1)}%
          </div>
          <div className="metric-label">Точность</div>
          <div
            className={`metric-trend ${getTrendClass(
              progress_indicators.accuracy_trend
            )}`}
          >
            {getTrendIcon(progress_indicators.accuracy_trend)}{" "}
            {getTrendText(progress_indicators.accuracy_trend)}
          </div>
        </div>

        {/* Learning Rate */}
        <div className="metric-item fade-in">
          <div
            className="metric-value"
            style={{ fontSize: "1.8em", color: "#8b5cf6" }}
          >
            {current_stats.learning_rate}
          </div>
          <div className="metric-label">Learning Rate</div>
          <div className="metric-trend trend-stable">⚙️ Adaptive</div>
        </div>

        {/* Training Duration */}
        <div className="metric-item fade-in">
          <div
            className="metric-value"
            style={{ fontSize: "1.4em", color: "#f59e0b" }}
          >
            {training_duration_formatted}
          </div>
          <div className="metric-label">Время обучения</div>
          {trainingStatus.is_training && (
            <div className="metric-trend trend-up">⏰ Идет</div>
          )}
        </div>

        {/* Samples Processed */}
        <div className="metric-item fade-in">
          <div
            className="metric-value"
            style={{ fontSize: "1.8em", color: "#06b6d4" }}
          >
            {current_stats.samples_processed.toLocaleString()}
          </div>
          <div className="metric-label">Обработано</div>
          <div className="metric-trend trend-stable">
            📈 {data_samples} за эпоху
          </div>
        </div>
      </div>

      {/* Progress Bars */}
      <div style={{ marginTop: "25px" }}>
        <div className="progress-container">
          <div className="progress-label">
            <span>Точность модели</span>
            <span>{(current_stats.accuracy * 100).toFixed(1)}%</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: `${current_stats.accuracy * 100}%`,
                background: `linear-gradient(90deg, ${getAccuracyColor(
                  current_stats.accuracy
                )}, ${getAccuracyColor(current_stats.accuracy)}aa)`,
              }}
            />
          </div>
        </div>

        <div className="progress-container">
          <div className="progress-label">
            <span>Прогресс обучения (обработано образцов)</span>
            <span>
              {(
                (current_stats.samples_processed /
                  Math.max(data_samples * 10, 1)) *
                100
              ).toFixed(0)}
              %
            </span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{
                width: `${Math.min(
                  (current_stats.samples_processed /
                    Math.max(data_samples * 10, 1)) *
                    100,
                  100
                )}%`,
              }}
            />
          </div>
        </div>
      </div>

      {/* Improvements Counter */}
      <div
        style={{
          marginTop: "20px",
          padding: "15px",
          background: progress_indicators.is_improving
            ? "linear-gradient(135deg, #f0fdf4, #dcfce7)"
            : "rgba(248, 250, 252, 0.8)",
          border: `1px solid ${
            progress_indicators.is_improving
              ? "#bbf7d0"
              : "rgba(226, 232, 240, 0.5)"
          }`,
          borderRadius: "12px",
          textAlign: "center",
        }}
      >
        <div
          style={{
            fontSize: "2em",
            fontWeight: "bold",
            color: progress_indicators.is_improving ? "#10b981" : "#6b7280",
            marginBottom: "5px",
          }}
        >
          {trainingStatus.improvements_count}
        </div>
        <div
          style={{
            color: progress_indicators.is_improving ? "#059669" : "#64748b",
            fontWeight: "600",
            marginBottom: "8px",
          }}
        >
          🏆 Улучшений модели
        </div>

        {progress_indicators.is_improving && (
          <div
            style={{
              background: "rgba(16, 185, 129, 0.1)",
              border: "1px solid rgba(16, 185, 129, 0.3)",
              borderRadius: "8px",
              padding: "8px",
              color: "#059669",
              fontSize: "0.8em",
              fontWeight: "500",
            }}
          >
            🎉 Модель активно улучшается!
          </div>
        )}
      </div>

      {/* Quality Assessment */}
      <div style={{ marginTop: "20px" }}>
        <h4
          style={{
            fontSize: "1em",
            color: "#374151",
            marginBottom: "10px",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          🎯 Оценка качества модели
        </h4>

        <div
          style={{
            display: "grid",
            gap: "10px",
          }}
        >
          <div
            className={`quality-indicator ${
              current_stats.accuracy >= 0.8
                ? "good"
                : current_stats.accuracy >= 0.6
                ? "medium"
                : "poor"
            }`}
          >
            <span className="quality-icon">
              {current_stats.accuracy >= 0.8
                ? "🟢"
                : current_stats.accuracy >= 0.6
                ? "🟡"
                : "🔴"}
            </span>
            <span className="quality-text">
              Точность:{" "}
              {current_stats.accuracy >= 0.8
                ? "Отличная"
                : current_stats.accuracy >= 0.6
                ? "Хорошая"
                : "Требует улучшения"}
            </span>
          </div>

          <div
            className={`quality-indicator ${
              current_stats.loss <= 0.5
                ? "good"
                : current_stats.loss <= 1.0
                ? "medium"
                : "poor"
            }`}
          >
            <span className="quality-icon">
              {current_stats.loss <= 0.5
                ? "🟢"
                : current_stats.loss <= 1.0
                ? "🟡"
                : "🔴"}
            </span>
            <span className="quality-text">
              Loss:{" "}
              {current_stats.loss <= 0.5
                ? "Отличный"
                : current_stats.loss <= 1.0
                ? "Хороший"
                : "Требует улучшения"}
            </span>
          </div>

          <div
            className={`quality-indicator ${
              data_samples >= 100
                ? "good"
                : data_samples >= 50
                ? "medium"
                : "poor"
            }`}
          >
            <span className="quality-icon">
              {data_samples >= 100 ? "🟢" : data_samples >= 50 ? "🟡" : "🔴"}
            </span>
            <span className="quality-text">
              Данные:{" "}
              {data_samples >= 100
                ? "Достаточно"
                : data_samples >= 50
                ? "Минимум"
                : "Нужно больше"}
            </span>
          </div>
        </div>
      </div>

      {/* Last Update */}
      {current_stats.last_update && (
        <div
          style={{
            marginTop: "15px",
            fontSize: "0.8em",
            color: "#64748b",
            textAlign: "center",
            padding: "8px",
            background: "rgba(248, 250, 252, 0.5)",
            borderRadius: "8px",
          }}
        >
          🔄 Последнее обновление:{" "}
          {new Date(current_stats.last_update).toLocaleString()}
        </div>
      )}
    </div>
  );
};

export default TrainingMetrics;
