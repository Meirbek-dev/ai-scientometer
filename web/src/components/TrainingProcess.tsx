import React from "react";

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
  data_samples: number;
  training_duration_seconds: number;
  progress_indicators: {
    loss_trend: string;
    accuracy_trend: string;
    is_improving: boolean;
  };
}

interface TrainingProcessProps {
  trainingStatus: TrainingStatus;
}

const TrainingProcess: React.FC<TrainingProcessProps> = ({
  trainingStatus,
}) => {
  const { current_stats, data_samples, is_training } = trainingStatus;

  // Определяем этапы процесса обучения
  const getTrainingSteps = () => {
    const steps = [
      {
        id: "data_loading",
        title: "Загрузка данных",
        description: "Получение научных статей из OpenAlex API",
        status:
          data_samples > 0 ? "completed" : is_training ? "active" : "pending",
        details: `${data_samples} образцов загружено`,
        icon: "📚",
      },
      {
        id: "preprocessing",
        title: "Предобработка данных",
        description: "Векторизация текстов и подготовка признаков",
        status:
          data_samples > 0 && current_stats.epoch >= 0
            ? "completed"
            : data_samples > 0 && is_training
            ? "active"
            : "pending",
        details: "TF-IDF векторизация, создание меток",
        icon: "⚙️",
      },
      {
        id: "model_training",
        title: "Обучение модели",
        description: "SGDClassifier с градиентным спуском",
        status: current_stats.epoch > 0 ? "active" : "pending",
        details: `Эпоха ${current_stats.epoch}, LR: ${current_stats.learning_rate}`,
        icon: "🧠",
      },
      {
        id: "evaluation",
        title: "Оценка качества",
        description: "Вычисление метрик accuracy и loss",
        status: current_stats.epoch > 0 ? "completed" : "pending",
        details: `Accuracy: ${(current_stats.accuracy * 100).toFixed(
          2
        )}%, Loss: ${current_stats.loss.toFixed(4)}`,
        icon: "📊",
      },
      {
        id: "model_saving",
        title: "Сохранение модели",
        description: "Сохранение лучших весов модели",
        status:
          current_stats.improvements?.length > 0 ? "completed" : "pending",
        details: `${
          current_stats.improvements?.length || 0
        } улучшений сохранено`,
        icon: "💾",
      },
      {
        id: "next_iteration",
        title: "Следующая итерация",
        description: "Подготовка к следующей эпохе обучения",
        status: is_training && current_stats.epoch > 0 ? "active" : "pending",
        details: "Цикл повторяется каждые 10 секунд",
        icon: "🔄",
      },
    ];

    return steps;
  };

  const steps = getTrainingSteps();
  const completedSteps = steps.filter(
    (step) => step.status === "completed"
  ).length;
  const progress = (completedSteps / steps.length) * 100;

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return "✅";
      case "active":
        return "⏳";
      default:
        return "⭕";
    }
  };

  const getStatusClass = (status: string) => {
    switch (status) {
      case "completed":
        return "completed";
      case "active":
        return "active";
      default:
        return "pending";
    }
  };

  return (
    <div>
      <h3 className="card-title">🔄 Процесс обучения AI модели</h3>

      {/* Overall Progress */}
      <div className="progress-container">
        <div className="progress-label">
          <span>Общий прогресс</span>
          <span>{Math.round(progress)}%</span>
        </div>
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
      </div>

      {/* Training Pipeline */}
      <div className="training-process">
        <div className="process-title">
          <span>🏭</span>
          Конвейер машинного обучения
        </div>

        <div className="process-steps">
          {steps.map((step, index) => (
            <div
              key={step.id}
              className={`process-step ${
                step.status === "active" ? "active" : ""
              }`}
            >
              <div className={`step-icon ${getStatusClass(step.status)}`}>
                {getStatusIcon(step.status)}
              </div>

              <div className="step-content" style={{ flex: 1 }}>
                <div className="step-header">
                  <div className="step-title">
                    {step.icon} {step.title}
                  </div>
                  <div className="step-status">
                    {step.status === "completed" && "✅ Завершено"}
                    {step.status === "active" && "⏳ Выполняется"}
                    {step.status === "pending" && "⭕ Ожидание"}
                  </div>
                </div>

                <div className="step-description">{step.description}</div>

                <div className="step-details">{step.details}</div>
              </div>

              {index < steps.length - 1 && (
                <div className="step-connector">↓</div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Current Packages/Libraries Being Used */}
      <div style={{ marginTop: "25px" }}>
        <h4
          style={{
            marginBottom: "15px",
            color: "#374151",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          📦 Используемые пакеты и технологии
        </h4>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: "15px",
          }}
        >
          <div className="tech-stack-item">
            <div className="tech-icon">🐍</div>
            <div className="tech-info">
              <div className="tech-name">Python</div>
              <div className="tech-desc">Основной язык</div>
            </div>
          </div>

          <div className="tech-stack-item">
            <div className="tech-icon">🤖</div>
            <div className="tech-info">
              <div className="tech-name">scikit-learn</div>
              <div className="tech-desc">SGDClassifier</div>
            </div>
          </div>

          <div className="tech-stack-item">
            <div className="tech-icon">🧠</div>
            <div className="tech-info">
              <div className="tech-name">SentenceTransformers</div>
              <div className="tech-desc">Эмбеддинги текста</div>
            </div>
          </div>

          <div className="tech-stack-item">
            <div className="tech-icon">📊</div>
            <div className="tech-info">
              <div className="tech-name">TF-IDF</div>
              <div className="tech-desc">Векторизация</div>
            </div>
          </div>

          <div className="tech-stack-item">
            <div className="tech-icon">🗄️</div>
            <div className="tech-info">
              <div className="tech-name">MongoDB</div>
              <div className="tech-desc">База данных</div>
            </div>
          </div>

          <div className="tech-stack-item">
            <div className="tech-icon">🌐</div>
            <div className="tech-info">
              <div className="tech-name">OpenAlex API</div>
              <div className="tech-desc">Источник данных</div>
            </div>
          </div>
        </div>
      </div>

      {/* Training Parameters */}
      <div style={{ marginTop: "25px" }}>
        <h4
          style={{
            marginBottom: "15px",
            color: "#374151",
            display: "flex",
            alignItems: "center",
            gap: "8px",
          }}
        >
          ⚙️ Параметры обучения
        </h4>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
            gap: "15px",
            padding: "15px",
            background: "rgba(248, 250, 252, 0.8)",
            borderRadius: "12px",
            border: "1px solid rgba(226, 232, 240, 0.5)",
          }}
        >
          <div className="param-item">
            <div className="param-label">Learning Rate</div>
            <div className="param-value">{current_stats.learning_rate}</div>
          </div>

          <div className="param-item">
            <div className="param-label">Batch Size</div>
            <div className="param-value">{data_samples}</div>
          </div>

          <div className="param-item">
            <div className="param-label">Optimizer</div>
            <div className="param-value">SGD Adaptive</div>
          </div>

          <div className="param-item">
            <div className="param-label">Update Frequency</div>
            <div className="param-value">10 секунд</div>
          </div>

          <div className="param-item">
            <div className="param-label">Features</div>
            <div className="param-value">TF-IDF (1000)</div>
          </div>

          <div className="param-item">
            <div className="param-label">Classes</div>
            <div className="param-value">3 (High/Med/Low)</div>
          </div>
        </div>
      </div>

      {/* Real-time Insights */}
      {is_training && (
        <div
          style={{
            marginTop: "20px",
            padding: "15px",
            background: "linear-gradient(135deg, #f0fdf4, #dcfce7)",
            border: "1px solid #bbf7d0",
            borderRadius: "12px",
          }}
        >
          <div
            style={{
              fontWeight: "600",
              color: "#059669",
              marginBottom: "10px",
              display: "flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            ⚡ Реальное время: Что происходит сейчас
          </div>

          <div style={{ fontSize: "0.9em", color: "#065f46" }}>
            <div>• Модель обучается на {data_samples} научных статьях</div>
            <div>
              • Текущая эпоха: {current_stats.epoch} (обновляется каждые 10
              секунд)
            </div>
            <div>• Обработано образцов: {current_stats.samples_processed}</div>
            <div>
              • Тренд точности:{" "}
              {trainingStatus.progress_indicators.accuracy_trend ===
              "increasing"
                ? "📈 Растет"
                : trainingStatus.progress_indicators.accuracy_trend ===
                  "decreasing"
                ? "📉 Снижается"
                : "📊 Стабильно"}
            </div>
            <div>
              • Тренд loss:{" "}
              {trainingStatus.progress_indicators.loss_trend === "decreasing"
                ? "📉 Снижается (хорошо!)"
                : trainingStatus.progress_indicators.loss_trend === "increasing"
                ? "📈 Растет"
                : "📊 Стабильно"}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingProcess;
