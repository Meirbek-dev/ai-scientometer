import React, { useState, useEffect } from "react";
import axios from "axios";
import { API_BASE_URL } from "../config/api";

interface Model {
  filename: string;
  size_mb: number;
  created_at: string;
  is_best: boolean;
}

interface ModelsResponse {
  models: Model[];
  total_models: number;
  best_model: string;
}

const ModelsList: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchModels = async () => {
    try {
      const response = await axios.get<ModelsResponse>(
        `${API_BASE_URL}/api/v1/training/models`
      );
      setModels(response.data.models || []);
      setError(null);
    } catch (err) {
      console.error("Error fetching models:", err);
      setError("Ошибка загрузки моделей");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();

    // Обновляем каждые 10 секунд
    const interval = setInterval(fetchModels, 10000);

    return () => clearInterval(interval);
  }, []);

  const downloadModel = async (filename: string) => {
    try {
      // Создаем ссылку для скачивания
      const link = document.createElement("a");
      link.href = `${API_BASE_URL}/api/v1/datasets/download/latest?format=csv`;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      console.error("Error downloading model:", err);
    }
  };

  const formatFileSize = (sizeMb: number): string => {
    if (sizeMb < 1) {
      return `${(sizeMb * 1024).toFixed(0)} KB`;
    }
    return `${sizeMb.toFixed(2)} MB`;
  };

  const getModelType = (filename: string): string => {
    if (filename.includes("best_model")) {
      return "🏆 Лучшая модель";
    }
    if (filename.includes("checkpoint")) {
      return "💾 Checkpoint";
    }
    return "🤖 Модель";
  };

  if (loading) {
    return (
      <div className="dashboard-card">
        <h3 className="card-title">🏆 Обученные модели</h3>
        <div className="loading">Загрузка моделей...</div>
      </div>
    );
  }

  return (
    <div>
      <h3 className="card-title">🏆 Обученные модели</h3>

      {error && (
        <div className="error" style={{ marginBottom: "15px" }}>
          {error}
        </div>
      )}

      {models.length === 0 ? (
        <div
          style={{
            textAlign: "center",
            padding: "40px",
            color: "#666",
            background: "#f8fafc",
            borderRadius: "8px",
            border: "2px dashed #d1d5db",
          }}
        >
          <div style={{ fontSize: "3em", marginBottom: "10px" }}>🤖</div>
          <div
            style={{
              fontSize: "1.1em",
              fontWeight: "bold",
              marginBottom: "5px",
            }}
          >
            Модели еще не созданы
          </div>
          <div style={{ fontSize: "0.9em" }}>
            Запустите обучение, чтобы создать первую модель
          </div>
        </div>
      ) : (
        <>
          {/* Summary */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
              gap: "15px",
              marginBottom: "20px",
              padding: "15px",
              background: "#f0fdf4",
              borderRadius: "8px",
              border: "1px solid #bbf7d0",
            }}
          >
            <div style={{ textAlign: "center" }}>
              <div
                style={{
                  fontSize: "1.5em",
                  fontWeight: "bold",
                  color: "#059669",
                }}
              >
                {models.length}
              </div>
              <div style={{ fontSize: "0.8em", color: "#065f46" }}>
                Всего моделей
              </div>
            </div>

            <div style={{ textAlign: "center" }}>
              <div
                style={{
                  fontSize: "1.5em",
                  fontWeight: "bold",
                  color: "#0891b2",
                }}
              >
                {models
                  .reduce((sum, model) => sum + model.size_mb, 0)
                  .toFixed(1)}
              </div>
              <div style={{ fontSize: "0.8em", color: "#0e7490" }}>
                MB общий размер
              </div>
            </div>

            <div style={{ textAlign: "center" }}>
              <div
                style={{
                  fontSize: "1.5em",
                  fontWeight: "bold",
                  color: "#7c3aed",
                }}
              >
                {models.filter((m) => m.is_best).length}
              </div>
              <div style={{ fontSize: "0.8em", color: "#5b21b6" }}>
                Лучших моделей
              </div>
            </div>
          </div>

          {/* Models Table */}
          <div style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Тип</th>
                  <th>Имя файла</th>
                  <th>Размер</th>
                  <th>Создана</th>
                  <th>Действия</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model, index) => (
                  <tr
                    key={index}
                    style={{
                      background: model.is_best ? "#fef3c7" : undefined,
                    }}
                  >
                    <td>
                      <span
                        style={{
                          fontSize: "0.8em",
                          padding: "2px 6px",
                          borderRadius: "4px",
                          background: model.is_best ? "#fbbf24" : "#e5e7eb",
                          color: model.is_best ? "white" : "#374151",
                        }}
                      >
                        {getModelType(model.filename)}
                      </span>
                    </td>
                    <td>
                      <div
                        style={{
                          fontFamily: "monospace",
                          fontSize: "0.8em",
                          wordBreak: "break-all",
                        }}
                      >
                        {model.filename}
                      </div>
                    </td>
                    <td>{formatFileSize(model.size_mb)}</td>
                    <td style={{ fontSize: "0.8em" }}>
                      {new Date(model.created_at).toLocaleString()}
                    </td>
                    <td>
                      <button
                        className="download-btn"
                        onClick={() => downloadModel(model.filename)}
                        title="Скачать модель"
                      >
                        ⬇️ Скачать
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Model Usage Instructions */}
          <div
            style={{
              marginTop: "20px",
              padding: "15px",
              background: "#f0f9ff",
              border: "1px solid #bae6fd",
              borderRadius: "8px",
            }}
          >
            <div
              style={{
                fontWeight: "bold",
                marginBottom: "10px",
                color: "#0369a1",
              }}
            >
              📖 Как использовать модели:
            </div>
            <div style={{ fontSize: "0.8em", color: "#0369a1" }}>
              <div style={{ marginBottom: "5px" }}>
                <strong>Python:</strong>
              </div>
              <pre
                style={{
                  background: "#e0f2fe",
                  padding: "8px",
                  borderRadius: "4px",
                  fontSize: "0.7em",
                  overflow: "auto",
                }}
              >
                {`import joblib
model = joblib.load('best_model_epoch_X.joblib')
predictions = model.predict(your_data)`}
              </pre>
            </div>
          </div>

          {/* Refresh Button */}
          <div style={{ textAlign: "center", marginTop: "15px" }}>
            <button
              className="control-button btn-secondary"
              onClick={fetchModels}
              disabled={loading}
            >
              🔄 Обновить список
            </button>
          </div>
        </>
      )}
    </div>
  );
};

export default ModelsList;
