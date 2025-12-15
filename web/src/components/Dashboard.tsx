import React, { useState, useEffect } from "react";
import axios from "axios";
import TrainingMetrics from "./TrainingMetrics";
import TrainingChart from "./TrainingChart";
import ModelsList from "./ModelsList";
import ControlPanel from "./ControlPanel";
import DataStats from "./DataStats";
import TrainingProcess from "./TrainingProcess";
import {
  Brain,
  Activity,
  TrendingUp,
  Database,
  Clock,
  Target,
  Zap,
  AlertCircle,
  CheckCircle,
  Loader,
  RefreshCw,
} from "lucide-react";
import { API_BASE_URL } from "../config/api";

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
    improvements: Array<{
      epoch: number;
      accuracy: number;
      loss: number;
      timestamp: string;
    }>;
  };
  recent_history: Array<{
    epoch: number;
    loss: number;
    accuracy: number;
    timestamp: string;
    samples: number;
  }>;
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

interface DataStatsType {
  papers_count: number;
  journals_count: number;
  top_concepts: Array<{
    name: string;
    count: number;
  }>;
}

const Dashboard: React.FC = () => {
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(
    null
  );
  const [dataStats, setDataStats] = useState<DataStatsType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [connectionStatus, setConnectionStatus] = useState<
    "connected" | "connecting" | "error"
  >("connecting");

  // Функция для получения статуса обучения
  const fetchTrainingStatus = async () => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/v1/training/status`
      );
      setTrainingStatus(response.data);
      setConnectionStatus("connected");
      setError(null);
    } catch (err) {
      console.error("Error fetching training status:", err);
      setConnectionStatus("error");
      setError("Не удается подключиться к серверу AI");
    }
  };

  // Функция для получения статистики данных
  const fetchDataStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/v1/data/stats`);
      // Проверяем что ответ не содержит ошибку
      if (response.data && !response.data.error) {
        setDataStats(response.data);
      } else {
        console.error("API returned error:", response.data?.error);
        // Устанавливаем пустые данные по умолчанию
        setDataStats({
          papers_count: 0,
          journals_count: 0,
          top_concepts: [],
        });
      }
    } catch (err) {
      console.error("Error fetching data stats:", err);
      // Устанавливаем пустые данные по умолчанию
      setDataStats({
        papers_count: 0,
        journals_count: 0,
        top_concepts: [],
      });
    }
  };

  // Функция для запуска обучения
  const startTraining = async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/v1/training/start`);
      await fetchTrainingStatus();
    } catch (err) {
      console.error("Error starting training:", err);
      setError("Ошибка запуска обучения");
    }
  };

  // Функция для остановки обучения
  const stopTraining = async () => {
    try {
      await axios.post(`${API_BASE_URL}/api/v1/training/stop`);
      await fetchTrainingStatus();
    } catch (err) {
      console.error("Error stopping training:", err);
      setError("Ошибка остановки обучения");
    }
  };

  // Функция для загрузки данных
  const loadData = async (
    query: string = "artificial intelligence",
    papers_limit: number = 50
  ) => {
    try {
      await axios.post(`${API_BASE_URL}/api/v1/data/load`, {
        query,
        papers_limit,
        journals_limit: 20,
      });
      // Подождем немного и обновим статистику
      setTimeout(() => {
        fetchDataStats();
      }, 2000);
    } catch (err) {
      console.error("Error loading data:", err);
      setError("Ошибка загрузки данных");
    }
  };

  // Эффект для периодического обновления данных
  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      await Promise.all([fetchTrainingStatus(), fetchDataStats()]);
      setLoading(false);
      setLastUpdate(new Date());
    };

    // Первоначальная загрузка
    fetchData();

    // Обновление каждые 2 секунды для более плавного UX
    const interval = setInterval(fetchData, 2000);

    return () => clearInterval(interval);
  }, []);

  const getMainStatus = () => {
    if (connectionStatus === "connecting" || loading) {
      return {
        text: "ПОДКЛЮЧЕНИЕ",
        class: "status-loading",
        icon: <Loader className="w-4 h-4 animate-spin" />,
      };
    }
    if (connectionStatus === "error") {
      return {
        text: "НЕТ СВЯЗИ",
        class: "status-stopped",
        icon: <AlertCircle className="w-4 h-4" />,
      };
    }
    if (trainingStatus?.is_training) {
      return {
        text: "ОБУЧАЕТСЯ",
        class: "status-training",
        icon: <Brain className="w-4 h-4" />,
      };
    }
    return {
      text: "ОСТАНОВЛЕНО",
      class: "status-stopped",
      icon: <Activity className="w-4 h-4" />,
    };
  };

  const getCurrentActivity = () => {
    if (!trainingStatus) return "Инициализация системы...";

    if (trainingStatus.is_training) {
      if (trainingStatus.data_samples === 0) {
        return "Загрузка данных для обучения...";
      }
      if (trainingStatus.current_stats.epoch === 0) {
        return "Подготовка к первой эпохе обучения...";
      }
      return `Обучение модели (Эпоха ${trainingStatus.current_stats.epoch})`;
    }

    return "Система готова к работе";
  };

  if (loading && !trainingStatus) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-8 text-center">
          <div className="flex flex-col items-center space-y-4">
            <div className="p-4 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full">
              <Brain className="w-8 h-8 text-white animate-pulse" />
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-semibold text-white">
                Инициализация AI Scientometer
              </h3>
              <p className="text-slate-400">Подключение к системе...</p>
            </div>
            <Loader className="w-6 h-6 text-blue-400 animate-spin" />
          </div>
        </div>
      </div>
    );
  }

  const mainStatus = getMainStatus();

  return (
    <div className="min-h-screen bg-slate-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6">
          <div className="text-center mb-6">
            <div className="flex items-center justify-center space-x-3 mb-2">
              <div className="p-3 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl">
                <Brain className="w-8 h-8 text-white" />
              </div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                AI SCIENTOMETER
              </h1>
            </div>
            <p className="text-slate-400 text-lg">
              Интеллектуальная система анализа научных публикаций с непрерывным
              обучением
            </p>
          </div>

          {/* Status Bar */}
          <div className="flex flex-col lg:flex-row items-center justify-between gap-4 p-4 bg-slate-900/50 rounded-xl border border-slate-700/30">
            <div className="flex items-center space-x-4">
              <div
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium ${
                  mainStatus.class === "status-training"
                    ? "bg-green-500/20 text-green-400 border border-green-500/30"
                    : mainStatus.class === "status-loading"
                    ? "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30"
                    : "bg-red-500/20 text-red-400 border border-red-500/30"
                }`}
              >
                {mainStatus.icon}
                <span className="font-semibold">{mainStatus.text}</span>
              </div>

              {trainingStatus?.is_training && (
                <div className="flex items-center space-x-2 px-3 py-1.5 bg-blue-500/10 border border-blue-500/20 rounded-full">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse"></div>
                  <span className="text-sm font-medium text-blue-400">
                    LIVE TRAINING
                  </span>
                </div>
              )}
            </div>

            {trainingStatus && (
              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center space-x-2 px-3 py-1.5 bg-slate-700/50 rounded-lg">
                  <TrendingUp className="w-4 h-4 text-blue-400" />
                  <span className="text-sm text-slate-300">
                    Эпоха:{" "}
                    <span className="font-semibold text-white">
                      {trainingStatus.current_stats.epoch}
                    </span>
                  </span>
                </div>
                <div className="flex items-center space-x-2 px-3 py-1.5 bg-slate-700/50 rounded-lg">
                  <Target className="w-4 h-4 text-green-400" />
                  <span className="text-sm text-slate-300">
                    Точность:{" "}
                    <span className="font-semibold text-white">
                      {(trainingStatus.current_stats.accuracy * 100).toFixed(1)}
                      %
                    </span>
                  </span>
                </div>
                <div className="flex items-center space-x-2 px-3 py-1.5 bg-slate-700/50 rounded-lg">
                  <Database className="w-4 h-4 text-purple-400" />
                  <span className="text-sm text-slate-300">
                    Данных:{" "}
                    <span className="font-semibold text-white">
                      {trainingStatus.data_samples}
                    </span>
                  </span>
                </div>
                {trainingStatus.is_training && (
                  <div className="flex items-center space-x-2 px-3 py-1.5 bg-slate-700/50 rounded-lg">
                    <Clock className="w-4 h-4 text-orange-400" />
                    <span className="text-sm text-slate-300">
                      {trainingStatus.training_duration_formatted}
                    </span>
                  </div>
                )}
              </div>
            )}

            <div className="flex items-center space-x-2 text-sm text-slate-400">
              <RefreshCw className="w-4 h-4" />
              <span>Обновлено: {lastUpdate.toLocaleTimeString()}</span>
            </div>
          </div>

          {/* Current Activity */}
          <div className="mt-4 p-4 bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-xl">
            <div className="flex items-center justify-center space-x-2 text-blue-400">
              <Activity className="w-5 h-5" />
              <span className="font-medium">Текущая активность:</span>
              <span className="text-white font-semibold">
                {getCurrentActivity()}
              </span>
            </div>
          </div>

          {error && (
            <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-xl flex items-center space-x-3">
              <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
              <span className="text-red-400 font-medium">{error}</span>
            </div>
          )}
        </div>

        {/* Training Process Indicator */}
        {trainingStatus && (
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6">
            <TrainingProcess trainingStatus={trainingStatus} />
          </div>
        )}

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Training Metrics */}
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6">
            <TrainingMetrics trainingStatus={trainingStatus} />
          </div>

          {/* Control Panel */}
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6">
            <ControlPanel
              isTraining={trainingStatus?.is_training || false}
              onStart={startTraining}
              onStop={stopTraining}
              onLoadData={loadData}
              trainingStatus={trainingStatus}
            />
          </div>

          {/* Data Statistics */}
          <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6">
            <DataStats dataStats={dataStats} />
          </div>
        </div>

        {/* Charts Section */}
        {trainingStatus?.recent_history &&
          trainingStatus.recent_history.length > 0 && (
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6">
              <TrainingChart history={trainingStatus.recent_history} />
            </div>
          )}

        {/* Models List */}
        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6">
          <ModelsList />
        </div>

        {/* Improvements History */}
        {trainingStatus?.current_stats?.improvements &&
          trainingStatus.current_stats.improvements.length > 0 && (
            <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6">
              <div className="flex items-center space-x-3 mb-6">
                <div className="p-2 bg-gradient-to-br from-yellow-500 to-orange-500 rounded-lg">
                  <TrendingUp className="w-5 h-5 text-white" />
                </div>
                <h3 className="text-xl font-semibold text-white">
                  История улучшений модели
                </h3>
              </div>
              <div className="space-y-4">
                {trainingStatus.current_stats.improvements
                  .slice(-5)
                  .reverse()
                  .map((improvement, index) => (
                    <div
                      key={index}
                      className="bg-slate-900/50 border border-slate-700/30 rounded-xl p-4 hover:bg-slate-900/70 transition-colors"
                    >
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-3">
                        <div className="flex items-center space-x-2">
                          <Target className="w-4 h-4 text-green-400" />
                          <span className="font-semibold text-green-400">
                            Эпоха {improvement.epoch}
                          </span>
                        </div>
                        <span className="text-sm text-slate-400">
                          {new Date(improvement.timestamp).toLocaleString()}
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-4">
                        <div className="flex items-center space-x-2">
                          <TrendingUp className="w-4 h-4 text-blue-400" />
                          <span className="text-sm text-slate-300">
                            Точность:{" "}
                            <span className="font-semibold text-white">
                              {(improvement.accuracy * 100).toFixed(2)}%
                            </span>
                          </span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <Zap className="w-4 h-4 text-purple-400" />
                          <span className="text-sm text-slate-300">
                            Loss:{" "}
                            <span className="font-semibold text-white">
                              {improvement.loss.toFixed(4)}
                            </span>
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          )}

        {/* System Information */}
        <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="p-2 bg-gradient-to-br from-gray-500 to-gray-600 rounded-lg">
              <Activity className="w-5 h-5 text-white" />
            </div>
            <h3 className="text-xl font-semibold text-white">
              Системная информация
            </h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="bg-slate-900/50 border border-slate-700/30 rounded-xl p-4 text-center">
              <div className="text-xs text-slate-400 mb-1">Backend</div>
              <div className="font-semibold text-white">FastAPI + MongoDB</div>
            </div>
            <div className="bg-slate-900/50 border border-slate-700/30 rounded-xl p-4 text-center">
              <div className="text-xs text-slate-400 mb-1">AI Model</div>
              <div className="font-semibold text-white">
                SentenceTransformer
              </div>
            </div>
            <div className="bg-slate-900/50 border border-slate-700/30 rounded-xl p-4 text-center">
              <div className="text-xs text-slate-400 mb-1">Training</div>
              <div className="font-semibold text-white">SGDClassifier</div>
            </div>
            <div className="bg-slate-900/50 border border-slate-700/30 rounded-xl p-4 text-center">
              <div className="text-xs text-slate-400 mb-1">Data Source</div>
              <div className="font-semibold text-white">OpenAlex API</div>
            </div>
            <div className="bg-slate-900/50 border border-slate-700/30 rounded-xl p-4 text-center">
              <div className="text-xs text-slate-400 mb-1">Update Rate</div>
              <div className="font-semibold text-white">2 секунды</div>
            </div>
            <div className="bg-slate-900/50 border border-slate-700/30 rounded-xl p-4 text-center">
              <div className="text-xs text-slate-400 mb-1">Connection</div>
              <div
                className={`font-semibold flex items-center justify-center space-x-1 ${
                  connectionStatus === "connected"
                    ? "text-green-400"
                    : connectionStatus === "connecting"
                    ? "text-yellow-400"
                    : "text-red-400"
                }`}
              >
                {connectionStatus === "connected" ? (
                  <>
                    <CheckCircle className="w-4 h-4" />
                    <span>Подключено</span>
                  </>
                ) : connectionStatus === "connecting" ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    <span>Подключение...</span>
                  </>
                ) : (
                  <>
                    <AlertCircle className="w-4 h-4" />
                    <span>Ошибка</span>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
