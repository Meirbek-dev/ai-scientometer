import React, { useState } from "react";
import {
  Play,
  Square,
  Download,
  Settings,
  BookOpen,
  Search,
  BarChart3,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  CheckCircle,
  Info,
  Loader,
  Brain,
  Database,
  Globe,
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

interface ControlPanelProps {
  isTraining: boolean;
  onStart: () => Promise<void>;
  onStop: () => Promise<void>;
  onLoadData: (query: string, limit: number) => Promise<void>;
  trainingStatus: TrainingStatus | null;
}

const ControlPanel: React.FC<ControlPanelProps> = ({
  isTraining,
  onStart,
  onStop,
  onLoadData,
  trainingStatus,
}) => {
  const [loading, setLoading] = useState(false);
  const [dataQuery, setDataQuery] = useState("artificial intelligence");
  const [dataLimit, setDataLimit] = useState(50);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleStart = async () => {
    setLoading(true);
    try {
      await onStart();
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      await onStop();
    } finally {
      setLoading(false);
    }
  };

  const handleLoadData = async () => {
    setLoading(true);
    try {
      await onLoadData(dataQuery, dataLimit);
    } finally {
      setLoading(false);
    }
  };

  const quickQueries = [
    { name: "AI & ML", query: "artificial intelligence machine learning" },
    { name: "Deep Learning", query: "deep learning neural networks" },
    { name: "Computer Vision", query: "computer vision image recognition" },
    { name: "NLP", query: "natural language processing text mining" },
    { name: "Robotics", query: "robotics automation control systems" },
    {
      name: "Quantum Computing",
      query: "quantum computing quantum algorithms",
    },
    {
      name: "Blockchain",
      query: "blockchain cryptocurrency distributed systems",
    },
    {
      name: "Bioinformatics",
      query: "bioinformatics computational biology genomics",
    },
  ];

  const getTrainingRecommendation = () => {
    if (!trainingStatus) return null;

    const { current_stats, data_samples, progress_indicators } = trainingStatus;

    if (data_samples < 50) {
      return {
        type: "warning",
        title: "⚠️ Недостаточно данных",
        message:
          "Рекомендуется загрузить минимум 50-100 статей для качественного обучения",
      };
    }

    if (current_stats.accuracy < 0.6) {
      return {
        type: "info",
        title: "📈 Низкая точность",
        message:
          "Модель все еще обучается. Попробуйте загрузить больше данных или подождите несколько эпох",
      };
    }

    if (progress_indicators.is_improving) {
      return {
        type: "success",
        title: "🎉 Отличный прогресс!",
        message:
          "Модель активно улучшается. Продолжайте обучение для достижения лучших результатов",
      };
    }

    return {
      type: "info",
      title: "💡 Совет",
      message:
        "Модель стабильна. Можете загрузить новые данные для дальнейшего улучшения",
    };
  };

  const recommendation = getTrainingRecommendation();

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3">
        <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
          <Settings className="w-5 h-5 text-white" />
        </div>
        <h3 className="text-xl font-semibold text-white">Панель управления</h3>
      </div>

      {/* Training Status Overview */}
      {trainingStatus && (
        <div
          className={`p-4 rounded-xl border ${
            isTraining
              ? "bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-green-500/30"
              : "bg-gradient-to-br from-red-500/10 to-rose-500/10 border-red-500/30"
          }`}
        >
          <div className="flex items-center space-x-3 mb-3">
            <div
              className={`p-2 rounded-lg ${
                isTraining ? "bg-green-500/20" : "bg-red-500/20"
              }`}
            >
              {isTraining ? (
                <Brain className="w-5 h-5 text-green-400" />
              ) : (
                <Square className="w-5 h-5 text-red-400" />
              )}
            </div>
            <div>
              <div
                className={`font-bold ${
                  isTraining ? "text-green-400" : "text-red-400"
                }`}
              >
                {isTraining
                  ? "Система активно обучается"
                  : "Обучение остановлено"}
              </div>
              <div
                className={`text-sm ${
                  isTraining ? "text-green-300/80" : "text-red-300/80"
                }`}
              >
                {isTraining
                  ? `Эпоха ${trainingStatus.current_stats.epoch} • ${trainingStatus.data_samples} образцов`
                  : 'Нажмите "Запустить обучение" для начала'}
              </div>
            </div>
          </div>

          {isTraining && (
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="flex items-center space-x-2 text-green-300">
                <BarChart3 className="w-4 h-4" />
                <span>
                  Точность:{" "}
                  {(trainingStatus.current_stats.accuracy * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex items-center space-x-2 text-green-300">
                <BarChart3 className="w-4 h-4" />
                <span>
                  Loss: {trainingStatus.current_stats.loss.toFixed(3)}
                </span>
              </div>
              <div className="flex items-center space-x-2 text-green-300">
                <BarChart3 className="w-4 h-4" />
                <span>{trainingStatus.training_duration_formatted}</span>
              </div>
              <div className="flex items-center space-x-2 text-green-300">
                <BarChart3 className="w-4 h-4" />
                <span>{trainingStatus.improvements_count} улучш.</span>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Training Controls */}
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <Brain className="w-5 h-5 text-blue-400" />
          <h4 className="text-lg font-semibold text-white">
            Управление обучением
          </h4>
        </div>

        <div className="flex flex-wrap gap-3">
          {!isTraining ? (
            <button
              className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white font-medium rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={handleStart}
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" />
                  <span>Запуск...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  <span>Запустить обучение</span>
                </>
              )}
            </button>
          ) : (
            <button
              className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-red-500 to-rose-500 hover:from-red-600 hover:to-rose-600 text-white font-medium rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              onClick={handleStop}
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader className="w-4 h-4 animate-spin" />
                  <span>Остановка...</span>
                </>
              ) : (
                <>
                  <Square className="w-4 h-4" />
                  <span>Остановить обучение</span>
                </>
              )}
            </button>
          )}

          <button
            className="flex items-center space-x-2 px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white font-medium rounded-lg transition-all duration-200"
            onClick={() => window.open(`${API_BASE_URL}/docs`, "_blank")}
          >
            <BookOpen className="w-4 h-4" />
            <span>API Документация</span>
          </button>

          <button
            className="flex items-center space-x-2 px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white font-medium rounded-lg transition-all duration-200"
            onClick={() =>
              window.open(`${API_BASE_URL}/api/v1/training/status`, "_blank")
            }
          >
            <Search className="w-4 h-4" />
            <span>Raw JSON</span>
          </button>
        </div>

        {/* Training Info */}
        <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-xl">
          <div className="flex items-center space-x-2 mb-3">
            <Info className="w-4 h-4 text-blue-400" />
            <span className="font-semibold text-blue-400">
              Как работает обучение:
            </span>
          </div>
          <ul className="space-y-1 text-sm text-blue-300 pl-4">
            <li>• Автоматический цикл каждые 10 секунд</li>
            <li>• SGDClassifier с адаптивной скоростью обучения</li>
            <li>• Сохранение лучших моделей автоматически</li>
            <li>• Загрузка новых данных при необходимости</li>
          </ul>
        </div>
      </div>

      {/* Data Loading Controls */}
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <Database className="w-5 h-5 text-purple-400" />
          <h4 className="text-lg font-semibold text-white">
            Управление данными
          </h4>
        </div>

        <div className="space-y-3">
          <div>
            <label className="flex items-center space-x-2 text-sm font-medium text-slate-300 mb-2">
              <Search className="w-4 h-4" />
              <span>Поисковый запрос:</span>
            </label>
            <input
              type="text"
              value={dataQuery}
              onChange={(e) => setDataQuery(e.target.value)}
              placeholder="Введите тему для поиска научных статей..."
              className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
            />
          </div>

          <div>
            <label className="flex items-center space-x-2 text-sm font-medium text-slate-300 mb-2">
              <BarChart3 className="w-4 h-4" />
              <span>Количество статей:</span>
            </label>
            <select
              value={dataLimit}
              onChange={(e) => setDataLimit(Number(e.target.value))}
              className="w-full px-4 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 cursor-pointer"
            >
              <option value={20}>20 статей (быстро)</option>
              <option value={50}>50 статей (рекомендуется)</option>
              <option value={100}>100 статей (качественно)</option>
              <option value={200}>200 статей (максимум)</option>
            </select>
          </div>

          <button
            className="w-full flex items-center justify-center space-x-2 px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white font-medium rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleLoadData}
            disabled={loading}
          >
            {loading ? (
              <>
                <Loader className="w-4 h-4 animate-spin" />
                <span>Загрузка данных...</span>
              </>
            ) : (
              <>
                <Download className="w-4 h-4" />
                <span>Загрузить новые данные</span>
              </>
            )}
          </button>
        </div>

        {/* Quick Query Buttons */}
        <div className="space-y-3">
          <div className="flex items-center space-x-2 text-sm font-medium text-slate-300">
            <Search className="w-4 h-4" />
            <span>Быстрые запросы:</span>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {quickQueries.map((item) => (
              <button
                key={item.name}
                onClick={() => setDataQuery(item.query)}
                className={`px-3 py-2 text-sm font-medium rounded-lg transition-all duration-200 ${
                  dataQuery === item.query
                    ? "bg-blue-500 text-white border border-blue-500"
                    : "bg-slate-700/50 text-slate-300 border border-slate-600 hover:bg-slate-600/50 hover:text-white"
                }`}
              >
                {item.name}
              </button>
            ))}
          </div>
        </div>

        {/* Data Source Info */}
        <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-xl">
          <div className="flex items-center space-x-2 mb-2">
            <Globe className="w-4 h-4 text-emerald-400" />
            <span className="font-semibold text-emerald-400">
              Источник данных: OpenAlex API
            </span>
          </div>
          <p className="text-sm text-emerald-300">
            Крупнейшая открытая база научных публикаций с миллионами статей
          </p>
        </div>
      </div>

      {/* Advanced Settings */}
      <div className="space-y-3">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center space-x-2 text-blue-400 hover:text-blue-300 transition-colors duration-200"
        >
          {showAdvanced ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
          <span className="font-medium">Расширенные настройки</span>
        </button>

        {showAdvanced && (
          <div className="p-4 bg-slate-700/30 border border-slate-600/50 rounded-xl">
            <div className="space-y-2 text-sm text-slate-300">
              <div className="flex items-center space-x-2">
                <Settings className="w-3 h-3" />
                <span>Модель: SGDClassifier (Stochastic Gradient Descent)</span>
              </div>
              <div className="flex items-center space-x-2">
                <Settings className="w-3 h-3" />
                <span>Векторизация: TF-IDF (1000 признаков)</span>
              </div>
              <div className="flex items-center space-x-2">
                <Settings className="w-3 h-3" />
                <span>Классы: 3 уровня (High/Medium/Low citation)</span>
              </div>
              <div className="flex items-center space-x-2">
                <Settings className="w-3 h-3" />
                <span>Обновление: каждые 10 секунд</span>
              </div>
              <div className="flex items-center space-x-2">
                <Settings className="w-3 h-3" />
                <span>Сохранение: лучшие модели автоматически</span>
              </div>
              <div className="flex items-center space-x-2">
                <Settings className="w-3 h-3" />
                <span>Эмбеддинги: SentenceTransformer</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Recommendation */}
      {recommendation && (
        <div
          className={`p-4 rounded-xl border ${
            recommendation.type === "success"
              ? "bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-green-500/30"
              : recommendation.type === "warning"
              ? "bg-gradient-to-br from-yellow-500/10 to-orange-500/10 border-yellow-500/30"
              : "bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border-blue-500/30"
          }`}
        >
          <div
            className={`flex items-center space-x-2 font-semibold mb-2 ${
              recommendation.type === "success"
                ? "text-green-400"
                : recommendation.type === "warning"
                ? "text-yellow-400"
                : "text-blue-400"
            }`}
          >
            {recommendation.type === "success" ? (
              <CheckCircle className="w-4 h-4" />
            ) : recommendation.type === "warning" ? (
              <AlertTriangle className="w-4 h-4" />
            ) : (
              <Info className="w-4 h-4" />
            )}
            <span>{recommendation.title}</span>
          </div>
          <p
            className={`text-sm ${
              recommendation.type === "success"
                ? "text-green-300"
                : recommendation.type === "warning"
                ? "text-yellow-300"
                : "text-blue-300"
            }`}
          >
            {recommendation.message}
          </p>
        </div>
      )}
    </div>
  );
};

export default ControlPanel;
