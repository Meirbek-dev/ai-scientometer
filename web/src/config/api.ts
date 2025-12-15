// API Configuration
// In production, VITE_API_URL is set to '/api' which nginx proxies to the backend
// In development, it defaults to localhost:8000

const getApiBaseUrl = (): string => {
  const envUrl = import.meta.env.VITE_API_URL;
  
  // If env var is set, use it
  if (envUrl) {
    // If it's a relative path like '/api', convert to full URL
    if (envUrl.startsWith('/')) {
      return `${window.location.origin}${envUrl}`;
    }
    return envUrl;
  }
  
  // Default to localhost for development
  return 'http://localhost:8000';
};

export const API_BASE_URL = getApiBaseUrl();

// API endpoints
export const API_ENDPOINTS = {
  // Chat
  chat: `${API_BASE_URL}/api/v1/chat`,
  chatSuggestions: `${API_BASE_URL}/api/v1/chat/suggestions`,
  chatEvaluate: `${API_BASE_URL}/api/v1/chat/evaluate`,

  // Training
  trainingStatus: `${API_BASE_URL}/api/v1/training/status`,
  trainingStart: `${API_BASE_URL}/api/v1/training/start`,
  trainingStop: `${API_BASE_URL}/api/v1/training/stop`,
  trainingModels: `${API_BASE_URL}/api/v1/training/models`,
  trainingMetrics: `${API_BASE_URL}/api/v1/training/metrics`,

  // Data
  dataStats: `${API_BASE_URL}/api/v1/data/stats`,
  dataLoad: `${API_BASE_URL}/api/v1/data/load`,

  // Analysis
  analysisSearch: `${API_BASE_URL}/api/v1/analysis/search`,

  // Health & Docs
  health: `${API_BASE_URL}/health`,
  docs: `${API_BASE_URL}/docs`,
};

export default API_BASE_URL;
