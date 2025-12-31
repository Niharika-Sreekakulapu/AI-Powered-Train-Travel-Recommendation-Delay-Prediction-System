import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  // Increased timeout to 30s to allow longer server-side processing (model/data loads or complex searches)
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const apiService = {
  // Predict train delay
  predictDelay: async (data) => {
    const response = await api.post('/predict', data);
    return response.data;
  },

  // Get train recommendations
  getRecommendations: async (data) => {
    const response = await api.post('/recommend', data);
    return response.data;
  },

  // Get weather data
  getWeather: async (city) => {
    const response = await api.get(`/weather?city=${city}`);
    return response.data;
  },

  // Get available trains
  getTrains: async (source, destination) => {
    const params = new URLSearchParams();
    if (source) params.append('source', source);
    if (destination) params.append('destination', destination);
    
    const response = await api.get(`/trains?${params.toString()}`);
    return response.data;
  },

  // Get train by ID (Feature 2)
  getTrainById: async (trainId) => {
    const response = await api.get(`/train/${trainId}`);
    return response.data;
  },

  // Get available running days for a route
  getAvailableDays: async (source, destination) => {
    const params = new URLSearchParams();
    params.append('source', source);
    params.append('destination', destination);
    const response = await api.get(`/available_days?${params.toString()}`);
    return response.data?.available_days || [];
  },

  // Get analytics for a route (monthly trends, reliability, seasonal)
  getAnalytics: async (source, destination) => {
    const params = new URLSearchParams();
    params.append('source', source);
    params.append('destination', destination);
    const response = await api.get(`/analytics/route_trends?${params.toString()}`);
    return response.data || {};
  },

  // Historical propagation/backtest
  propagateHistorical: async (data) => {
    const response = await api.post('/predict/propagate/historical', data, { timeout: 60000 });
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default apiService;
