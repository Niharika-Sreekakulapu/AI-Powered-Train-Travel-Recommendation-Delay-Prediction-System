import axios from 'axios';
import { TrainPrediction, RecommendationResponse, WeatherData, Train } from '../types';

const API_BASE_URL = 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
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
  predictDelay: async (data: {
    source: string;
    destination: string;
    travel_date: string;
    train_id?: string;
  }): Promise<TrainPrediction> => {
    const response = await api.post('/predict', data);
    return response.data;
  },

  // Get train recommendations
  getRecommendations: async (data: {
    source: string;
    destination: string;
    travel_date: string;
    preference: 'fastest' | 'cheapest' | 'most_reliable';
  }): Promise<RecommendationResponse> => {
    const response = await api.post('/recommend', data);
    return response.data;
  },

  // Get weather data
  getWeather: async (city: string): Promise<WeatherData> => {
    const response = await api.get(`/weather?city=${city}`);
    return response.data;
  },

  // Get available trains
  getTrains: async (source?: string, destination?: string): Promise<{ trains: Train[]; total: number }> => {
    const params = new URLSearchParams();
    if (source) params.append('source', source);
    if (destination) params.append('destination', destination);
    
    const response = await api.get(`/trains?${params.toString()}`);
    return response.data;
  },

  // Get train by ID (Feature 2)
  getTrainById: async (trainId: string): Promise<any> => {
    const response = await api.get(`/train/${trainId}`);
    return response.data;
  },

  // Get available running days for a route
  getAvailableDays: async (source: string, destination: string): Promise<{ day: number; name: string }[]> => {
    const params = new URLSearchParams();
    params.append('source', source);
    params.append('destination', destination);
    const response = await api.get(`/available_days?${params.toString()}`);
    return response.data?.available_days || [];
  },

  // Get analytics for a route (monthly trends, reliability, seasonal)
  getAnalytics: async (source: string, destination: string): Promise<any> => {
    const params = new URLSearchParams();
    params.append('source', source);
    params.append('destination', destination);
    const response = await api.get(`/analytics/route_trends?${params.toString()}`);
    return response.data || {};
  },

  // Health check
  healthCheck: async (): Promise<{ status: string; model_loaded: boolean; timestamp: string }> => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default apiService;