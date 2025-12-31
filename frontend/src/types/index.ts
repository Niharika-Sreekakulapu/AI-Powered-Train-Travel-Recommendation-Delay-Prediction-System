export interface DelayFactor {
  factor: string;
  impact: string;
  description: string;
}

export interface PredictionRecommendation {
  should_travel: boolean;
  alternative_suggestion: string;
  risk_level: string;
}

export interface TrainPredictionItem {
  train_id: string;
  train_name: string;
  departure_time?: string | null;
  arrival_time?: string | null;
  predicted_delay_min: number;
  delay_probability: number;
  confidence: string;
  delay_severity: string;
  estimated_delay_percentage?: number;
  distance_km: number;
  price: number;
  is_best?: boolean;
  reason?: string;
  delay_factors?: DelayFactor[];
  recommendation?: PredictionRecommendation;
  recommendation_reason?: string;
  // Imputation / uncertainty fields (optional)
  rr_mean_pred?: number;
  rr_mean_lo?: number;
  rr_mean_hi?: number;
  rr_std_pred?: number;
  rr_std_lo?: number;
  rr_std_hi?: number;
  rr_imputation_flag_conservative?: boolean;
  rr_imputation_flag_conformal?: boolean;
  rr_imputation_flag_final?: boolean;
  // Risk fields added by backend
  risk?: {
    risk_score: number; // 0-100
    confidence: 'High' | 'Medium' | 'Low' | string;
    advice: string;
    breakdown?: { [k: string]: number | string };
  };
  // Attribution fields (Delay Reason Attribution Dashboard)
  feature_contributions?: { [k: string]: number };
  top_contributors?: string[];
}

export interface TrainPrediction {
  has_direct_trains?: boolean;
  connecting_route?: ConnectingRoute;
  message?: string;
  note?: string;
  total_delay?: number;
  total_price?: number;
  total_distance?: number;
  all_trains?: TrainPredictionItem[];
  best_route?: TrainPredictionItem;
  total_trains?: number;
  prediction?: {
    train_id: string;
    train_name: string;
    departure_time?: string | null;
    arrival_time?: string | null;
    predicted_delay_min: number;
    delay_probability: number;
    confidence: string;
    delay_severity: string;
    estimated_delay_percentage: number;
    reason: string;
    delay_factors: DelayFactor[];
    route_info: RouteInfo & {
      source_name?: string;
      destination_name?: string;
    };
    weather: WeatherData;
    recommendation: PredictionRecommendation;
    // Imputation fields for the top-level prediction
    rr_mean_pred?: number;
    rr_mean_lo?: number;
    rr_mean_hi?: number;
    rr_std_pred?: number;
    rr_std_lo?: number;
    rr_std_hi?: number;
    rr_imputation_flag_conservative?: boolean;
    rr_imputation_flag_conformal?: boolean;
    rr_imputation_flag_final?: boolean;
    // Top-level risk object when prediction is nested
    risk?: {
      risk_score: number;
      confidence: 'High' | 'Medium' | 'Low' | string;
      advice: string;
      breakdown?: { [k: string]: number | string };
    };
    // Attribution fields
    feature_contributions?: { [k: string]: number };
    top_contributors?: string[];
  };
  // Direct fields (when not nested)
  train_id?: string;
  train_name?: string;
  predicted_delay_min?: number;
  delay_probability?: number;
  confidence?: string;
  delay_severity?: string;
  estimated_delay_percentage?: number;
  reason?: string;
  delay_factors?: DelayFactor[];
  route_info?: RouteInfo & {
    source_name?: string;
    destination_name?: string;
  };
  weather?: WeatherData;
  recommendation?: PredictionRecommendation;
}

export interface WeatherData {
  temp: number;
  condition: string;
  humidity: number;
  wind_speed: number;
}

export interface RouteInfo {
  source: string;
  destination: string;
  distance_km: number;
  day_of_week: number;
  month: number;
  season: string;
}

export interface TrainRecommendation {
  train_id: string;
  train_name: string;
  train_type?: string; // Express, Superfast, Premium, Special
  departure_time?: string | null;
  arrival_time?: string | null;
  price: number;
  predicted_delay_min: number;
  reliability_score: number;
  speed_kmph: number;
  distance_km: number;
  estimated_journey_hours?: number;
  estimated_journey_minutes?: number;
  value_score?: number;
  recommendation_reason?: string;
  delay_category?: string; // On Time, Minor Delay, Significant Delay
  tags?: string[]; // Fastest, Cheapest, Most Reliable, Best Value
  // Risk object propagated from backend (0-100 score, confidence, advice)
  risk?: {
    risk_score: number;
    confidence: 'High' | 'Medium' | 'Low' | string;
    advice: string;
    breakdown?: { [k: string]: number | string };
    // Attribution fields
    feature_contributions?: { [k: string]: number };
    top_contributors?: string[];
  };
  rank: number;
  is_best: boolean;
}

export interface ConnectingRoute {
  connecting_station: string;
  train1: {
    train_id: string;
    train_name: string;
    source: string;
    destination: string;
    distance_km: number;
    price: number;
    price_source?: string;
    predicted_delay_min: number;
    departure_time?: string | null;
    arrival_time?: string | null;
  };
  train2: {
    train_id: string;
    train_name: string;
    source: string;
    destination: string;
    distance_km: number;
    price: number;
    price_source?: string;
    predicted_delay_min: number;
    departure_time?: string | null;
    arrival_time?: string | null;
  };
  total_distance: number;
  total_price: number;
  total_delay: number;
  layover_time: number;
}

export interface RecommendationSummary {
  fastest?: TrainRecommendation;
  cheapest?: TrainRecommendation;
  most_reliable?: TrainRecommendation;
  best_value?: TrainRecommendation;
  total_options: number;
}

export interface RouteInfoSummary {
  source: string;
  destination: string;
  distance_range: {
    min: number;
    max: number;
  };
  price_range: {
    min: number;
    max: number;
  };
}

export interface RecommendationResponse {
  has_direct_trains?: boolean;
  recommendations: TrainRecommendation[];
  best_route?: TrainRecommendation;
  all_trains?: TrainRecommendation[];
  recommendation_summary?: RecommendationSummary; // New: Multiple categories
  route_info?: RouteInfoSummary; // New: Route statistics
  connecting_route?: ConnectingRoute;
  message?: string;
  preference: string;
  weather: WeatherData;
  total_trains: number;
}

export interface Train {
  train_id: string;
  train_name: string;
  source: string;
  destination: string;
  distance_km: number;
  price: number;
}

export interface SearchFormData {
  source: string;
  destination: string;
  travel_date: string;
  preference: 'fastest' | 'cheapest' | 'most_reliable';
  train_id?: string;
}

export interface ApiResponse<T> {
  data?: T;
  error?: string;
  status: number;
}