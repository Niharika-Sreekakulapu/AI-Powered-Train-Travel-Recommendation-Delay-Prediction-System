import React from 'react';
import { RecommendationResponse } from '../types';
import { Train, Clock, DollarSign, Shield, Zap, Star, TrendingUp, ArrowRight, Timer, Award, Tag } from 'lucide-react';
import RiskBreakdownModal from './RiskBreakdownModal';
import DelayContributorsModal from './DelayContributorsModal';
import { formatStation } from '../utils/stationUtils';

interface RecommendationsListProps {
  recommendations: RecommendationResponse;
}

const RecommendationsList: React.FC<RecommendationsListProps> = ({ recommendations }) => {
  const [riskModalOpen, setRiskModalOpen] = React.useState(false);
  const [riskModalData, setRiskModalData] = React.useState<any>(null);
  // Delay contributors modal state
  const [delayModalOpen, setDelayModalOpen] = React.useState(false);
  const [delayContribData, setDelayContribData] = React.useState<any>(null);

  const getPreferenceIcon = (preference: string) => {
    switch (preference) {
      case 'fastest':
        return <Zap className="w-5 h-5 text-yellow-500" />;
      case 'cheapest':
        return <DollarSign className="w-5 h-5 text-green-500" />;
      case 'most_reliable':
        return <Shield className="w-5 h-5 text-blue-500" />;
      default:
        return <TrendingUp className="w-5 h-5 text-gray-500" />;
    }
  };

  const getPreferenceColor = (preference: string) => {
    switch (preference) {
      case 'fastest':
        return 'from-yellow-400 to-orange-500';
      case 'cheapest':
        return 'from-green-400 to-emerald-500';
      case 'most_reliable':
        return 'from-blue-400 to-indigo-500';
      default:
        return 'from-gray-400 to-gray-500';
    }
  };

  const getDelayColor = (delay: number) => {
    if (delay <= 15) return 'text-green-600 bg-green-100';
    if (delay <= 30) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getReliabilityColor = (score: number) => {
    if (score >= 80) return 'text-green-600 bg-green-100';
    if (score >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const safeToFixed = (value: any, decimals: number = 1, fallback: string = '0.0') => {
    try {
      if (value === undefined || value === null) return fallback;
      const num = Number(value);
      if (isNaN(num) || !isFinite(num)) return fallback;
      return num.toFixed(decimals);
    } catch (e) {
      return fallback;
    }
  };

  // Risk helpers
  const getRiskColor = (score: number) => {
    if (score <= 40) return 'bg-green-100 text-green-800';
    if (score <= 70) return 'bg-yellow-100 text-yellow-800';
    return 'bg-red-100 text-red-800';
  };

  const renderRiskBadge = (risk: any) => {
    if (!risk) return null;
    const score = Number(risk.risk_score || 0);
    const title = `${risk.advice} • Confidence: ${risk.confidence}`;
    return (
      <div className="inline-flex flex-col items-end ml-3" title={title}>
        <div className={`px-3 py-1 rounded-full text-sm font-semibold ${getRiskColor(score)}`}>
          {score}% • {risk.confidence}
        </div>
        <div className="text-xs text-gray-500 mt-1">{risk.advice}</div>
      </div>
    );
  };

  const renderImputationInfo = (item: any) => {
    if (!item) return null;
    const hasInterval = item.rr_mean_lo !== undefined && item.rr_mean_hi !== undefined && item.rr_mean_pred !== undefined;
    const flagged = item.rr_imputation_flag_final || item.rr_imputation_flag_conservative || item.rr_imputation_flag_conformal;
    return (
      <div className="mt-2 text-xs text-gray-500">
        {hasInterval && (
          <div>Imputed rr_mean: {safeToFixed(item.rr_mean_pred,1)} min ({safeToFixed(item.rr_mean_lo,1)}–{safeToFixed(item.rr_mean_hi,1)})</div>
        )}
        {flagged && (
          <div className="inline-block mt-1 px-2 py-1 rounded text-xs font-semibold text-white bg-red-600" title={
            item.rr_imputation_flag_final ? 'Final imputation flag (conservative OR conformal)' : 'High uncertainty imputation'
          }>
            ⚠ High Uncertainty
          </div>
        )}
      </div>
    );
  };

  // Feature 5: Show connecting trains if no direct trains
  if (!recommendations.has_direct_trains && recommendations.connecting_route) {
    const route = recommendations.connecting_route;
    return (
      <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 animate-fade-in">
        <div className="bg-yellow-50 border-2 border-yellow-400 rounded-xl p-6 mb-6">
          <h2 className="text-2xl font-bold text-yellow-800 mb-2">
            🔄 Connecting Route Required
          </h2>
          <p className="text-yellow-700">
            No direct trains found. Here's the shortest connecting route:
          </p>
        </div>

        <div className="space-y-6">
          {/* Train 1: Source to Connecting Station */}
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-xl border-2 border-blue-300">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-xl font-bold text-gray-900">{route.train1.train_name}</h3>
                <p className="text-sm text-gray-600">Train ID: {route.train1.train_id}</p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-gray-900">₹{route.train1.price.toLocaleString()}</div>
                <div className="text-sm text-gray-600">Fare</div>
              </div>
            </div>
            <div className="flex items-center space-x-4 text-lg font-semibold">
              <span className="text-blue-700">{formatStation(route.train1.source)}</span>
              <ArrowRight className="w-6 h-6 text-blue-500" />
              <span className="text-blue-700">{formatStation(route.connecting_station)}</span>
            </div>
            <div className="grid grid-cols-3 gap-4 mt-4">
              <div className="text-center p-3 bg-white rounded-lg">
                <div className="text-lg font-bold text-gray-900">{route.train1.distance_km} km</div>
                <div className="text-xs text-gray-600">Distance</div>
              </div>
              <div className="text-center p-3 bg-white rounded-lg">
                <div className={`text-lg font-bold ${getDelayColor(route.train1.predicted_delay_min).split(' ')[0]}`}>
                  {route.train1.predicted_delay_min} min
                </div>
                <div className="text-xs text-gray-600">Delay</div>
              </div>
              <div className="text-center p-3 bg-white rounded-lg">
                <div className="text-lg font-bold text-gray-900">{route.layover_time} min</div>
                <div className="text-xs text-gray-600">Layover</div>
              </div>
            </div>
          </div>

          {/* Connecting Station Info */}
          <div className="text-center py-4">
            <div className="inline-block px-6 py-3 bg-purple-100 rounded-full">
              <p className="font-semibold text-purple-800">Change at: {formatStation(route.connecting_station)}</p>
              <p className="text-sm text-purple-600">Layover: {route.layover_time} minutes</p>
            </div>
          </div>

          {/* Train 2: Connecting Station to Destination */}
          <div className="bg-gradient-to-r from-green-50 to-emerald-50 p-6 rounded-xl border-2 border-green-300">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-xl font-bold text-gray-900">{route.train2.train_name}</h3>
                <p className="text-sm text-gray-600">Train ID: {route.train2.train_id}</p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-gray-900">₹{route.train2.price.toLocaleString()}</div>
                <div className="text-sm text-gray-600">Fare</div>
              </div>
            </div>
            <div className="flex items-center space-x-4 text-lg font-semibold">
              <span className="text-green-700">{formatStation(route.connecting_station)}</span>
              <ArrowRight className="w-6 h-6 text-green-500" />
              <span className="text-green-700">{formatStation(route.train2.destination)}</span>
            </div>
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="text-center p-3 bg-white rounded-lg">
                <div className="text-lg font-bold text-gray-900">{route.train2.distance_km} km</div>
                <div className="text-xs text-gray-600">Distance</div>
              </div>
              <div className="text-center p-3 bg-white rounded-lg">
                <div className={`text-lg font-bold ${getDelayColor(route.train2.predicted_delay_min).split(' ')[0]}`}>
                  {route.train2.predicted_delay_min} min
                </div>
                <div className="text-xs text-gray-600">Delay</div>
              </div>
            </div>
          </div>

          {/* Summary */}
          <div className="bg-gray-50 rounded-xl p-6">
            <h4 className="font-bold text-gray-900 mb-4">Journey Summary</h4>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">{route.total_distance} km</div>
                <div className="text-sm text-gray-600">Total Distance</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">₹{route.total_price.toLocaleString()}</div>
                <div className="text-sm text-gray-600">Total Fare</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">{route.total_delay} min</div>
                <div className="text-sm text-gray-600">Total Delay</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Feature 3: Show best route prominently + all trains
  return (
    <>
    <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 animate-fade-in">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Train Recommendations</h2>
        <div className={`px-4 py-2 rounded-full bg-gradient-to-r ${getPreferenceColor(recommendations.preference)} text-white flex items-center space-x-2`}>
          {getPreferenceIcon(recommendations.preference)}
          <span className="font-semibold capitalize">{recommendations.preference} First</span>
        </div>
      </div>

      <div className="text-sm text-gray-600 mb-6">
        Found {recommendations.total_trains} trains for your route. Showing <span className="font-bold text-primary-600">Top 3 Recommendations</span> sorted by {recommendations.preference} preference
      </div>

      {/* Recommendation Summary - Different Categories */}
      {recommendations.recommendation_summary && recommendations.recommendation_summary.total_options > 1 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          {recommendations.recommendation_summary.fastest && (
            <div className="bg-yellow-50 border-2 border-yellow-300 rounded-xl p-4">
              <div className="flex items-center space-x-2 mb-2">
                <Zap className="w-5 h-5 text-yellow-600" />
                <h4 className="font-bold text-yellow-800">Fastest</h4>
              </div>
              <p className="text-sm font-semibold text-gray-900">{recommendations.recommendation_summary.fastest.train_name}</p>
              <p className="text-xs text-gray-600">{recommendations.recommendation_summary.fastest.speed_kmph.toFixed(1)} km/h</p>
            </div>
          )}
          {recommendations.recommendation_summary.cheapest && (
            <div className="bg-green-50 border-2 border-green-300 rounded-xl p-4">
              <div className="flex items-center space-x-2 mb-2">
                <DollarSign className="w-5 h-5 text-green-600" />
                <h4 className="font-bold text-green-800">Cheapest</h4>
              </div>
              <p className="text-sm font-semibold text-gray-900">{recommendations.recommendation_summary.cheapest.train_name}</p>
              <p className="text-xs text-gray-600">₹{recommendations.recommendation_summary.cheapest.price.toLocaleString()}</p>
            </div>
          )}
          {recommendations.recommendation_summary.most_reliable && (
            <div className="bg-blue-50 border-2 border-blue-300 rounded-xl p-4">
              <div className="flex items-center space-x-2 mb-2">
                <Shield className="w-5 h-5 text-blue-600" />
                <h4 className="font-bold text-blue-800">Most Reliable</h4>
              </div>
              <p className="text-sm font-semibold text-gray-900">{recommendations.recommendation_summary.most_reliable.train_name}</p>
              <p className="text-xs text-gray-600">{recommendations.recommendation_summary.most_reliable.reliability_score.toFixed(0)}% reliable</p>
            </div>
          )}
          {recommendations.recommendation_summary.best_value && (
            <div className="bg-purple-50 border-2 border-purple-300 rounded-xl p-4">
              <div className="flex items-center space-x-2 mb-2">
                <Award className="w-5 h-5 text-purple-600" />
                <h4 className="font-bold text-purple-800">Best Value</h4>
              </div>
              <p className="text-sm font-semibold text-gray-900">{recommendations.recommendation_summary.best_value.train_name}</p>
              <p className="text-xs text-gray-600">Score: {recommendations.recommendation_summary.best_value.value_score?.toFixed(0)}</p>
            </div>
          )}
        </div>
      )}

      {/* Route Info Summary */}
      {recommendations.route_info && (
        <div className="bg-gray-50 rounded-xl p-4 mb-6">
          <h4 className="font-semibold text-gray-900 mb-2">Route Information</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Distance: </span>
              <span className="font-semibold text-gray-900">
                {recommendations.route_info.distance_range.min === recommendations.route_info.distance_range.max
                  ? `${recommendations.route_info.distance_range.min} km`
                  : `${recommendations.route_info.distance_range.min}-${recommendations.route_info.distance_range.max} km`}
              </span>
            </div>
            <div>
              <span className="text-gray-600">Price Range: </span>
              <span className="font-semibold text-gray-900">
                ₹{recommendations.route_info.price_range.min.toLocaleString()} - ₹{recommendations.route_info.price_range.max.toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-gray-600">From: </span>
              <span className="font-semibold text-gray-900">{formatStation(recommendations.route_info.source)}</span>
            </div>
            <div>
              <span className="text-gray-600">To: </span>
              <span className="font-semibold text-gray-900">{formatStation(recommendations.route_info.destination)}</span>
            </div>
          </div>
        </div>
      )}

      {/* Feature 3: Best Route Highlight */}
      {recommendations.best_route && (
        <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-6 mb-6 border-2 border-green-500 shadow-lg">
          <div className="flex items-center space-x-2 mb-4">
            <Star className="w-6 h-6 text-yellow-500" />
            <h3 className="text-xl font-bold text-green-800">⭐ Best Recommended Route</h3>
          </div>
          <div className="bg-white rounded-lg p-4">
            {renderImputationInfo(recommendations.best_route)}
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-lg font-semibold text-gray-900">{recommendations.best_route.train_name}</h4>
                <p className="text-sm text-gray-600">Train ID: {recommendations.best_route.train_id}</p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-gray-900">₹{(recommendations.best_route.price || 0).toLocaleString()}</div>
                <div className="text-sm text-gray-600">Fare</div>
                {recommendations.best_route.risk && (
                  <div className="mt-2">
                    <button
                      onClick={() => { setRiskModalData(recommendations.best_route?.risk || null); setRiskModalOpen(true); }}
                      className="focus:outline-none"
                      aria-label="Open risk details"
                    >
                      {renderRiskBadge(recommendations.best_route.risk)}
                    </button>
                  </div>
                )}
                {(((recommendations.best_route as any).feature_contributions) || recommendations.best_route.risk?.feature_contributions) && (
                  <div className="mt-2">
                    <button
                      className="text-xs text-indigo-600 hover:underline"
                      onClick={() => { setDelayContribData({ contributions: ((recommendations.best_route as any)?.feature_contributions) || recommendations.best_route?.risk?.feature_contributions, top_contributors: ((recommendations.best_route as any)?.top_contributors) || recommendations.best_route?.risk?.top_contributors }); setDelayModalOpen(true); }}
                    >
                      View contributors
                    </button>
                  </div>
                )}
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-7 gap-4">
              {/* Departure Time - Always Show */}
              <div className="text-center p-3 bg-blue-50 rounded-lg border border-blue-200">
                <div className="flex items-center justify-center mb-1">
                  <Clock className="w-4 h-4 text-blue-500 mr-1" />
                  <div className="text-lg font-bold text-blue-600">
                    {recommendations.best_route.departure_time || '--'}
                  </div>
                </div>
                <div className="text-xs text-gray-600">Departure</div>
              </div>
              
              {/* Arrival Time - Always Show */}
              <div className="text-center p-3 bg-green-50 rounded-lg border border-green-200">
                <div className="flex items-center justify-center mb-1">
                  <Clock className="w-4 h-4 text-green-500 mr-1" />
                  <div className="text-lg font-bold text-green-600">
                    {recommendations.best_route.arrival_time || '--'}
                  </div>
                </div>
                <div className="text-xs text-gray-600">Arrival</div>
              </div>
              {/* Speed */}
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center mb-1">
                  <Zap className="w-4 h-4 text-yellow-500 mr-1" />
                  <div className="text-lg font-bold text-gray-900">{(recommendations.best_route.speed_kmph || 0).toFixed(1)}</div>
                </div>
                <div className="text-xs text-gray-600">km/h</div>
              </div>
              
              {/* Delay */}
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center mb-1">
                  <Clock className="w-4 h-4 text-gray-500 mr-1" />
                  <div className={`text-lg font-bold ${getDelayColor(recommendations.best_route.predicted_delay_min).split(' ')[0]}`}>
                    {recommendations.best_route.predicted_delay_min.toFixed(1)}
                  </div>
                </div>
                <div className="text-xs text-gray-600">min delay</div>
              </div>
              
              {/* Reliability */}
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center mb-1">
                  <Shield className="w-4 h-4 text-blue-500 mr-1" />
                  <div className={`text-lg font-bold ${getReliabilityColor(recommendations.best_route.reliability_score).split(' ')[0]}`}>
                    {recommendations.best_route.reliability_score.toFixed(0)}
                  </div>
                </div>
                <div className="text-xs text-gray-600">reliability</div>
              </div>
              
              {/* Journey Time */}
              {recommendations.best_route.estimated_journey_hours && (
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-center mb-1">
                    <Timer className="w-4 h-4 text-purple-500 mr-1" />
                    <div className="text-lg font-bold text-gray-900">
                      {Math.floor(recommendations.best_route.estimated_journey_hours)}h {recommendations.best_route.estimated_journey_minutes ? recommendations.best_route.estimated_journey_minutes % 60 : 0}m
                    </div>
                  </div>
                  <div className="text-xs text-gray-600">journey</div>
                </div>
              )}
              
              {/* Distance */}
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center mb-1">
                  <Train className="w-4 h-4 text-gray-500 mr-1" />
                  <div className="text-lg font-bold text-gray-900">{recommendations.best_route.distance_km || 0}</div>
                </div>
                <div className="text-xs text-gray-600">km</div>
              </div>
            </div>
            
            {/* Best Route Tags */}
            {recommendations.best_route.tags && recommendations.best_route.tags.length > 0 && (
              <div className="flex flex-wrap gap-2 mt-4">
                {recommendations.best_route.tags.map((tag, idx) => (
                  <span key={idx} className="px-3 py-1 rounded-full text-xs font-semibold bg-yellow-100 text-yellow-800 border border-yellow-300">
                    {tag}
                  </span>
                ))}
              </div>
            )}
            
            {/* Best Route Reason */}
            {recommendations.best_route.recommendation_reason && (
              <div className="mt-3 p-3 bg-green-50 rounded-lg border border-green-200">
                <p className="text-sm text-green-800">
                  <span className="font-semibold">💡 Why this train:</span> {recommendations.best_route.recommendation_reason}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Top 3 Recommendations List */}
      <div className="mb-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Top 3 Recommendations
          {recommendations.recommendations && recommendations.recommendations.length > 0 && (
            <span className="ml-2 text-sm font-normal text-gray-500">
              ({recommendations.recommendations.length} of {recommendations.total_trains} trains)
            </span>
          )}
        </h3>
      </div>

      <div className="space-y-4">
        {recommendations.recommendations && recommendations.recommendations.length > 0 ? (
          recommendations.recommendations.slice(0, 3).map((train, index) => (
          <div
            key={`${train.train_id}-${index}`}
            className={`p-6 rounded-xl border-2 transition-all duration-200 hover:shadow-lg ${
              train.is_best
                ? 'border-primary-500 bg-primary-50 shadow-lg'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                    train.is_best ? 'bg-primary-500' : 'bg-gray-400'
                  }`}>
                    {train.rank}
                  </div>
                  {train.is_best && (
                    <div className="flex items-center text-primary-600">
                      <Star className="w-4 h-4 mr-1" />
                      <span className="text-sm font-semibold">Best Option</span>
                    </div>
                  )}
                </div>
                <div>
                  <div className="flex items-center space-x-2">
                    <h3 className="text-lg font-semibold text-gray-900">{train.train_name}</h3>
                    {train.train_type && (
                      <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs font-semibold rounded-full">
                        {train.train_type}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600">Train ID: {train.train_id}</p>
                  {train.recommendation_reason && (
                    <p className="text-xs text-green-600 mt-1">💡 {train.recommendation_reason}</p>
                  )}
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-gray-900">₹{(train.price || 0).toLocaleString()}</div>
                <div className="text-sm text-gray-600">Fare</div>
                {train.risk && (
                  <div className="mt-2">
                    <button
                      onClick={() => { setRiskModalData(train.risk); setRiskModalOpen(true); }}
                      className="focus:outline-none"
                      aria-label="Open risk details"
                    >
                      {renderRiskBadge(train.risk)}
                    </button>
                  </div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-7 gap-4">
              {/* Departure Time - Always Show */}
              <div className="text-center p-3 bg-blue-50 rounded-lg border border-blue-200">
                <div className="flex items-center justify-center mb-1">
                  <Clock className="w-4 h-4 text-blue-500 mr-1" />
                  <span className="text-lg font-bold text-blue-600">
                    {train.departure_time || '--'}
                  </span>
                </div>
                <div className="text-xs text-gray-600">Departure</div>
              </div>
              
              {/* Arrival Time - Always Show */}
              <div className="text-center p-3 bg-green-50 rounded-lg border border-green-200">
                <div className="flex items-center justify-center mb-1">
                  <Clock className="w-4 h-4 text-green-500 mr-1" />
                  <span className="text-lg font-bold text-green-600">
                    {train.arrival_time || '--'}
                  </span>
                </div>
                <div className="text-xs text-gray-600">Arrival</div>
              </div>
              
              {/* Speed */}
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center mb-1">
                  <Zap className="w-4 h-4 text-yellow-500 mr-1" />
                  <span className="text-lg font-bold text-gray-900">{(train.speed_kmph || 0).toFixed(1)}</span>
                </div>
                <div className="text-xs text-gray-600">km/h</div>
              </div>

              {/* Delay */}
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center mb-1">
                  <Clock className="w-4 h-4 text-gray-500 mr-1" />
                  <span className={`text-lg font-bold ${getDelayColor(train.predicted_delay_min).split(' ')[0]}`}>
                    {train.predicted_delay_min.toFixed(1)}
                  </span>
                </div>
                <div className="text-xs text-gray-600">min delay</div>
                {renderImputationInfo(train)}
              </div>

              {/* Reliability */}
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center mb-1">
                  <Shield className="w-4 h-4 text-blue-500 mr-1" />
                  <span className={`text-lg font-bold ${getReliabilityColor(train.reliability_score).split(' ')[0]}`}>
                    {train.reliability_score.toFixed(0)}
                  </span>
                </div>
                <div className="text-xs text-gray-600">reliability</div>
              </div>

              {/* Journey Time */}
              {train.estimated_journey_hours && (
                <div className="text-center p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-center mb-1">
                    <Timer className="w-4 h-4 text-purple-500 mr-1" />
                    <span className="text-lg font-bold text-gray-900">
                      {Math.floor(train.estimated_journey_hours)}h {train.estimated_journey_minutes ? train.estimated_journey_minutes % 60 : 0}m
                    </span>
                  </div>
                  <div className="text-xs text-gray-600">journey time</div>
                </div>
              )}

              {/* Distance */}
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center mb-1">
                  <Train className="w-4 h-4 text-gray-500 mr-1" />
                  <span className="text-lg font-bold text-gray-900">{train.distance_km || 0}</span>
                </div>
                <div className="text-xs text-gray-600">km</div>
              </div>
            </div>

            {/* Value Score */}
            {train.value_score && (
              <div className="mt-3 p-2 bg-purple-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Award className="w-4 h-4 text-purple-600" />
                    <span className="text-sm font-semibold text-purple-800">Value Score:</span>
                  </div>
                  <span className="text-lg font-bold text-purple-900">{train.value_score.toFixed(0)}</span>
                </div>
              </div>
            )}

            {/* Tags and Badges */}
            <div className="flex flex-wrap gap-2 mt-4">
              {/* Tags */}
              {train.tags && train.tags.length > 0 && (
                <>
                  {train.tags.map((tag, idx) => (
                    <span key={idx} className="px-3 py-1 rounded-full text-xs font-semibold bg-yellow-100 text-yellow-800 border border-yellow-300 flex items-center space-x-1">
                      <Tag className="w-3 h-3" />
                      <span>{tag}</span>
                    </span>
                  ))}
                </>
              )}
              
              {/* Delay Category */}
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${getDelayColor(train.predicted_delay_min)}`}>
                {train.delay_category || (train.predicted_delay_min <= 15 ? 'On Time' : 
                 train.predicted_delay_min <= 30 ? 'Minor Delay' : 'Significant Delay')}
              </span>
              
              {/* Reliability */}
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${getReliabilityColor(train.reliability_score)}`}>
                {train.reliability_score >= 80 ? 'Highly Reliable' : 
                 train.reliability_score >= 60 ? 'Moderately Reliable' : 'Low Reliability'}
              </span>

              {(((train as any).feature_contributions) || train.risk?.feature_contributions) && (
                <button
                  className="ml-2 text-xs text-indigo-600 hover:underline"
                  onClick={() => { setDelayContribData({ contributions: (train as any).feature_contributions || train.risk?.feature_contributions, top_contributors: (train as any).top_contributors || train.risk?.top_contributors }); setDelayModalOpen(true); }}
                >
                  View contributors
                </button>
              )}
            </div>
          </div>
          ))
        ) : (
          <div className="text-center py-12 bg-gray-50 rounded-xl">
            <p className="text-gray-600 text-lg">No trains available for this route.</p>
            <p className="text-gray-500 text-sm mt-2">Please try a different source or destination.</p>
          </div>
        )}
      </div>

      {/* Weather Summary */}
      {recommendations.weather && (
        <div className="mt-6 p-4 bg-blue-50 rounded-xl">
          <h4 className="font-semibold text-blue-900 mb-2">Current Weather</h4>
          <div className="flex items-center space-x-4 text-sm">
            <span className="text-blue-700">
              {recommendations.weather.condition} • {recommendations.weather.temp}°C
            </span>
            <span className="text-blue-600">
              Humidity: {recommendations.weather.humidity}%
            </span>
            <span className="text-blue-600">
              Wind: {recommendations.weather.wind_speed} km/h
            </span>
          </div>
        </div>
      )}
    </div>
    <DelayContributorsModal isOpen={delayModalOpen} onClose={() => setDelayModalOpen(false)} title="Delay Contributors" contributions={delayContribData?.contributions} top_contributors={delayContribData?.top_contributors} />
    <RiskBreakdownModal isOpen={riskModalOpen} onClose={() => setRiskModalOpen(false)} risk={riskModalData} />
    </>
  );
};

export default RecommendationsList;
