import React, { useState } from 'react';
import { TrainPrediction, DelayFactor } from '../types';
import apiService from '../services/api';
import { Clock, AlertTriangle, CheckCircle, Thermometer, Droplets, Wind, ArrowRight, Train, Star } from 'lucide-react';
import RunningDaysModal from './RunningDaysModal';
import RiskBreakdownModal from './RiskBreakdownModal';
import DelayContributorsModal from './DelayContributorsModal';
import { formatStation } from '../utils/stationUtils';

interface PredictionCardProps {
  prediction: TrainPrediction;
}

const PredictionCard: React.FC<PredictionCardProps> = ({ prediction }) => {
  // Helper function to safely format numbers
  const safeToFixed = (value: any, decimals: number = 1, fallback: string = '0.0'): string => {
    try {
      if (value === undefined || value === null) {
        return fallback;
      }
      const num = Number(value);
      if (isNaN(num) || !isFinite(num)) {
        return fallback;
      }
      return num.toFixed(decimals);
    } catch (e) {
      console.warn('Error in safeToFixed:', e, value);
      return fallback;
    }
  };

  // Local state for running days modal (declare unconditionally so hooks are not called conditionally)
  const [runningDaysOpen, setRunningDaysOpen] = React.useState(false);
  const [runningDays, setRunningDays] = React.useState<{ day: number; name: string }[]>([]);
  const [runningDaysLoading, setRunningDaysLoading] = React.useState(false);
  // Local state for risk breakdown modal
  const [riskModalOpen, setRiskModalOpen] = React.useState(false);
  const [riskModalData, setRiskModalData] = React.useState<any>(null);
  // Local state for delay contributors modal
  const [delayModalOpen, setDelayModalOpen] = React.useState(false);
  const [delayContribData, setDelayContribData] = React.useState<any>(null);

  // Safety check: render a lightweight placeholder when no prediction is available
  if (!prediction) {
    return (
      <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
        <p className="text-gray-600">No prediction data available</p>
      </div>
    );
  }

  // Handle connecting trains (no direct trains available)
  if (prediction.has_direct_trains === false && prediction.connecting_route) {
    const route = prediction.connecting_route;
    const getDelayColor = (delay: number) => {
      if (delay <= 15) return 'text-green-600 bg-green-100';
      if (delay <= 30) return 'text-yellow-600 bg-yellow-100';
      return 'text-red-600 bg-red-100';
    };

    return (
      <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 animate-fade-in">
        <div className="bg-yellow-50 border-2 border-yellow-400 rounded-xl p-6 mb-6">
          <h2 className="text-2xl font-bold text-yellow-800 mb-2">
            🔄 Connecting Route Required
          </h2>
          <p className="text-yellow-700">
            {prediction.message || 'No direct trains found. Here\'s the shortest connecting route with delay predictions:'}
          </p>
        </div>

        {prediction.note && (
          <div className="bg-yellow-50 border-l-4 border-yellow-400 rounded p-4 mb-6">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-semibold text-yellow-800 mb-1">⚠️ Note</div>
                <div className="text-sm text-yellow-700">{prediction.note} Please verify train running days before booking or try a different travel date.</div>
              </div>
              <div className="ml-4">
                <CheckCircle className="w-5 h-5 text-yellow-700 mr-2 inline" />
                <button
                  className="inline-flex items-center px-3 py-1 bg-yellow-600 text-white rounded-md shadow-sm hover:bg-yellow-700 text-sm"
                  onClick={async () => {
                    try {
                      const src = prediction.route_info?.source || '';
                      const dst = prediction.route_info?.destination || '';
                      if (!src || !dst) return;
                      // Signal to parent to open modal — use event state
                      setRunningDaysLoading(true);
                      const days = await apiService.getAvailableDays(src, dst);
                      setRunningDays(days || []);
                      setRunningDaysLoading(false);
                      setRunningDaysOpen(true);
                    } catch (e) {
                      setRunningDaysLoading(false);
                      alert('Error fetching available days');
                    }
                  }}
                >
                  {runningDaysLoading ? 'Checking...' : 'Check running days'}
                </button>
              </div>
            </div>
          </div>
        )}

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
                <div className="text-xs text-gray-500 mt-1">Source: {route.train1.price_source || 'estimated'}</div>
              </div>
            </div>
            <div className="flex items-center space-x-4 text-lg font-semibold mb-4">
              <span className="text-blue-700">{formatStation(route.train1.source)}</span>
              <ArrowRight className="w-6 h-6 text-blue-500" />
              <span className="text-blue-700">{formatStation(route.connecting_station)}</span>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-3 bg-white rounded-lg">
                <div className="text-lg font-bold text-gray-900">{route.train1.distance_km} km</div>
                <div className="text-xs text-gray-600">Distance</div>
              </div>
              <div className="text-center p-3 bg-white rounded-lg">
                <div className={`text-lg font-bold ${getDelayColor(route.train1.predicted_delay_min).split(' ')[0]}`}>
                  {route.train1.predicted_delay_min} min
                </div>
                <div className="text-xs text-gray-600">Predicted Delay</div>
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
                <div className="text-xs text-gray-500 mt-1">Source: {route.train2.price_source || 'estimated'}</div>
              </div>
            </div>
            <div className="flex items-center space-x-4 text-lg font-semibold mb-4">
              <span className="text-green-700">{formatStation(route.connecting_station)}</span>
              <ArrowRight className="w-6 h-6 text-green-500" />
              <span className="text-green-700">{formatStation(route.train2.destination)}</span>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 bg-white rounded-lg">
                <div className="text-lg font-bold text-gray-900">{route.train2.distance_km} km</div>
                <div className="text-xs text-gray-600">Distance</div>
              </div>
              <div className="text-center p-3 bg-white rounded-lg">
                <div className={`text-lg font-bold ${getDelayColor(route.train2.predicted_delay_min).split(' ')[0]}`}>
                  {route.train2.predicted_delay_min} min
                </div>
                <div className="text-xs text-gray-600">Predicted Delay</div>
              </div>
            </div>
          </div>

          {/* Summary */}
          <div className="bg-gray-50 rounded-xl p-6">
            <h4 className="font-bold text-gray-900 mb-4 flex items-center">
              <Train className="w-5 h-5 mr-2" />
              Journey Summary
            </h4>
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
                <div className={`text-2xl font-bold ${getDelayColor(route.total_delay).split(' ')[0]}`}>
                  {route.total_delay} min
                </div>
                <div className="text-sm text-gray-600">Total Predicted Delay</div>
              </div>
            </div>
          </div>

          {/* Weather Information */}
          {prediction.weather && (
            <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Thermometer className="w-5 h-5 mr-2 text-blue-500" />
                Weather Conditions
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl mb-1">
                    {prediction.weather.condition === 'Clear' ? '☀️' : 
                     prediction.weather.condition === 'Rainy' ? '🌧️' : 
                     prediction.weather.condition === 'Foggy' ? '🌫️' : '☁️'}
                  </div>
                  <div className="text-sm text-gray-600">{prediction.weather.condition}</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-900">{prediction.weather.temp}°C</div>
                  <div className="text-sm text-gray-600">Temperature</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-900 flex items-center justify-center">
                    <Droplets className="w-5 h-5 mr-1 text-blue-500" />
                    {prediction.weather.humidity}%
                  </div>
                  <div className="text-sm text-gray-600">Humidity</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-900 flex items-center justify-center">
                    <Wind className="w-5 h-5 mr-1 text-blue-500" />
                    {prediction.weather.wind_speed} km/h
                  </div>
                  <div className="text-sm text-gray-600">Wind Speed</div>
                </div>
              </div>
            </div>
          )}
        </div>

        <RunningDaysModal isOpen={runningDaysOpen} onClose={() => setRunningDaysOpen(false)} days={runningDays} />
      </div>
    );
  }

  // Helper functions (defined early so they can be used everywhere)
  const getDelayColor = (delay: number) => {
    if (delay <= 15) return 'text-green-600 bg-green-100';
    if (delay <= 30) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getDelayIcon = (delay: number) => {
    if (delay <= 15) return <CheckCircle className="w-5 h-5" />;
    if (delay <= 30) return <Clock className="w-5 h-5" />;
    return <AlertTriangle className="w-5 h-5" />;
  };

  const getDelayStatus = (delay: number) => {
    if (delay <= 15) return 'On Time';
    if (delay <= 30) return 'Minor Delay';
    return 'Significant Delay';
  };

  // Risk display helpers
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
      <div className={`inline-flex flex-col items-end ml-3`} title={title}>
        <div className={`px-3 py-1 rounded-full text-sm font-semibold ${getRiskColor(score)}`}>
          {score}% • {risk.confidence}
        </div>
        <div className="text-xs text-gray-500 mt-1">{risk.advice}</div>
      </div>
    );
  };

  const renderImputationInfo = (train: any) => {
    if (!train) return null;
    const hasInterval = train.rr_mean_lo !== undefined && train.rr_mean_hi !== undefined && train.rr_mean_pred !== undefined;
    const flagged = train.rr_imputation_flag_final || train.rr_imputation_flag_conservative || train.rr_imputation_flag_conformal;
    return (
      <div className="mt-2 text-xs text-gray-500">
        {hasInterval && (
          <div>Imputed rr_mean: {safeToFixed(train.rr_mean_pred,1)} min ({safeToFixed(train.rr_mean_lo,1)}–{safeToFixed(train.rr_mean_hi,1)})</div>
        )}
        {flagged && (
          <div className="inline-block mt-1 px-2 py-1 rounded text-xs font-semibold text-white bg-red-600" title={
            train.rr_imputation_flag_final ? 'Final imputation flag (conservative OR conformal)' : 'High uncertainty imputation'
          }>
            ⚠ High Uncertainty
          </div>
        )}
      </div>
    );
  };

  // Handle both old and new response structures for direct trains
  const pred = (prediction as any).prediction || prediction;
  
  // Check if we have all_trains (new structure with multiple trains)
  if (prediction.all_trains && Array.isArray(prediction.all_trains) && prediction.all_trains.length > 0) {
    const allTrains = prediction.all_trains;
    const bestRoute = prediction.best_route;
    
    return (
      <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 animate-fade-in">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">✨ Best Predicted Route & All Trains</h2>
          <p className="text-gray-600">
            {formatStation(prediction.route_info?.source || '')} → {formatStation(prediction.route_info?.destination || '')}
          </p>
          <p className="text-sm text-gray-500 mt-1">
            Found {allTrains.length} train option(s) • See all trains with delay predictions
          </p>
        </div>

        {/* Best Route Highlight */}
        {bestRoute && (
          <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-6 mb-6 border-2 border-green-500 shadow-lg">
            <div className="flex items-center space-x-2 mb-4">
              <CheckCircle className="w-6 h-6 text-green-600" />
              <h3 className="text-xl font-bold text-green-800">⭐ Best Predicted Route</h3>
            </div>
            <div className="bg-white rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h4 className="text-lg font-semibold text-gray-900">{bestRoute.train_name}</h4>
                  <p className="text-sm text-gray-600">Train ID: {bestRoute.train_id}</p>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-gray-900">₹{bestRoute.price?.toLocaleString() || 0}</div>
                  <div className="text-sm text-gray-600">Fare</div>
                  {bestRoute.risk && (
                    <div className="mt-2">
                      <button
                        onClick={() => { setRiskModalData(bestRoute.risk); setRiskModalOpen(true); }}
                        className="focus:outline-none"
                        aria-label="Open risk details"
                      >
                        {renderRiskBadge(bestRoute.risk)}
                      </button>
                    </div>
                  )}
                </div>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
                {(bestRoute.departure_time || bestRoute.arrival_time) && (
                  <>
                    <div className="text-center">
                      <div className="text-lg font-bold text-blue-600">
                        {bestRoute.departure_time || '--'}
                      </div>
                      <div className="text-xs text-gray-600">Departure</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-green-600">
                        {bestRoute.arrival_time || '--'}
                      </div>
                      <div className="text-xs text-gray-600">Arrival</div>
                    </div>
                  </>
                )}
                <div className="text-center">
                  <div className={`text-lg font-bold ${getDelayColor(bestRoute.predicted_delay_min || 0).split(' ')[0]}`}>
                    {safeToFixed(bestRoute.predicted_delay_min, 1, '0.0')} min
                  </div>
                  <div className="text-xs text-gray-600">Predicted Delay</div>
                  {renderImputationInfo(bestRoute)}
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-gray-900">
                    {safeToFixed(bestRoute.delay_probability ? bestRoute.delay_probability * 100 : 0, 1, '0.0')}%
                  </div>
                  <div className="text-xs text-gray-600">Delay Probability</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-gray-900">{bestRoute.distance_km || 0} km</div>
                  <div className="text-xs text-gray-600">Distance</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-gray-900">{bestRoute.confidence || 'N/A'}</div>
                  <div className="text-xs text-gray-600">Confidence</div>
                </div>
              </div>
              {bestRoute.delay_factors && Array.isArray(bestRoute.delay_factors) && bestRoute.delay_factors.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <h5 className="text-sm font-semibold text-gray-900 mb-2">Delay Factors:</h5>
                  <div className="flex flex-wrap gap-2">
                    {bestRoute.delay_factors.map((factor: DelayFactor, idx: number) => (
                      <span key={idx} className={`px-2 py-1 rounded text-xs ${
                        factor.impact === 'High' ? 'bg-red-100 text-red-700' :
                        factor.impact === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                        'bg-green-100 text-green-700'
                      }`}>
                        {factor.factor}: {factor.impact}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Short explanation about why this is the best predicted route */}
        <div className="mb-6">
          <div className="bg-blue-50 border border-blue-100 rounded-lg p-4 text-sm text-gray-800">
            <div className="font-semibold mb-1">💡 Why this route is recommended</div>
            <div className="text-sm space-y-2">
              {/* Build a succinct, factual explanation using available fields */}
              {(() => {
                try {
                  if (!bestRoute) return 'Recommended based on model predictions and historical baselines.';

                  const lines: string[] = [];

                  // One-line human reason if provided
                  if (bestRoute.recommendation_reason) {
                    lines.push(bestRoute.recommendation_reason);
                  }

                  // Main contributors summary (short)
                  if (Array.isArray(bestRoute.delay_factors) && bestRoute.delay_factors.length > 0) {
                    const high = bestRoute.delay_factors.filter((f: any) => f.impact === 'High').map((f: any) => f.factor);
                    const medium = bestRoute.delay_factors.filter((f: any) => f.impact === 'Medium').map((f: any) => f.factor);
                    if (high.length) lines.push(`Main reasons: ${high.join(', ')} (high impact).`);
                    else if (medium.length) lines.push(`Main reasons: ${medium.join(', ')}.`);
                  }

                  // Numeric summary: predicted delay & probability
                  if (bestRoute.predicted_delay_min !== undefined && bestRoute.predicted_delay_min !== null) {
                    const dp = Number(bestRoute.predicted_delay_min);
                    const prob = bestRoute.delay_probability ? Number(bestRoute.delay_probability) * 100.0 : null;
                    lines.push(`Model predicts ~${dp.toFixed(1)} min delay${prob !== null ? ` (${prob.toFixed(1)}% chance)` : ''}.`);
                  }

                  // Confidence label
                  if (bestRoute.confidence) {
                    lines.push(`Confidence: ${bestRoute.confidence}.`);
                  }

                  // Short safety guidance
                  const dpVal = Number(bestRoute.predicted_delay_min || 0);
                  const probVal = bestRoute.delay_probability ? Number(bestRoute.delay_probability) : 0;
                  if (!bestRoute.risk || !bestRoute.risk.advice) {
                    if (dpVal <= 15 && probVal <= 0.25) lines.push('Recommendation: safe for most travellers.');
                    else if (dpVal <= 30 && probVal <= 0.5) lines.push('Recommendation: acceptable if you have some buffer time.');
                    else lines.push('Recommendation: consider alternatives if you have tight connections.');
                  } else {
                    lines.push(bestRoute.risk.advice);
                  }

                  return lines.map((l, i) => <div key={i}>{l}</div>);
                } catch (e) {
                  return 'Recommended based on model predictions and historical baselines.';
                }
              })()}
            </div>

            <div className="mt-3 flex items-center gap-3">
              {bestRoute?.feature_contributions && (
                <button
                  type="button"
                  data-testid="btn-view-contributors"
                  title="View the feature-level contributions that explain the prediction"
                  className="inline-flex items-center px-3 py-1 text-sm text-indigo-600 font-semibold hover:underline"
                  onClick={() => { setDelayContribData({ contributions: bestRoute.feature_contributions, top_contributors: bestRoute.top_contributors }); setDelayModalOpen(true); }}
                >
                  View contributors & attribution
                </button>
              )}

              {bestRoute?.risk && (
                <button
                  type="button"
                  data-testid="btn-view-risk"
                  title="View detailed risk breakdown and safety advice"
                  className="inline-flex items-center px-3 py-1 text-sm text-indigo-600 font-semibold hover:underline"
                  onClick={() => { setRiskModalData(bestRoute.risk); setRiskModalOpen(true); }}
                >
                  View risk details
                </button>
              )}

              {/* Short helper text so users know what the buttons do */}
              <div className="text-xs text-gray-500 ml-auto">Click to see detailed contributors or risk advice.</div>
            </div>
          </div>
        </div>

        {/* All Trains List */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">All Available Trains</h3>
          {allTrains.map((train: any, index: number) => (
            <div
              key={`${train.train_id}-${index}`}
              className={`p-4 rounded-xl border-2 transition-all ${
                train.is_best
                  ? 'border-green-500 bg-green-50 shadow-md'
                  : 'border-gray-200 hover:border-gray-300 hover:shadow-md'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  {train.is_best && (
                    <div className="flex items-center text-green-600">
                      <Star className="w-5 h-5 mr-1" />
                      <span className="text-sm font-semibold">Best</span>
                    </div>
                  )}
                  <div>
                    <h4 className="text-lg font-semibold text-gray-900">{train.train_name}</h4>
                    <p className="text-sm text-gray-600">Train ID: {train.train_id}</p>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-xl font-bold text-gray-900">₹{train.price?.toLocaleString() || 0}</div>
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
              <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mt-4">
                {(train.departure_time || train.arrival_time) && (
                  <>
                    <div className="text-center">
                      <div className="text-lg font-bold text-blue-600">
                        {train.departure_time || '--'}
                      </div>
                      <div className="text-xs text-gray-600">Departure</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-green-600">
                        {train.arrival_time || '--'}
                      </div>
                      <div className="text-xs text-gray-600">Arrival</div>
                    </div>
                  </>
                )}
                <div className="text-center">
                  <div className={`text-lg font-bold ${getDelayColor(train.predicted_delay_min || 0).split(' ')[0]}`}>
                    {safeToFixed(train.predicted_delay_min, 1, '0.0')} min
                  </div>
                  <div className="text-xs text-gray-600">Delay</div>
                  {renderImputationInfo(train)}
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-gray-900">
                    {safeToFixed(train.delay_probability ? train.delay_probability * 100 : 0, 1, '0.0')}%
                  </div>
                  <div className="text-xs text-gray-600">Probability</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-gray-900">{train.distance_km || 0} km</div>
                  <div className="text-xs text-gray-600">Distance</div>
                </div>
                <div className="text-center">
                  <div className={`text-lg font-bold ${
                    train.delay_severity === 'Minimal' ? 'text-green-600' :
                    train.delay_severity === 'Low' ? 'text-yellow-600' :
                    'text-red-600'
                  }`}>
                    {train.delay_severity || 'N/A'}
                  </div>
                  <div className="text-xs text-gray-600">Severity</div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Weather Information */}
        {prediction.weather && (
          <div className="mt-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Thermometer className="w-5 h-5 mr-2 text-blue-500" />
              Weather Conditions
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl mb-1">
                  {prediction.weather.condition === 'Clear' ? '☀️' : 
                   prediction.weather.condition === 'Rainy' ? '🌧️' : 
                   prediction.weather.condition === 'Foggy' ? '🌫️' : '☁️'}
                </div>
                <div className="text-sm text-gray-600">{prediction.weather.condition}</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">{prediction.weather.temp || 0}°C</div>
                <div className="text-sm text-gray-600">Temperature</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900 flex items-center justify-center">
                  <Droplets className="w-5 h-5 mr-1 text-blue-500" />
                  {prediction.weather.humidity || 0}%
                </div>
                <div className="text-sm text-gray-600">Humidity</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900 flex items-center justify-center">
                  <Wind className="w-5 h-5 mr-1 text-blue-500" />
                  {prediction.weather.wind_speed || 0} km/h
                </div>
                <div className="text-sm text-gray-600">Wind Speed</div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }
  
  // Safety check - ensure pred exists and has required fields (for old structure)
  if (!pred || (typeof pred.predicted_delay_min === 'undefined' && !pred.connecting_route)) {
    return (
      <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
        <p className="text-gray-600">Invalid prediction data structure</p>
      </div>
    );
  }
  
  const getWeatherIcon = (condition: string) => {
    switch (condition.toLowerCase()) {
      case 'clear':
        return '☀️';
      case 'cloudy':
        return '☁️';
      case 'rainy':
        return '🌧️';
      case 'hot':
        return '🔥';
      case 'foggy':
        return '🌫️';
      default:
        return '🌤️';
    }
  };

  return (
    <>
    <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 animate-fade-in">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">{pred.train_name || 'N/A'}</h2>
          <p className="text-gray-600">Train ID: {pred.train_id || 'N/A'}</p>
          {pred.confidence && (
            <p className="text-sm text-blue-600 mt-1">
              Confidence: <span className="font-semibold">{pred.confidence}</span> | 
              Severity: <span className="font-semibold">{pred.delay_severity || 'N/A'}</span>
            </p>
          )}
        </div>
        <div className="flex items-center space-x-4">
          {pred.risk && (
            <button
              onClick={() => { setRiskModalData(pred.risk); setRiskModalOpen(true); }}
              className="focus:outline-none"
              aria-label="Open risk details"
            >
              {renderRiskBadge(pred.risk)}
            </button>
          )}
          <div className={`px-4 py-2 rounded-full flex items-center space-x-2 ${getDelayColor(pred.predicted_delay_min || 0)}`}>
          {getDelayIcon(pred.predicted_delay_min || 0)}
          <span className="font-semibold">{getDelayStatus(pred.predicted_delay_min || 0)}</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {/* Delay Prediction */}
        <div className="text-center p-4 bg-gray-50 rounded-xl">
          <div className="text-3xl font-bold text-gray-900 mb-1">
            {safeToFixed(pred.predicted_delay_min, 1, '0.0')} min
          </div>
          <div className="text-sm text-gray-600">Predicted Delay</div>
        </div>

        {/* Delay Probability */}
        <div className="text-center p-4 bg-gray-50 rounded-xl">
          <div className="text-3xl font-bold text-gray-900 mb-1">
            {safeToFixed(pred.delay_probability ? pred.delay_probability * 100 : 0, 1, '0.0')}%
          </div>
          <div className="text-sm text-gray-600">Delay Probability</div>
        </div>

        {/* Delay Percentage */}
        {pred.estimated_delay_percentage !== undefined && pred.estimated_delay_percentage !== null && !isNaN(pred.estimated_delay_percentage) && (
          <div className="text-center p-4 bg-gray-50 rounded-xl">
            <div className="text-3xl font-bold text-gray-900 mb-1">
              {safeToFixed(pred.estimated_delay_percentage, 1, '0.0')}%
            </div>
            <div className="text-sm text-gray-600">Delay % of Journey</div>
          </div>
        )}

        {/* Distance */}
        <div className="text-center p-4 bg-gray-50 rounded-xl">
          <div className="text-3xl font-bold text-gray-900 mb-1">
            {pred.route_info?.distance_km || 0} km
          </div>
          <div className="text-sm text-gray-600">Distance</div>
        </div>
      </div>

      {/* Delay Factors Analysis */}
      {pred.delay_factors && Array.isArray(pred.delay_factors) && pred.delay_factors.length > 0 && (
        <div className="bg-gradient-to-r from-orange-50 to-red-50 rounded-xl p-6 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2 text-orange-500" />
            Delay Factors Analysis
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {pred.delay_factors.map((factor: DelayFactor, idx: number) => (
              <div key={idx} className="bg-white rounded-lg p-4 border border-gray-200">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-gray-900">{factor.factor}</span>
                  <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                    factor.impact === 'High' ? 'bg-red-100 text-red-700' :
                    factor.impact === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
                    'bg-green-100 text-green-700'
                  }`}>
                    {factor.impact} Impact
                  </span>
                </div>
                <p className="text-sm text-gray-600">{factor.description}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Weather Information */}
      {pred.weather && (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
            <Thermometer className="w-5 h-5 mr-2 text-blue-500" />
            Weather Conditions
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl mb-1">{getWeatherIcon(pred.weather.condition || 'Clear')}</div>
              <div className="text-sm text-gray-600">{pred.weather.condition || 'N/A'}</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">{pred.weather.temp || 0}°C</div>
              <div className="text-sm text-gray-600">Temperature</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900 flex items-center justify-center">
                <Droplets className="w-5 h-5 mr-1 text-blue-500" />
                {pred.weather.humidity || 0}%
              </div>
              <div className="text-sm text-gray-600">Humidity</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900 flex items-center justify-center">
                <Wind className="w-5 h-5 mr-1 text-blue-500" />
                {pred.weather.wind_speed || 0} km/h
              </div>
              <div className="text-sm text-gray-600">Wind Speed</div>
            </div>
          </div>
        </div>
      )}

      {/* Route Information */}
      {pred.route_info && (
        <div className="bg-gray-50 rounded-xl p-6 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Route Information</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-gray-600">From</div>
              <div className="font-semibold text-gray-900">
                {pred.route_info.source_name || formatStation(pred.route_info.source || 'N/A')}
              </div>
            </div>
            <div>
              <div className="text-gray-600">To</div>
              <div className="font-semibold text-gray-900">
                {pred.route_info.destination_name || formatStation(pred.route_info.destination || 'N/A')}
              </div>
            </div>
            <div>
              <div className="text-gray-600">Day</div>
              <div className="font-semibold text-gray-900">
                {pred.route_info.day_of_week ? 
                  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][pred.route_info.day_of_week - 1] : 
                  'N/A'}
              </div>
            </div>
            <div>
              <div className="text-gray-600">Season</div>
              <div className="font-semibold text-gray-900">{pred.route_info.season || 'N/A'}</div>
            </div>
          </div>
        </div>
      )}

      {/* Travel Recommendation */}
      {pred.recommendation && (
        <div className={`rounded-xl p-4 mb-6 border-2 ${
          pred.recommendation.should_travel 
            ? 'bg-green-50 border-green-300' 
            : 'bg-red-50 border-red-300'
        }`}>
          <div className="flex items-center space-x-2 mb-2">
            {pred.recommendation.should_travel ? (
              <CheckCircle className="w-5 h-5 text-green-600" />
            ) : (
              <AlertTriangle className="w-5 h-5 text-red-600" />
            )}
            <h4 className={`font-semibold ${
              pred.recommendation.should_travel ? 'text-green-800' : 'text-red-800'
            }`}>
              Travel Recommendation
            </h4>
            <span className={`px-3 py-1 rounded-full text-xs font-semibold ${
              pred.recommendation.risk_level === 'Low' ? 'bg-green-100 text-green-700' :
              pred.recommendation.risk_level === 'Medium' ? 'bg-yellow-100 text-yellow-700' :
              'bg-red-100 text-red-700'
            }`}>
              {pred.recommendation.risk_level} Risk
            </span>
          </div>
          <p className={`text-sm ${
            pred.recommendation.should_travel ? 'text-green-700' : 'text-red-700'
          }`}>
            {pred.recommendation.alternative_suggestion}
          </p>
        </div>
      )}

      {/* Explanation */}
      <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
        <h4 className="font-semibold text-yellow-800 mb-2">Prediction Explanation</h4>
        <p className="text-yellow-700 text-sm">{pred.reason}</p>
      </div>

      {pred.feature_contributions && (
        <div className="mt-4 text-right">
          <button
            className="text-sm text-indigo-600 font-semibold hover:underline"
            onClick={() => { setDelayContribData({ contributions: pred.feature_contributions, top_contributors: pred.top_contributors }); setDelayModalOpen(true); }}
          >
            View contributors & attribution
          </button>
        </div>
      )}
    </div>
    <DelayContributorsModal isOpen={delayModalOpen} onClose={() => setDelayModalOpen(false)} title="Delay Contributors" contributions={delayContribData?.contributions} top_contributors={delayContribData?.top_contributors} />
    <RiskBreakdownModal isOpen={riskModalOpen} onClose={() => setRiskModalOpen(false)} risk={riskModalData} />
    </>
  );
};

export default PredictionCard;
