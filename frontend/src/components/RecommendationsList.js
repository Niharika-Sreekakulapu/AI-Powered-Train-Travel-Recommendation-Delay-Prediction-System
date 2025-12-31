import React from 'react';
import { Train, Clock, DollarSign, Shield, Zap, Star, TrendingUp } from 'lucide-react';

const RecommendationsList = ({ recommendations }) => {
  // Handle error case
  if (recommendations.error) {
    return (
      <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 animate-fade-in">
        <div className="text-center py-8">
          <div className="text-red-500 mb-4">
            <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.354 16.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">Unable to Load Recommendations</h3>
          <p className="text-gray-600">{recommendations.error}</p>
          <p className="text-gray-500 text-sm mt-2">The prediction results are still available in the Prediction tab.</p>
        </div>
      </div>
    );
  }

  // Handle case with no data message (fallback)
  if (recommendations.no_data_message) {
    return (
      <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 animate-fade-in">
        <div className="text-center py-8">
          <div className="text-blue-500 mb-4">
            <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">Smart Fallback Mode</h3>
          <p className="text-gray-600">{recommendations.no_data_message}</p>
          <p className="text-gray-500 text-sm mt-2">Even without backend connectivity, this system adapts to provide delay predictions in the Predictions tab.</p>
        </div>
      </div>
    );
  }

  // Handle case where recommendations were generated locally - just add banner at top
  let showLocalBanner = recommendations.generated_locally;

  // Check if this is a connecting route vs direct route scenario
  let isConnectingRoute = recommendations.has_direct_trains === false;

  const getPreferenceIcon = (preference) => {
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

  const getPreferenceColor = (preference) => {
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

  const getDelayColor = (delay) => {
    const numDelay = parseFloat(delay);
    if (numDelay <= 15) return 'text-green-600 bg-green-100';
    if (numDelay <= 30) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getReliabilityColor = (score) => {
    const numScore = parseFloat(score);
    if (numScore >= 80) return 'text-green-600 bg-green-100';
    if (numScore >= 60) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100 animate-fade-in">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Train Recommendations</h2>
        <div className={`px-4 py-2 rounded-full bg-gradient-to-r ${getPreferenceColor(recommendations.preference)} text-white flex items-center space-x-2`}>
          {getPreferenceIcon(recommendations.preference)}
          <span className="font-semibold capitalize">{recommendations.preference} First</span>
        </div>
      </div>

        <div className="text-sm text-gray-600 mb-6">
          Found {recommendations.total_trains || 0} trains for your route, sorted by {recommendations.preference || 'best'} preference
        </div>



      {/* Best Route Highlight */}
      {recommendations.best_route && (
        <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-xl p-6 mb-6 border-2 border-green-500 shadow-lg">
          <div className="flex items-center space-x-2 mb-4">
            <Star className="w-6 h-6 text-yellow-500" />
            <h3 className="text-xl font-bold text-green-800">⭐ Best Recommended Route</h3>
          </div>
          <div className="bg-white rounded-lg p-4">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h4 className="text-lg font-semibold text-gray-900">{recommendations.best_route.train_name}</h4>
                <p className="text-sm text-gray-600">Train ID: {recommendations.best_route.train_id}</p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-gray-900">₹{(recommendations.best_route.price || 0).toLocaleString()}</div>
                <div className="text-sm text-gray-600">Fare</div>
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
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
              
              {/* Distance */}
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center mb-1">
                  <Train className="w-4 h-4 text-gray-500 mr-1" />
                  <div className="text-lg font-bold text-gray-900">{recommendations.best_route.distance_km || 0}</div>
                </div>
                <div className="text-xs text-gray-600">km</div>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-4">
        {(recommendations.recommendations || []).map((train, index) => (
          <div
            key={train?.train_id || index}
            className={`p-6 rounded-xl border-2 transition-all duration-200 hover:shadow-lg ${
              train?.is_best
                ? 'border-primary-500 bg-primary-50 shadow-lg'
                : 'border-gray-200 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                    train?.is_best ? 'bg-primary-500' : 'bg-gray-400'
                  }`}>
                    {train?.rank || index + 1}
                  </div>
                  {train?.is_best && (
                    <div className="flex items-center text-primary-600">
                      <Star className="w-4 h-4 mr-1" />
                      <span className="text-sm font-semibold">Best Option</span>
                    </div>
                  )}
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">{train?.train_name || 'Unknown Train'}</h3>
                  <p className="text-sm text-gray-600">Train ID: {train?.train_id || '--'}</p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-gray-900">₹{(train?.price || 0).toLocaleString()}</div>
                <div className="text-sm text-gray-600">Fare</div>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
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
                  <span className="text-lg font-bold text-gray-900">{train.speed_kmph}</span>
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

              {/* Distance */}
              <div className="text-center p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-center mb-1">
                  <Train className="w-4 h-4 text-gray-500 mr-1" />
                  <span className="text-lg font-bold text-gray-900">{train.distance_km}</span>
                </div>
                <div className="text-xs text-gray-600">km</div>
              </div>
            </div>

            {/* Delay and Reliability Badges */}
            <div className="flex flex-wrap gap-2 mt-4">
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${getDelayColor(train.predicted_delay_min)}`}>
                {train.predicted_delay_min <= 15 ? 'On Time' : 
                 train.predicted_delay_min <= 30 ? 'Minor Delay' : 'Significant Delay'}
              </span>
              <span className={`px-3 py-1 rounded-full text-xs font-medium ${getReliabilityColor(train.reliability_score)}`}>
                {train.reliability_score >= 80 ? 'Highly Reliable' : 
                 train.reliability_score >= 60 ? 'Moderately Reliable' : 'Low Reliability'}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Weather Summary */}
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
    </div>
  );
};

export default RecommendationsList;
