import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import SearchForm from './components/SearchForm';
import PredictionCard from './components/PredictionCard';
import RecommendationsList from './components/RecommendationsList.js';
import Charts from './components/Charts.js';
import LoadingSpinner from './components/LoadingSpinner.js';
import ErrorAlert from './components/ErrorAlert.js';
import { apiService } from './services/api.js';
import { Train, BarChart3, RefreshCw } from 'lucide-react';

// Generate basic recommendations from prediction data if API fails
const generateBasicRecommendations = (predictionData, preference = 'fastest') => {
  if (!predictionData || !predictionData.all_trains || predictionData.all_trains.length === 0) {
    return {
      recommendations: [],
      best_route: null,
      total_trains: 0,
      preference: preference,
      message: 'No train data available for recommendations.'
    };
  }

  const trains = predictionData.all_trains;

  // Calculate additional metrics for each train
  const trainsWithMetrics = trains.map(train => {
    const delay = train.predicted_delay_min || 0;
    const distance = train.distance_km || 1;

    // Calculate estimated journey time (assuming base speed of 60 km/h, adjusted for delay)
    const baseTime = distance / 60; // hours at 60 km/h
    const totalTime = baseTime + (delay / 60); // Add delay in hours
    const speed = distance / totalTime; // km/h

    // Calculate reliability score (inverse relationship with delay)
    const reliability = Math.max(0, Math.min(100, 100 - delay));

    // Calculate value score (reliability * speed / price)
    const price = train.price || distance * 2; // Fallback pricing
    const valueScore = (reliability * speed) / price * 1000;

    return {
      ...train,
      speed_kmph: Math.round(speed * 100) / 100,
      reliability_score: Math.round(reliability),
      estimated_journey_minutes: Math.round(totalTime * 60),
      estimated_journey_hours: Math.round(totalTime * 100) / 100,
      value_score: Math.round(valueScore * 100) / 100,
      delay_category: delay <= 15 ? 'On Time' : (delay <= 30 ? 'Minor Delay' : 'Significant Delay'),
      recommendation_reason: delay <= 15 ? 'Good time to travel' : (delay <= 30 ? 'Acceptable delay risk' : 'Consider alternative timing')
    };
  });

  // Find best route by preference
  let bestTrain;
  if (preference === 'fastest') {
    bestTrain = trainsWithMetrics.reduce((best, current) =>
      current.speed_kmph > best.speed_kmph ? current : best
    );
  } else if (preference === 'cheapest') {
    bestTrain = trainsWithMetrics.reduce((best, current) =>
      (current.price || 0) < (best.price || 0) ? current : best
    );
  } else if (preference === 'most_reliable') {
    bestTrain = trainsWithMetrics.reduce((best, current) =>
      current.reliability_score > best.reliability_score ? current : best
    );
  } else {
    // best_value
    bestTrain = trainsWithMetrics.reduce((best, current) =>
      current.value_score > best.value_score ? current : best
    );
  }

  // Mark best train and create rankings
  const rankedTrains = trainsWithMetrics.map((train, index) => ({
    ...train,
    rank: index + 1,
    is_best: train.train_id === bestTrain.train_id,
    tags: [
      train.train_id === trainsWithMetrics.reduce((best, current) => current.speed_kmph > best.speed_kmph ? current : best).train_id ? 'Fastest' : null,
      train.train_id === trainsWithMetrics.reduce((best, current) => (current.price || 0) < (best.price || 0) ? current : best).train_id ? 'Cheapest' : null,
      train.train_id === trainsWithMetrics.reduce((best, current) => current.reliability_score > best.reliability_score ? current : best).train_id ? 'Most Reliable' : null,
      train.train_id === trainsWithMetrics.reduce((best, current) => current.value_score > best.value_score ? current : best).train_id ? 'Best Value' : null
    ].filter(tag => tag !== null)
  }));

  // Limit to top 3 for UI performance
  const topRecommendations = rankedTrains.slice(0, 3);

  return {
    recommendations: topRecommendations,
    best_route: bestTrain,
    total_trains: trainsWithMetrics.length,
    preference: preference,
    has_direct_trains: true,
    weather: predictionData.weather,
    route_info: predictionData.route_info,
    message: 'Recommendations generated from basic analysis (API unavailable)',
    generated_locally: true
  };
};

function App() {
  const [prediction, setPrediction] = useState(null);

  // Lazy load propagation panel to keep initial bundle small
  const LazyPropagationPanel = React.lazy(() => import('./components/PropagationPanel'));
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('prediction');
  // Keep last searched route so analytics can be viewed even when no prediction is active
  const [lastRoute, setLastRoute] = useState({ source: '', destination: '' });

  // Check API health on component mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        await apiService.healthCheck();
      } catch (err) {
        setError('Unable to connect to the prediction service. Please make sure the backend is running.');
      }
    };
    checkHealth();
  }, []);

  const handleSearch = async (formData) => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    setRecommendations(null);
    // Remember last searched route for analytics tab
    setLastRoute({ source: formData.source || '', destination: formData.destination || '' });

    const hasRouteSearch = formData.source && formData.destination;
    const hasTrainOnlySearch = formData.train_id && !hasRouteSearch;

    try {
      // Always call predict API (it supports both route and train-only search)
      const predictionResult = await apiService.predictDelay({
        source: formData.source,
        destination: formData.destination,
        travel_date: formData.travel_date,
        train_id: formData.train_id
      });
      setPrediction(predictionResult);

      // Only call recommendations API if we have route search (source + destination)
      if (hasRouteSearch) {
        try {
          const recommendationsResult = await apiService.getRecommendations({
            source: formData.source,
            destination: formData.destination,
            travel_date: formData.travel_date,
            preference: formData.preference
          });
          setRecommendations(recommendationsResult);
          console.log('Recommendations loaded successfully:', recommendationsResult);
        } catch (recommendationsErr) {
          console.warn('Recommendations API failed:', recommendationsErr);
          // Generate basic recommendations from prediction data instead of showing error
          console.log('Generating recommendations from prediction data...');

          // Create basic recommendations from the prediction result
          if (predictionResult && predictionResult.all_trains) {
            const recommendationsData = generateBasicRecommendations(predictionResult, formData.preference);
            setRecommendations(recommendationsData);
            console.log('Generated basic recommendations:', recommendationsData);
          } else {
            // Fallback: create helpful message about checking predictions tab
            const fallbackState = {
              recommendations: [],
              best_route: null,
              total_trains: 0,
              no_data_message: 'Check the Predictions tab for delay information. The recommendations feature requires a backend connection.',
              route_info: predictionResult?.route_info || {}
            };
            setRecommendations(fallbackState);
          }
        }
      }

      setActiveTab('prediction');
    } catch (err) {
      console.error('Prediction API error:', err);
      setError(err.response?.data?.error || err.message || 'An error occurred while fetching predictions');
      // Ensure loading is stopped even on error
      setLoading(false);
      return;
    }

    setLoading(false);
  };

  const clearResults = () => {
    setPrediction(null);
    setRecommendations(null);
    setError(null);
  };

  const refreshData = () => {
    if (prediction) {
      // Re-run the last search
      const formData = {
        source: prediction.route_info.source,
        destination: prediction.route_info.destination,
        travel_date: new Date().toISOString().split('T')[0], // Today's date
        preference: 'fastest'
      };
      handleSearch(formData);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-primary-500 to-primary-600 rounded-xl flex items-center justify-center">
                <Train className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">TrainDelay AI</h1>
                <p className="text-sm text-gray-600">Intelligent Delay Prediction</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {prediction && (
                <button
                  onClick={refreshData}
                  disabled={loading}
                  className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors disabled:opacity-50"
                >
                  <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                  <span>Refresh</span>
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Search Form */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <SearchForm onSubmit={handleSearch} loading={loading} />
        </motion.div>

        {/* Error Alert */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="mt-6"
            >
              <ErrorAlert 
                message={error} 
                onClose={() => setError(null)}
                type="error"
              />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Loading State */}
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-6"
          >
            <LoadingSpinner 
              message="Analyzing your route and predicting delays..." 
              size="lg"
            />
          </motion.div>
        )}

        {/* Results */}
        <AnimatePresence>
          {(prediction || recommendations) && !loading && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="mt-8"
            >
              {/* Tab Navigation */}
              <div className="flex space-x-1 bg-gray-100 p-1 rounded-xl mb-6">
                <button
                  onClick={() => setActiveTab('prediction')}
                  className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-all ${
                    activeTab === 'prediction'
                      ? 'bg-white text-primary-600 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <Train className="w-4 h-4" />
                  <span>Prediction</span>
                </button>
                <button
                  onClick={() => setActiveTab('recommendations')}
                  className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-all ${
                    activeTab === 'recommendations'
                      ? 'bg-white text-primary-600 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <BarChart3 className="w-4 h-4" />
                  <span>Recommendations</span>
                </button>
                <button
                  onClick={() => setActiveTab('analytics')}
                  className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-all ${
                    activeTab === 'analytics'
                      ? 'bg-white text-primary-600 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <BarChart3 className="w-4 h-4" />
                  <span>Analytics</span>
                </button>
                <button
                  onClick={() => setActiveTab('propagation')}
                  className={`flex-1 flex items-center justify-center space-x-2 py-3 px-4 rounded-lg font-medium transition-all ${
                    activeTab === 'propagation'
                      ? 'bg-white text-primary-600 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <BarChart3 className="w-4 h-4" />
                  <span>Propagation</span>
                </button>
              </div>

              {/* Tab Content */}
              <AnimatePresence mode="wait">
                {activeTab === 'prediction' && prediction && (
                  <motion.div
                    key="prediction"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <PredictionCard prediction={prediction} />
                  </motion.div>
                )}

                {activeTab === 'recommendations' && recommendations && (
                  <motion.div
                    key="recommendations"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <RecommendationsList recommendations={recommendations} />
                  </motion.div>
                )}

                {activeTab === 'analytics' && (
                  <motion.div
                    key="analytics"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <Charts 
                      prediction={prediction || undefined} 
                      recommendations={recommendations?.recommendations}
                      source={lastRoute.source}
                      destination={lastRoute.destination}
                    />
                  </motion.div>
                )}

                {activeTab === 'propagation' && (
                  <motion.div
                    key="propagation"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 20 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="grid grid-cols-1 md:grid-cols-1 gap-6">
                      <div>
                        <h3 className="text-lg font-semibold mb-3">Propagation Simulator</h3>
                        <div className="bg-white p-6 rounded-xl shadow-sm">
                          <p className="text-sm text-gray-600">Run a historical-day propagation/backtest and visualize results. This uses a simplified dependency graph based on train terminal/source timing.</p>
                          <div className="mt-4">
                            {/* Lazy load component to reduce initial bundle size */}
                            <React.Suspense fallback={<div>Loading...</div>}>
                              <LazyPropagationPanel />
                            </React.Suspense>
                          </div>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center text-gray-600">
            <p>&copy; 2024 TrainDelay AI. Built with React, Flask, and Machine Learning.</p>
            <p className="mt-2 text-sm">
              This is a demonstration project for train delay prediction using AI.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
