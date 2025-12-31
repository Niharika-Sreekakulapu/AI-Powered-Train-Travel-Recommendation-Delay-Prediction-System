import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { apiService } from '../services/api';
import { TrainPrediction, TrainRecommendation } from '../types';

interface ChartsProps {
  prediction?: TrainPrediction;
  recommendations?: TrainRecommendation[];
  source?: string;
  destination?: string;
}

const Charts: React.FC<ChartsProps> = ({ prediction, recommendations, source, destination }) => {
  const [delayTrendData, setDelayTrendData] = React.useState<{month:string, delay:number}[]>([ ]);
  const [reliabilityData, setReliabilityData] = React.useState<any[]>([]);
  const [seasonData, setSeasonData] = React.useState<any[]>([]);
  const [keyInsights, setKeyInsights] = React.useState<any | null>(null);
  const [isModelBased, setIsModelBased] = React.useState(false);
  const [loading, setLoading] = React.useState(false);

  // Fetch analytics when prediction / route info is available,
  // or when explicit source/destination props are provided (allow viewing analytics without running a prediction)
  React.useEffect(() => {
    let mounted = true;
    async function fetchAnalytics() {
      let src: string | undefined = undefined;
      let dst: string | undefined = undefined;

      if (prediction && prediction.route_info) {
        src = prediction.route_info.source;
        dst = prediction.route_info.destination;
      } else if (source || destination) {
        src = source;
        dst = destination;
      } else {
        return;
      }

      if (!src || !dst) return;

      // Debug: log which route we are fetching analytics for
      console.log('Charts: fetching analytics for', { src, dst });

      // Debug: inspect apiService object
      try {
        console.log('Charts: apiService', apiService);
        console.log('Charts: getAnalytics exists?', typeof apiService.getAnalytics);
      } catch (err) {
        console.warn('Charts: failed to inspect apiService', err);
      }

      setLoading(true);
      try {
        const res = await apiService.getAnalytics(src, dst);
        // Debug: log the analytics response
        console.log('Charts: analytics response', res);
        if (!mounted) return;
        if (res.delayTrendData) setDelayTrendData(res.delayTrendData);
        if (res.reliabilityData) setReliabilityData(res.reliabilityData);
        if (res.seasonData) setSeasonData(res.seasonData);
        if (res.keyInsights) setKeyInsights(res.keyInsights);
        setIsModelBased(Boolean(res.model_based));
      } catch (e) {
        console.warn('Failed to fetch analytics', e);
      } finally {
        if (mounted) setLoading(false);
      }
    }
    fetchAnalytics();
    return () => { mounted = false; };
  }, [prediction, source, destination]);

  // Fallbacks for initial rendering
  const staticDelay = delayTrendData.length ? delayTrendData : [
    { month: 'Jan', delay: 0 }, { month: 'Feb', delay: 0 }, { month: 'Mar', delay: 0 }, { month: 'Apr', delay: 0 }, { month: 'May', delay: 0 }, { month: 'Jun', delay: 0 }, { month: 'Jul', delay: 0 }, { month: 'Aug', delay: 0 }, { month: 'Sep', delay: 0 }, { month: 'Oct', delay: 0 }, { month: 'Nov', delay: 0 }, { month: 'Dec', delay: 0 }
  ];

  // Feature flags to control chart visibility
  const SHOW_DELAY_TREND = true;   // show Monthly Delay Trends
  const SHOW_SEASONAL = false;     // seasonal performance chart
  const SHOW_PIE_CHART = false;    // delay distribution pie chart

  const isEmpty = !loading && (!delayTrendData || delayTrendData.length === 0) && (!reliabilityData || reliabilityData.length === 0) && (!seasonData || seasonData.length === 0);


  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

  return (
    <div className="space-y-6">
      {/* Delay Trends Chart */}
      {SHOW_DELAY_TREND && (
      <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
        <h3 className="text-xl font-bold text-gray-900 mb-4">Monthly Delay Trends</h3>
        <div className="h-80">
          {isEmpty ? (
            <div className="flex items-center justify-center h-full text-gray-500">No analytics available for this route</div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={delayTrendData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="month" 
                stroke="#6b7280"
                fontSize={12}
              />
              <YAxis 
                stroke="#6b7280"
                fontSize={12}
                label={{ value: 'Delay (minutes)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="delay" 
                stroke="#3B82F6" 
                strokeWidth={3}
                dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
                activeDot={{ r: 6, stroke: '#3B82F6', strokeWidth: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
          )}
        </div>
      </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Reliability Distribution */}
        {SHOW_PIE_CHART && (
        <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Delay Distribution</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={reliabilityData.length ? reliabilityData : [ { name: 'On Time', value: 1, color: '#10B981' } ] }
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {(reliabilityData.length ? reliabilityData : [{ name: 'On Time', value: 1, color: '#10B981' }]).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'white',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center space-x-4 mt-4">
            {(() => {
              const total = reliabilityData.reduce((s, it) => s + (it.value || 0), 0) || 0;
              return reliabilityData.map((item, index) => {
                const pct = total ? Math.round(((item.value || 0) / total) * 100) : 0;
                return (
                  <div key={index} className="flex items-center space-x-2">
                    <div 
                      className="w-3 h-3 rounded-full" 
                      style={{ backgroundColor: item.color }}
                    ></div>
                    <span className="text-sm text-gray-600">{item.name} — {pct}%</span>
                  </div>
                );
              });
            })()}
          </div>
        </div>
        )}

        {/* Seasonal Performance */}
        {SHOW_SEASONAL && (
        <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Seasonal Performance</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={seasonData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="season" 
                  stroke="#6b7280"
                  fontSize={12}
                />
                <YAxis 
                  stroke="#6b7280"
                  fontSize={12}
                  label={{ value: 'Delay (min)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'white',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                />
                <Bar 
                  dataKey="delay" 
                  fill="#3B82F6" 
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
        )}
      </div>

      {/* Train Comparison Chart */}
      {recommendations && recommendations.length > 0 && (
        <div className="bg-white rounded-2xl shadow-xl p-6 border border-gray-100">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Train Comparison</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={recommendations.slice(0, 5)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="train_name" 
                  stroke="#6b7280"
                  fontSize={10}
                  angle={-45}
                  textAnchor="end"
                  height={80}
                />
                <YAxis 
                  stroke="#6b7280"
                  fontSize={12}
                  label={{ value: 'Delay (minutes)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  contentStyle={{
                    backgroundColor: 'white',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                />
                <Bar 
                  dataKey="predicted_delay_min" 
                  fill="#3B82F6" 
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Key Insights */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-6 border border-blue-200">
        <div className="flex items-start justify-between">
          <h3 className="text-xl font-bold text-gray-900 mb-4">Key Insights</h3>
          {isModelBased && (
            <div className="text-xs text-yellow-700 bg-yellow-50 px-3 py-1 rounded-full border border-yellow-100">
              Model-based estimate
            </div>
          )}
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-white rounded-xl">
            <div className="text-2xl font-bold text-blue-600 mb-1">{keyInsights ? `${keyInsights.on_time_percentage}%` : '—'}</div>
            <div className="text-sm text-gray-600">On-time Performance</div>
          </div>
          <div className="text-center p-4 bg-white rounded-xl">
            <div className="text-2xl font-bold text-green-600 mb-1">{keyInsights ? keyInsights.peak_delay_season : '—'}</div>
            <div className="text-sm text-gray-600">Peak Delay Season</div>
          </div>
          <div className="text-center p-4 bg-white rounded-xl">
            <div className="text-2xl font-bold text-purple-600 mb-1">{keyInsights ? `${keyInsights.average_delay_min} min` : '—'}</div>
            <div className="text-sm text-gray-600">Average Delay</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Charts;