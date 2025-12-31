import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import Charts from '../Charts';
import { apiService } from '../../services/api';

jest.mock('../../services/api');

const mockedApi = apiService as jest.Mocked<typeof apiService>;

test('fetches and displays analytics for a route', async () => {
  const analytics = {
    delayTrendData: [ { month: 'Jan', delay: 10 }, { month: 'Feb', delay: 12 }, { month: 'Mar', delay: 15 }, { month: 'Apr', delay: 20 }, { month: 'May', delay: 18 }, { month: 'Jun', delay: 25 }, { month: 'Jul', delay: 30 }, { month: 'Aug', delay: 28 }, { month: 'Sep', delay: 22 }, { month: 'Oct', delay: 19 }, { month: 'Nov', delay: 14 }, { month: 'Dec', delay: 11 } ],
    reliabilityData: [ { name: 'On Time', value: 50, color: '#10B981' }, { name: 'Minor Delay', value: 40, color: '#F59E0B' }, { name: 'Major Delay', value: 10, color: '#EF4444' } ],
    seasonData: [ { season: 'Winter', delay: 12, reliability: 85 }, { season: 'Spring', delay: 18, reliability: 80 } ],
    keyInsights: { on_time_percentage: 50, average_delay_min: 18, peak_delay_season: 'Spring' }
  };

  mockedApi.getAnalytics = jest.fn().mockResolvedValue(analytics);

  const fakePrediction: any = { route_info: { source: 'SRC', destination: 'DST' } };

  render(<Charts prediction={fakePrediction} recommendations={[]} />);

  await waitFor(() => {
    expect(screen.getByText('Key Insights')).toBeInTheDocument();
  });

  await waitFor(() => {
    expect(screen.getByText('50%')).toBeInTheDocument();
    expect(screen.getByText('Spring')).toBeInTheDocument();
  });

  // When analytics are historical, no model badge is shown
  expect(screen.queryByText('Model-based estimate')).not.toBeInTheDocument();

});

test('fetches analytics with explicit source/destination props (no prediction)', async () => {
  const analytics = {
    delayTrendData: [ { month: 'Jan', delay: 5 }, { month: 'Feb', delay: 7 }, { month: 'Mar', delay: 6 } ],
    reliabilityData: [ { name: 'On Time', value: 2, color: '#10B981' }, { name: 'Minor Delay', value: 1, color: '#F59E0B' } ],
    seasonData: [ { season: 'Winter', delay: 6, reliability: 80 } ],
    keyInsights: { on_time_percentage: 66.7, average_delay_min: 6, peak_delay_season: 'Winter' }
  };

  mockedApi.getAnalytics = jest.fn().mockResolvedValue({...analytics, model_based: true});

  render(<Charts source={'SRC'} destination={'DST'} />);

  await waitFor(() => {
    expect(screen.getByText('Key Insights')).toBeInTheDocument();
  });

  await waitFor(() => {
    expect(screen.getByText('66.7%')).toBeInTheDocument();
    expect(screen.getByText('Winter')).toBeInTheDocument();
  });

  // Model-based flag should render the badge
  await waitFor(() => {
    expect(screen.getByText('Model-based estimate')).toBeInTheDocument();
  });
});