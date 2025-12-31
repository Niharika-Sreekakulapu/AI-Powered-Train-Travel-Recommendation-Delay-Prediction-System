import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import RecommendationsList from '../RecommendationsList';

const sampleRecommendations = {
  preference: 'most_reliable',
  total_trains: 1,
  recommendations: [
    {
      train_id: '00002',
      train_name: 'RECOMMENDED LOCAL',
      predicted_delay_min: 5,
      reliability_score: 88,
      feature_contributions: { Day: 60, Season: 40 },
      top_contributors: ['Day (weekday)', 'Season (summer)']
    }
  ],
  best_route: {
    train_name: 'BEST TRAIN',
    train_id: 'R123',
    price: 150,
    predicted_delay_min: 7,
    speed_kmph: 80,
    reliability_score: 92,
    departure_time: '08:00',
    arrival_time: '11:00',
    feature_contributions: { Distance: 70, Other: 30 },
    top_contributors: ['Distance (70%)', 'Other (30%)']
  },
  route_info: {
    distance_range: { min: 50, max: 50 },
    price_range: { min: 150, max: 150 },
    source: 'AAA',
    destination: 'BBB'
  },
  recommendation_summary: {
    total_options: 1
  },
  weather: { condition: 'Clear', temp: 25, humidity: 40 }
};

describe('RecommendationsList contributors integration', () => {
  it('shows contributors link for best route and opens modal', () => {
    render(<RecommendationsList recommendations={sampleRecommendations} />);

    const btn = screen.getAllByText(/View contributors/i)[0];
    expect(btn).toBeInTheDocument();

    fireEvent.click(btn);

    expect(screen.getByText(/Delay Contributors/)).toBeInTheDocument();
    expect(screen.getByText(/Distance/)).toBeInTheDocument();
    expect(screen.getByText(/Distance \(70%\)/i)).toBeInTheDocument();
  });
});
