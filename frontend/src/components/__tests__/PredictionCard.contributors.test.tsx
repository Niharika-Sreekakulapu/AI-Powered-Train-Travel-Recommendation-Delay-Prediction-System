import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import PredictionCard from '../PredictionCard';

const samplePrediction = {
  prediction: {
    train_id: '00001',
    train_name: 'TEST EXPRESS',
    predicted_delay_min: 12,
    delay_probability: 0.3,
    confidence: 'High',
    delay_severity: 'Minor',
    estimated_delay_percentage: 30,
    reason: 'distance_km factor',
    delay_factors: [],
    route_info: { source: 'AAA', destination: 'BBB' },
    weather: { condition: 'Clear', temp: 25, humidity: 40, wind_speed: 5 },
    feature_contributions: { Distance: 80, Weather: 20 },
    top_contributors: ['Distance (200 km)', 'Weather (Clear)']
  }
};

describe('PredictionCard contributors integration', () => {
  it('shows contributors button and opens modal with top contributors', () => {
    render(<PredictionCard prediction={samplePrediction} />);

    const btn = screen.getByText(/View contributors & attribution/i);
    expect(btn).toBeInTheDocument();

    fireEvent.click(btn);

    expect(screen.getByText(/Delay Contributors/)).toBeInTheDocument();
    expect(screen.getByText(/Top contributors/i)).toBeInTheDocument();
    expect(screen.getByText(/Distance \(200 km\)/i)).toBeInTheDocument();
  });
});
