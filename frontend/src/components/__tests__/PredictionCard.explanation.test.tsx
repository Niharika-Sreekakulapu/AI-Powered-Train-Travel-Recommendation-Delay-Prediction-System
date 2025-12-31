jest.mock('../../services/api');
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import PredictionCard from '../PredictionCard';

describe('PredictionCard explanation text', () => {
  it('renders a detailed, factual explanation for the best route and shows detail buttons', async () => {
    const best: any = {
      train_id: '12345',
      train_name: 'TEST EXPRESS',
      predicted_delay_min: 12.3,
      delay_probability: 0.15,
      confidence: 'High',
      recommendation_reason: 'Good balance of speed and reliability',
      feature_contributions: { Distance: 70, Weather: 30 },
      top_contributors: ['Distance (70%)', 'Weather (30%)'],
      delay_factors: [
        { factor: 'Distance', impact: 'High', description: '300 km segment' },
        { factor: 'Weather', impact: 'Low', description: 'Clear' }
      ],
      risk: { advice: '✅ Recommended' }
    };

    const mockPrediction: any = {
      all_trains: [best],
      best_route: best,
      route_info: { source: 'VSKP', destination: 'BZA' }
    };

    render(<PredictionCard prediction={mockPrediction} />);

    // Check that explanation contains model prediction phrase
    expect(screen.getByText(/Model predicts ~12.3 min delay/i)).toBeInTheDocument();

    // Check that main contributors are mentioned
    expect(screen.getByText(/Main reasons: Distance/i)).toBeInTheDocument();

    // Buttons to view details should be present
    expect(screen.getByText(/View contributors & attribution/i)).toBeInTheDocument();
    expect(screen.getByText(/View risk details/i)).toBeInTheDocument();

    // Click contributors and risk buttons to ensure they are clickable and wired (modal rendering tested separately)
    fireEvent.click(screen.getByTestId('btn-view-contributors'));
    fireEvent.click(screen.getByTestId('btn-view-risk'));

    // Buttons should have helpful titles for users
    expect(screen.getByTestId('btn-view-contributors')).toHaveAttribute('title', expect.stringContaining('View the'));
    expect(screen.getByTestId('btn-view-risk')).toHaveAttribute('title', expect.stringContaining('View detailed'));
  });
});