jest.mock('../../services/api');
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import PredictionCard from '../PredictionCard';
import RunningDaysModal from '../RunningDaysModal';
import apiService from '../../services/api';

const mockedApi = apiService as jest.Mocked<typeof apiService>;

describe('PredictionCard connecting route running days modal', () => {
  it('opens modal and displays running days when button clicked', async () => {
    const mockPrediction: any = {
      has_direct_trains: false,
      connecting_route: {
        connecting_station: 'MID',
        train1: { train_id: '00001', train_name: 'Train A', source: 'SRC', destination: 'MID', distance_km: 100, price: 100, predicted_delay_min: 5 },
        train2: { train_id: '00002', train_name: 'Train B', source: 'MID', destination: 'DST', distance_km: 150, price: 200, predicted_delay_min: 10 },
        total_distance: 250,
        total_price: 300,
        total_delay: 15,
        layover_time: 90
      },
      note: 'Relaxed search used',
      message: 'Sample connecting route',
      route_info: { source: 'SRC', destination: 'DST' },
      weather: { temp: 25, condition: 'Clear', humidity: 50, wind_speed: 5 }
    };

    // Ensure the API mock is in place
    (apiService as any).getAvailableDays = jest.fn().mockResolvedValue([{ day: 3, name: 'Wednesday' }, { day: 5, name: 'Friday' }]);

    render(<PredictionCard prediction={mockPrediction} />);

    const checkBtn = screen.getByText('Check running days');
    fireEvent.click(checkBtn);

    await waitFor(() => {
      expect(screen.getByText('Available Running Days')).toBeInTheDocument();
    });

    expect(screen.getByText('Wednesday')).toBeInTheDocument();
    expect(screen.getByText('Friday')).toBeInTheDocument();
  });
});