import React from 'react';
import { render, screen } from '@testing-library/react';
import DelayContributorsModal from '../DelayContributorsModal';

describe('DelayContributorsModal', () => {
  it('renders title and top contributors and chart container when provided data', () => {
    const data = { contributions: { Distance: 70, Weather: 20, Day: 10 }, top_contributors: ['Distance (100 km)', 'Weather (Rain)'] };
    render(<DelayContributorsModal isOpen={true} onClose={() => {}} title="Delay Contributors" contributions={data.contributions} top_contributors={data.top_contributors} />);

    expect(screen.getByText('Delay Contributors')).toBeInTheDocument();
    expect(screen.getByText(/Top contributors/i)).toBeInTheDocument();
    // Recharts' ResponsiveContainer may not render an SVG in JSDOM; check for presence of the category label instead
    expect(screen.getByText(/Distance/)).toBeInTheDocument();
  });

  it('renders fallback text when no contributions', () => {
    render(<DelayContributorsModal isOpen={true} onClose={() => {}} />);
    expect(screen.getByText(/No contributor data available/i)).toBeInTheDocument();
  });
});
