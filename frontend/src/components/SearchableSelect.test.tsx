import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import SearchableSelect from './SearchableSelect';

const mockGroupedOptions = [
  { label: 'Andhra Pradesh Stations', options: [{ code: 'AP1', name: 'Visakhapatnam' }, { code: 'AP2', name: 'Vijayawada' }] },
  { label: 'All Other Stations', options: [{ code: 'MUM', name: 'Mumbai Cst' }, { code: 'CHN', name: 'Chennai Central' }] },
  { label: 'Other Major Cities', options: [{ code: 'PNQ', name: 'Pune' }, { code: 'BLR', name: 'Bengaluru' }] }
];

test('renders grouped options with AP stations first, then others', () => {
  const handleChange = jest.fn();
  render(
    <SearchableSelect groupedOptions={mockGroupedOptions as any} value={''} onChange={handleChange} />
  );

  // Open dropdown by clicking the placeholder text
  const trigger = screen.getByText('Search and select station...');
  fireEvent.click(trigger);

  // Validate group headers order (ensure specific headers are present and in document order)
  const apHeader = screen.getByText('Andhra Pradesh Stations');
  const allHeader = screen.getByText('All Other Stations');
  const majorsHeader = screen.getByText('Other Major Cities');

  // Ensure apHeader appears before allHeader, and allHeader before majorsHeader
  expect(apHeader.compareDocumentPosition(allHeader) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(allHeader.compareDocumentPosition(majorsHeader) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

  // Validate first option is AP station
  expect(screen.getByText('Visakhapatnam')).toBeTruthy();
});
