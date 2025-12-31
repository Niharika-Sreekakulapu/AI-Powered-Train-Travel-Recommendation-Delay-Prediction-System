import React, { useState, useEffect } from 'react';
import { SearchFormData } from '../types';
import { Calendar, MapPin, Train, Settings } from 'lucide-react';
import stationMapping from '../utils/stationMapping.json';
import SearchableSelect from './SearchableSelect';

interface SearchFormProps {
  onSubmit: (data: SearchFormData) => void;
  loading?: boolean;
}

// Get stations from mapping file
const allStations = stationMapping.stations || [];

// Filter Telangana stations (major cities in Telangana state)
const telanganaStations = allStations.filter(station =>
  station.name.includes('Hyderabad') ||
  station.name.includes('Secunderabad') ||
  station.name.includes('Warangal') ||
  station.name.includes('Karimnagar') ||
  station.name.includes('Nizamabad') ||
  station.name.includes('Khammam') ||
  station.name.includes('Ramagundam') ||
  station.name.includes('Mahbubnagar') ||
  station.name.includes('Nalgonda') ||
  station.name.includes('Adilabad') ||
  station.name.includes('Siddipet') ||
  station.name.includes('Medak') ||
  station.name.includes('Kamareddy') ||
  station.name.includes('Jangaon') ||
  station.name.includes('Kazipet') ||
  station.name.includes('Peddapalli') ||
  station.name.includes('Kaghaznagar') ||
  station.name.includes('Dornakal') ||
  station.name.includes('Mahbubabad') ||
  station.name.includes('Suryapet') ||
  station.name.includes('Vikarabad') ||
  station.name.includes('Tandur') ||
  station.name.includes('Secundrabad') || // Alternative spelling
  station.name.includes('Kacheguda') ||
  station.name.includes('Begumpet') ||
  station.name.includes('Lingampalli') ||
  station.name.includes('Medchal') ||
  station.name.includes('Shamshabad') ||
  station.name.includes('Hi-Tech') ||
  station.name.includes('Hitech') ||
  station.name.includes('Jubilee Hills') ||
  station.name.includes('Khairatabad') ||
  station.name.includes('Nampally')
);

// Filter Andhra Pradesh stations (remaining after Telangana separation)
const andhraPradeshStations = allStations.filter(station =>
  !telanganaStations.find(ts => ts.code === station.code) && (
  station.name.includes('Vijayawada') ||
  station.name.includes('Visakhapatnam') ||
  station.name.includes('Tirupati') ||
  station.name.includes('Nellore') ||
  station.name.includes('Guntur') ||
  station.name.includes('Rajahmundry') ||
  station.name.includes('Kakinada') ||
  station.name.includes('Anantapur') ||
  station.name.includes('Kadapa') ||
  station.name.includes('Chittoor') ||
  station.name.includes('Eluru') ||
  station.name.includes('Tenali') ||
  station.name.includes('Srikakulam') ||
  station.name.includes('Kadiri') ||
  station.name.includes('Hindupur') ||
  station.name.includes('Dharmavaram') ||
  station.name.includes('Bhimavaram') ||
  station.name.includes('Machilipatnam') ||
  station.name.includes('Tadepalligudem') ||
  station.name.includes('Tanuku') ||
  station.name.includes('Palakollu') ||
  station.name.includes('Gudur') ||
  station.name.includes('Duvvada') ||
  station.name.includes('Annavaram') ||
  station.name.includes('Samalkot') ||
  station.name.includes('Gudivada') ||
  station.name.includes('Chirala') ||
  station.name.includes('Amalapuram') ||
  station.name.includes('Narsipatnam') ||
  station.name.includes('Sompeta') ||
  station.name.includes('Palasa') ||
  station.name.includes('Berhampur') ||
  station.name.includes('Khallikot') ||
  station.name.includes('Kotabommali') ||
  station.name.includes('Mandasa') ||
  station.name.includes('Bobbili') ||
  station.name.includes('Parvatipuram') ||
  station.name.includes('Rayagada') ||
  station.name.includes('Naupada') ||
  station.name.includes('Pedana') ||
  station.name.includes('Repalle') ||
  station.name.includes('Vuyyuru') ||
  station.name.includes('Vetapalem') ||
  station.name.includes('Amalapuram') ||
  station.name.includes('Narsapur') ||
  station.name.includes('Palakollu') ||
  station.name.includes('Tanuku') ||
  station.name.includes('Tadepalligudem') ||
  station.name.includes('Eluru') ||
  station.name.includes('Wamtum') ||
  station.name.includes('Goligeni') ||
  station.name.includes('Navabpalem') ||
  station.name.includes('Marampalli') ||
  station.name.includes('Nujvid') ||
  station.name.includes('Kolena') ||
  station.name.includes('Chandanagar') ||
  station.name.includes('Diviti Pally') ||
  station.name.includes('Gangavathi') ||
  station.name.includes('Crianakalapalli') ||
  station.name.includes('Peddana') ||
  station.name.includes('Gudivada') ||
  station.name.includes('Tarigoppula') ||
  station.name.includes('Vijayawada Jn') ||
  station.name.includes('Visakhapatnam Jn') ||
  station.name.includes('Tirupati Jn') ||
  station.name.includes('Gudur Jn') ||
  station.name.includes('Ongole') ||
  station.name.includes('Guntur Jn') ||
  station.name.includes('Anantapur Jn') ||
  station.name.includes('Dharmavaram Jn') ||
  station.name.includes('Bhimavaram Town') ||
  station.name.includes('Machilipatnam Jn') ||
  station.name.includes('Samalkot Jn') ||
  station.name.includes('Gudivada Jn') ||
  station.name.includes('Chirala') ||
  station.name.includes('Palasa') ||
  station.name.includes('Khallikot') ||
  station.name.includes('Royagada') ||
  station.name.includes('Mandasa Road') ||
  station.name.includes('Kotabommali') ||
  station.name.includes('Bobbili') ||
  station.name.includes('Parvatipuram') ||
  station.name.includes('Parvatipuram Tn') ||
  station.name.includes('Salur') ||
  station.name.includes('Srikakulam Road') ||
  station.name.includes('Tekkali') ||
  station.name.includes('Gudlavalleru') ||
  station.name.includes('Gudl') || // For Gudlavalleru and similar
  station.name.includes('Kaikaluru') ||
  station.name.includes('Akividu') ||
  station.name.includes('Bhimadolu') ||
  station.name.includes('Manchalipuram') ||
  station.name.includes('Undi') ||
  station.name.includes('Kavutaram') ||
  station.name.includes('Vissannapeta') ||
  station.name.includes('Zamindar') ||
  station.name.includes('Indupalli') ||
  station.name.includes('Moturu') ||
  station.name.includes('Rayanapadu') ||
  station.name.includes('Krishna Canal') ||
  station.name.includes('Mustabada') ||
  station.name.includes('Gannavaram') ||
  station.name.includes('Pedana') ||
  station.name.includes('Manipal') ||
  station.name.includes('Bantumilli') ||
  station.name.includes('Mantripalem') ||
  station.name.includes('Ruthiyai')
));

// Get remaining stations (not Andhra Pradesh or Telangana)
const remainingStations = allStations.filter(station =>
  !telanganaStations.find(ts => ts.code === station.code) &&
  !andhraPradeshStations.find(ap => ap.code === station.code)
);

// Sort all three groups alphabetically
const sortedTelanganaStations = [...telanganaStations].sort((a, b) => a.name.localeCompare(b.name));
const sortedAPStations = [...andhraPradeshStations].sort((a, b) => a.name.localeCompare(b.name));
const sortedRemainingStations = [...remainingStations].sort((a, b) => a.name.localeCompare(b.name));

// Additional grouped stations: AP first, All stations, then major cities outside AP
const sortedAllStations = [...allStations].sort((a, b) => a.name.localeCompare(b.name));

// Major city keywords for non-AP 'major cities' group
const majorCityKeywords = [
  'Mumbai', 'Chennai', 'Bengaluru', 'Bangalore', 'New Delhi', 'Delhi', 'Pune', 'Kolkata', 'Jaipur', 'Lucknow', 'Bhubaneswar', 'Coimbatore', 'Madurai', 'Thiruvananthapuram', 'Ernakulam', 'Kochi', 'Vijayawada', 'Visakhapatnam'
];

// Filter remaining major cities from remainingStations
const sortedRemainingMajorCities = [...remainingStations]
  .filter(st => majorCityKeywords.some(k => st.name.includes(k)))
  .sort((a,b) => a.name.localeCompare(b.name));

// Grouped options: AP stations first, then all stations (complete list), then major cities outside AP
// Avoid duplicating AP stations in the 'All Stations' group by removing AP entries
const apCodes = new Set(sortedAPStations.map(s => s.code));
const sortedAllExcludingAP = sortedAllStations.filter(s => !apCodes.has(s.code));

const groupedStations = [
  { label: 'Andhra Pradesh Stations', options: sortedAPStations },
  { label: 'All Other Stations', options: sortedAllExcludingAP },
  { label: 'Other Major Cities', options: sortedRemainingMajorCities }
];

const SearchForm: React.FC<SearchFormProps> = ({ onSubmit, loading = false }) => {
  const [formData, setFormData] = useState<SearchFormData>({
    source: '',
    destination: '',
    travel_date: '',
    preference: 'fastest',
    train_id: ''
  });

  const [errors, setErrors] = useState<Partial<SearchFormData>>({});

  const handleInputChange = (field: keyof SearchFormData, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: undefined }));
    }
  };

  const validateForm = (): boolean => {
    const newErrors: Partial<SearchFormData> = {};

    // Flexible validation: allow any combination as long as at least one search method is provided
    const hasRouteIntent = formData.source && formData.destination;
    const hasTrainIntent = formData.train_id && formData.train_id.trim();

    // At least one search method must be selected
    if (!hasRouteIntent && !hasTrainIntent) {
      newErrors.source = 'Select stations or enter train ID';
      newErrors.destination = 'Select stations or enter train ID';
      newErrors.train_id = 'Select stations or enter train ID';
    } else {
      // If stations are selected, validate them
      if (hasRouteIntent && formData.source === formData.destination) {
        newErrors.destination = 'Source and destination must be different';
      }

      // If train ID is provided, validate it
      if (hasTrainIntent && formData.train_id && !/^\d{1,5}$/.test(formData.train_id)) {
        newErrors.train_id = 'Train ID must be up to 5 digits (e.g., 12951)';
      }
    }

    if (!formData.travel_date) newErrors.travel_date = 'Travel date is required';

    // Validate date is not in the past
    if (formData.travel_date) {
      const selectedDate = new Date(formData.travel_date);
      const today = new Date();
      today.setHours(0, 0, 0, 0);

      if (selectedDate < today) {
        newErrors.travel_date = 'Travel date cannot be in the past';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (validateForm()) {
      onSubmit(formData);
    }
  };

  const getMinDate = () => {
    const today = new Date();
    return today.toISOString().split('T')[0];
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl p-8 border border-gray-100">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          AI Train Delay Prediction
        </h1>
        <p className="text-gray-600">
          Get accurate delay predictions and find the best travel options
        </p>
        <p className="text-sm text-gray-500 mt-2">
          📍 {sortedAllStations.length.toLocaleString()} total stations available - Type to search any station!
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Source Station */}
          <div className="space-y-2">
            <label className="flex items-center text-sm font-medium text-gray-700">
              <MapPin className="w-4 h-4 mr-2 text-primary-500" />
              From
            </label>
            <SearchableSelect
              groupedOptions={groupedStations}
              value={formData.source}
              onChange={(value) => handleInputChange('source', value)}
              placeholder="Search and select source station..."
              disabled={loading}
              error={errors.source}
            />
          </div>

          {/* Destination Station */}
          <div className="space-y-2">
            <label className="flex items-center text-sm font-medium text-gray-700">
              <MapPin className="w-4 h-4 mr-2 text-primary-500" />
              To
            </label>
            <SearchableSelect
              groupedOptions={groupedStations}
              value={formData.destination}
              onChange={(value) => handleInputChange('destination', value)}
              placeholder="Search and select destination station..."
              disabled={loading}
              error={errors.destination}
              excludeValue={formData.source}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Travel Date */}
          <div className="space-y-2">
            <label className="flex items-center text-sm font-medium text-gray-700">
              <Calendar className="w-4 h-4 mr-2 text-primary-500" />
              Travel Date
            </label>
            <input
              type="date"
              value={formData.travel_date}
              onChange={(e) => handleInputChange('travel_date', e.target.value)}
              min={getMinDate()}
              className={`w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200 ${
                errors.travel_date ? 'border-red-300 bg-red-50' : 'border-gray-300'
              }`}
              disabled={loading}
            />
            {errors.travel_date && (
              <p className="text-red-500 text-sm">{errors.travel_date}</p>
            )}
          </div>

          {/* Preference */}
          <div className="space-y-2">
            <label className="flex items-center text-sm font-medium text-gray-700">
              <Settings className="w-4 h-4 mr-2 text-primary-500" />
              Preference
            </label>
            <select
              value={formData.preference}
              onChange={(e) => handleInputChange('preference', e.target.value as any)}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200"
              disabled={loading}
            >
              <option value="fastest">Fastest Journey</option>
              <option value="cheapest">Cheapest Fare</option>
              <option value="most_reliable">Most Reliable</option>
            </select>
          </div>
        </div>

        {/* Train ID Search */}
        <div className="space-y-2">
          <label className="flex items-center text-sm font-medium text-gray-700">
            <Train className="w-4 h-4 mr-2 text-primary-500" />
            Train ID (Optional - filters to specific train if you know the number)
          </label>
          <input
            type="text"
            value={formData.train_id}
            onChange={(e) => handleInputChange('train_id', e.target.value)}
            placeholder="Enter train ID (e.g., 12951)"
            className={`w-full px-4 py-3 border rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all duration-200 ${
              errors.train_id ? 'border-red-300 bg-red-50' : 'border-gray-300'
            }`}
            disabled={loading}
          />
          {errors.train_id && (
            <p className="text-red-500 text-sm mt-1">{errors.train_id}</p>
          )}
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className={`w-full py-4 px-6 rounded-xl font-semibold text-white transition-all duration-200 transform hover:scale-105 active:scale-95 ${
            loading
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-600 hover:to-primary-700 shadow-lg hover:shadow-xl'
          }`}
        >
          {loading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
              Predicting...
            </div>
          ) : (
            'Predict Delay & Get Recommendations'
          )}
        </button>
      </form>
    </div>
  );
};

export default SearchForm;
