import React, { useState, useRef, useEffect } from 'react';
import { Search, ChevronDown, X } from 'lucide-react';

interface Station {
  code: string;
  name: string;
}

interface OptionGroup {
  label: string;
  options: Station[];
}

interface SearchableSelectProps {
  options?: Station[];
  groupedOptions?: OptionGroup[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  error?: string;
  excludeValue?: string;
}

const SearchableSelect: React.FC<SearchableSelectProps> = ({
  options,
  groupedOptions,
  value,
  onChange,
  placeholder = 'Search and select station...',
  disabled = false,
  error,
  excludeValue
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredOptions, setFilteredOptions] = useState<Station[]>([]);
  const [filteredGroups, setFilteredGroups] = useState<OptionGroup[]>([]);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Initialize filtered options when component mounts or options change
  useEffect(() => {
    const hasGroups = Array.isArray(groupedOptions) && groupedOptions.length > 0;
    if (hasGroups) {
      const groups = groupedOptions as OptionGroup[];
      // Flatten for display but keep groups
      let flattened: Station[] = [];
      groups.forEach(g => {
        const groupOptions = g.options || [];
        const opts = excludeValue ? groupOptions.filter(opt => opt.code !== excludeValue) : groupOptions;
        flattened = flattened.concat(opts);
      });
      const displayCount = Math.min(flattened.length, 1000);
      setFilteredOptions(flattened.slice(0, displayCount));
      setFilteredGroups(groups.map(g => ({
        label: g.label,
        options: excludeValue ? g.options.filter(o => o.code !== excludeValue) : g.options
      })));
    } else if (options && options.length > 0) {
      let initial = [...options];
      if (excludeValue) {
        initial = initial.filter(opt => opt.code !== excludeValue);
      }
      // Show first 1000 stations initially (all are searchable)
      const displayCount = Math.min(initial.length, 1000);
      setFilteredOptions(initial.slice(0, displayCount));
      console.log(`SearchableSelect initialized with ${displayCount} of ${initial.length} stations (total: ${options.length})`);
    } else {
      console.warn('SearchableSelect: No options provided');
      setFilteredOptions([]);
    }
  }, [options, groupedOptions, excludeValue]);

  // Debounce the search term for performance with large datasets
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState(searchTerm);
  const [isSearching, setIsSearching] = useState(false);

  // Debounce search term changes to avoid filtering on every keystroke
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearchTerm(searchTerm);
      setIsSearching(false);
    }, 150); // 150ms debounce delay

    setIsSearching(true); // Show searching state during debounce
    return () => clearTimeout(timer);
  }, [searchTerm]);

  // Filter options based on debounced search term
  useEffect(() => {
    if ((!options || options.length === 0) && (!groupedOptions || groupedOptions.length === 0)) {
      console.warn('No options provided to SearchableSelect');
      setFilteredOptions([]);
      return;
    }

    const hasGroups = Array.isArray(groupedOptions) && groupedOptions.length > 0;
    let filtered: Station[] = [];
    let groupsToDisplay: OptionGroup[] = [];
    if (hasGroups) {
      const groups = groupedOptions as OptionGroup[];
      // Search within each group
      groupsToDisplay = groups.map(g => ({
        label: g.label,
        options: (g.options || []).filter(opt => {
          if (excludeValue && opt.code === excludeValue) return false;
          if (!debouncedSearchTerm) return true;
          const term = debouncedSearchTerm.toLowerCase();
          return opt.name.toLowerCase().includes(term) || opt.code.toLowerCase().includes(term);
        })
      })).filter(g => g.options.length > 0);
      filtered = groupsToDisplay.flatMap(g => g.options);
      setFilteredGroups(groupsToDisplay);
    } else {
      filtered = [...(options || [])]; // Create a copy
      // Exclude the excludeValue if provided
      if (excludeValue) {
        filtered = filtered.filter(opt => opt.code !== excludeValue);
      }

      // Filter by debounced search term
      if (debouncedSearchTerm) {
        const term = debouncedSearchTerm.toLowerCase();
        filtered = filtered.filter(opt =>
          opt.name.toLowerCase().includes(term) ||
          opt.code.toLowerCase().includes(term)
        );
      }
    }

    // Exclude the excludeValue if provided
    if (excludeValue) {
      filtered = filtered.filter(opt => opt.code !== excludeValue);
    }

    // Filter by debounced search term
    if (debouncedSearchTerm) {
      const term = debouncedSearchTerm.toLowerCase();
      filtered = filtered.filter(opt =>
        opt.name.toLowerCase().includes(term) ||
        opt.code.toLowerCase().includes(term)
      );
    }

    // Add custom option if search term entered but no matches found
    if (debouncedSearchTerm && filtered.length === 0) {
      filtered = [{ code: '__CUSTOM__', name: `Use custom: ${debouncedSearchTerm}`, }];
    }

    // Show all filtered results (up to 1000 for performance, but all are searchable)
    const displayCount = Math.min(filtered.length, 1000);
    const toDisplay = filtered.slice(0, displayCount);
    setFilteredOptions(toDisplay);
    setIsSearching(false); // Done searching

    // Debug log when dropdown is open
    if (isOpen && debouncedSearchTerm) {
      console.log(`SearchableSelect: Search "${debouncedSearchTerm}" found ${filtered.length} stations, showing ${displayCount}`);
    }
  }, [debouncedSearchTerm, options, groupedOptions, excludeValue, isOpen]);

  // Get selected station name
  const allOpts = groupedOptions && groupedOptions.length > 0 ? groupedOptions.flatMap(g => g.options) : options || [];
  const selectedStation = allOpts.find(opt => opt.code === value);
  const displayValue = selectedStation ? `${selectedStation.name} (${selectedStation.code})` : value;

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setSearchTerm('');
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleSelect = (code: string) => {
    if (code === '__CUSTOM__') {
      // Custom entry - use the search term as station code
      onChange(searchTerm);
    } else {
      onChange(code);
    }
    setIsOpen(false);
    setSearchTerm('');
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    onChange('');
    setSearchTerm('');
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <div
        className={`w-full px-4 py-3 border rounded-xl focus-within:ring-2 focus-within:ring-primary-500 focus-within:border-transparent transition-all duration-200 cursor-pointer ${
          error ? 'border-red-300 bg-red-50' : 'border-gray-300 bg-white'
        } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
        onClick={() => !disabled && setIsOpen(!isOpen)}
      >
        <div className="flex items-center justify-between">
          {value ? (
            <span className="text-gray-900">{displayValue}</span>
          ) : (
            <span className="text-gray-400">{placeholder}</span>
          )}
          <div className="flex items-center space-x-2">
            {value && !disabled && (
              <X
                className="w-4 h-4 text-gray-400 hover:text-gray-600"
                onClick={handleClear}
              />
            )}
            <ChevronDown
              className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'transform rotate-180' : ''}`}
            />
          </div>
        </div>
      </div>

      {isOpen && !disabled && (
        <div className="absolute z-50 w-full mt-1 bg-white border border-gray-300 rounded-xl shadow-lg max-h-80 overflow-hidden">
          <div className="p-2 border-b border-gray-200 sticky top-0 bg-white">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                ref={inputRef}
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Type to search station..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                autoFocus
              />
            </div>
            {isSearching && searchTerm && (
              <div className="flex items-center text-xs text-blue-600 mt-1">
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-600 mr-2"></div>
                Searching...
              </div>
            )}
            {!isSearching && searchTerm && (
              <p className="text-xs text-gray-500 mt-1">
                {filteredOptions.length} station{filteredOptions.length !== 1 ? 's' : ''} found
              </p>
            )}
          </div>
          <div className="overflow-y-auto max-h-80">
            {(filteredGroups && filteredGroups.length > 0) ? (
              <>
                {filteredGroups.map((group) => (
                  <div key={group.label} className="border-b border-gray-100">
                    <div className="px-4 py-2 bg-gray-50 text-xs font-medium text-gray-600">{group.label}</div>
                    {group.options.map((option) => {
                      const isCustomOption = option.code === '__CUSTOM__';
                      return (
                        <div
                          key={option.code}
                          onClick={() => handleSelect(option.code)}
                          className={`px-4 py-2 hover:bg-primary-50 cursor-pointer transition-colors border-b border-gray-100 ${
                            isCustomOption ? 'bg-orange-50 hover:bg-orange-100' : value === option.code ? 'bg-primary-100 font-semibold' : ''
                          }`}
                        >
                          {isCustomOption ? (
                            <>
                              <div className="text-orange-700 font-medium">✨ Use custom station name</div>
                              <div className="text-xs text-orange-600">Station: {option.name.replace('Use custom: ', '')}</div>
                            </>
                          ) : (
                            <>
                              <div className="text-gray-900 font-medium">{option.name}</div>
                              <div className="text-xs text-gray-500">Code: {option.code}</div>
                            </>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ))}
              </>
            ) : filteredOptions.length > 0 ? (
              <>
                {filteredOptions.map((option) => {
                  const isCustomOption = option.code === '__CUSTOM__';
                  return (
                    <div
                      key={option.code}
                      onClick={() => handleSelect(option.code)}
                      className={`px-4 py-2 hover:bg-primary-50 cursor-pointer transition-colors border-b border-gray-100 ${
                        isCustomOption ? 'bg-orange-50 hover:bg-orange-100' : value === option.code ? 'bg-primary-100 font-semibold' : ''
                      }`}
                    >
                      {isCustomOption ? (
                        <>
                          <div className="text-orange-700 font-medium">✨ Use custom station name</div>
                          <div className="text-xs text-orange-600">Station: {option.name.replace('Use custom: ', '')}</div>
                        </>
                      ) : (
                        <>
                          <div className="text-gray-900 font-medium">{option.name}</div>
                          <div className="text-xs text-gray-500">Code: {option.code}</div>
                        </>
                      )}
                    </div>
                  );
                })}
              </>
            ) : (
              <div className="px-4 py-8 text-center text-gray-500">
                {searchTerm ? 'No stations found. Try a different search term.' : 'Loading stations...'}
              </div>
            )}
          </div>
          {!searchTerm && (
            <div className="px-4 py-2 bg-blue-50 border-t border-gray-200 text-xs text-blue-700 text-center">
              Showing first 1,000 of {allOpts.length.toLocaleString()} stations. <strong>Type to search all stations!</strong>
            </div>
          )}
          {searchTerm && filteredOptions.length >= 1000 && (
            <div className="px-4 py-2 bg-yellow-50 border-t border-gray-200 text-xs text-yellow-700 text-center">
              Showing first 1,000 of {filteredOptions.length.toLocaleString()} results. Refine your search for more specific results.
            </div>
          )}
          {searchTerm && filteredOptions.length > 0 && filteredOptions.length < 1000 && (
            <div className="px-4 py-2 bg-green-50 border-t border-gray-200 text-xs text-green-700 text-center">
              Found {filteredOptions.length} station{filteredOptions.length !== 1 ? 's' : ''}
            </div>
          )}
        </div>
      )}

      {error && (
        <p className="text-red-500 text-sm mt-1">{error}</p>
      )}
    </div>
  );
};

export default SearchableSelect;
