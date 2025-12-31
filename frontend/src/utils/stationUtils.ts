import stationMapping from './stationMapping.json';
let stationMappingData: any = stationMapping;
try {
  // Prefer a strict AP-specific mapping if available
  // eslint-disable-next-line @typescript-eslint/no-var-requires
  const strict = require('./stationMapping_ap_strict.json');
  if (strict && Array.isArray(strict.stations) && strict.stations.length > 0) {
    stationMappingData = strict;
  }
} catch (e) {
  // No strict mapping available; continue with full mapping
}

/**
 * Get station name from station code
 */
export const getStationName = (code: string): string => {
  if (!code) return code;
  const station = stationMappingData.stations.find((s: any) => s.code === code);
  return station ? station.name : code;
};

/**
 * Get station code from station name
 */
export const getStationCode = (name: string): string => {
  if (!name) return name;
  const station = stationMapping.stations.find((s: any) => s.name === name);
  return station ? station.code : name;
};

/**
 * Format station display: "Station Name (CODE)"
 */
export const formatStation = (code: string): string => {
  const name = getStationName(code);
  return name !== code ? `${name} (${code})` : code;
};

