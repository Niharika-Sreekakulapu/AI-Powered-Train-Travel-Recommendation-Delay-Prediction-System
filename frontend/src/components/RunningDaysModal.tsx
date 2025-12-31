import React from 'react';

interface DayItem {
  day: number;
  name: string;
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
  days: DayItem[];
}

const RunningDaysModal: React.FC<Props> = ({ isOpen, onClose, days }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="fixed inset-0 bg-black opacity-40" onClick={onClose}></div>
      <div className="bg-white rounded-xl shadow-xl p-6 z-50 w-11/12 max-w-md">
        <h3 className="text-lg font-semibold mb-4">Available Running Days</h3>
        {days && days.length ? (
          <ul className="space-y-2">
            {days.map((d) => (
              <li key={d.day} className="flex items-center justify-between p-2 border rounded">
                <span>{d.name}</span>
                <span className="text-sm text-gray-500">Day {d.day}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-gray-600">No running days found for this route.</p>
        )}
        <div className="mt-4 text-right">
          <button
            className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700"
            onClick={onClose}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default RunningDaysModal;