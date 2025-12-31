import React from 'react';
import DelayContributorsChart from './DelayContributorsChart';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  contributions?: { [k: string]: number } | null;
  top_contributors?: string[] | null;
}

const DelayContributorsModal: React.FC<Props> = ({ isOpen, onClose, title = 'Delay Contributors', contributions, top_contributors }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="fixed inset-0 bg-black opacity-40" onClick={onClose}></div>
      <div className="bg-white rounded-xl shadow-xl p-6 z-50 w-11/12 max-w-md">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold">{title}</h3>
            {top_contributors && top_contributors.length > 0 && (
              <div className="text-sm text-gray-600 mt-1">Top contributors: {top_contributors.slice(0,3).join(', ')}</div>
            )}
          </div>
          <button className="text-gray-500 hover:text-gray-700" onClick={onClose}>✕</button>
        </div>

        <div className="mt-2">
          {contributions && Object.keys(contributions).length ? (
            <DelayContributorsChart data={contributions} />
          ) : (
            <p className="text-sm text-gray-500">No contributor data available.</p>
          )}
        </div>

        <div className="mt-4 text-right">
          <button className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
};

export default DelayContributorsModal;
