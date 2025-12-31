import React from 'react';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  risk?: {
    risk_score?: number;
    confidence?: string;
    advice?: string;
    breakdown?: { [k: string]: number | string };
  } | null;
}

const RiskBreakdownModal: React.FC<Props> = ({ isOpen, onClose, title = 'Risk Details', risk }) => {
  if (!isOpen) return null;

  const breakdown = risk?.breakdown || {};

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="fixed inset-0 bg-black opacity-40" onClick={onClose}></div>
      <div className="bg-white rounded-xl shadow-xl p-6 z-50 w-11/12 max-w-md">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold">{title}</h3>
            <div className="text-sm text-gray-600">Score: <span className="font-semibold">{risk?.risk_score ?? 'N/A'}</span> • Confidence: <span className="font-semibold">{risk?.confidence ?? 'N/A'}</span></div>
            {risk?.advice && <div className="mt-2 text-sm text-gray-700">{risk.advice}</div>}
          </div>
          <button className="text-gray-500 hover:text-gray-700" onClick={onClose}>✕</button>
        </div>

        <div className="mt-2">
          <h4 className="text-sm font-semibold mb-2">Breakdown</h4>
          {Object.keys(breakdown).length ? (
            <ul className="space-y-2 text-sm text-gray-700">
              {Object.entries(breakdown).map(([k, v]) => (
                <li key={k} className="flex justify-between border-b pb-2">
                  <span className="text-gray-600">{k}</span>
                  <span className="font-semibold">{String(v)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-500">No detailed breakdown available.</p>
          )}
        </div>

        <div className="mt-4 text-right">
          <button className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
};

export default RiskBreakdownModal;
