import React from 'react';
import { AlertCircle, X } from 'lucide-react';

interface ErrorAlertProps {
  message: string;
  onClose?: () => void;
  type?: 'error' | 'warning' | 'info';
}

const ErrorAlert: React.FC<ErrorAlertProps> = ({ 
  message, 
  onClose, 
  type = 'error' 
}) => {
  const getAlertStyles = () => {
    switch (type) {
      case 'error':
        return 'bg-red-50 border-red-200 text-red-800';
      case 'warning':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      case 'info':
        return 'bg-blue-50 border-blue-200 text-blue-800';
      default:
        return 'bg-red-50 border-red-200 text-red-800';
    }
  };

  const getIconColor = () => {
    switch (type) {
      case 'error':
        return 'text-red-500';
      case 'warning':
        return 'text-yellow-500';
      case 'info':
        return 'text-blue-500';
      default:
        return 'text-red-500';
    }
  };

  return (
    <div className={`rounded-xl border p-4 ${getAlertStyles()} animate-slide-up`}>
      <div className="flex items-start">
        <AlertCircle className={`w-5 h-5 mr-3 mt-0.5 ${getIconColor()}`} />
        <div className="flex-1">
          <p className="font-medium">{message}</p>
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className={`ml-3 ${getIconColor()} hover:opacity-70 transition-opacity`}
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
};

export default ErrorAlert;