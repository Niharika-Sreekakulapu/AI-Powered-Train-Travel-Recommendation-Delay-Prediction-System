import React from 'react';

const LoadingSpinner = ({ 
  message = 'Loading...', 
  size = 'md' 
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12'
  };

  return (
    <div className="flex flex-col items-center justify-center p-8">
      <div className={`animate-spin rounded-full border-4 border-gray-200 border-t-primary-500 ${sizeClasses[size]}`}></div>
      <p className="mt-4 text-gray-600 font-medium">{message}</p>
    </div>
  );
};

export default LoadingSpinner;
