'use client';

import { useEffect } from 'react';

export default function ErrorBoundary({
  error,
  reset,
}: {
  error: Error;
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Error:', error);
  }, [error]);

  return (
    <div className="p-4 rounded-lg bg-red-50 border border-red-200">
      <h2 className="text-lg font-semibold text-red-800">Something went wrong!</h2>
      <button
        className="mt-4 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        onClick={reset}
      >
        Try again
      </button>
    </div>
  );
}
