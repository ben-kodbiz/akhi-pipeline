import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Activity, Zap, Clock, TrendingDown } from 'lucide-react';
import { useTrainingMetrics } from '../../hooks/qlora/useTrainingMetrics';

const ProgressMonitor = ({ jobId }) => {
  const { metrics, isLoading, error } = useTrainingMetrics(jobId);
  const [selectedMetric, setSelectedMetric] = useState('loss');

  const formatTime = (seconds) => {
    if (!seconds) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hours}h ${minutes}m ${secs}s`;
  };

  const formatNumber = (num) => {
    if (num === null || num === undefined) return 'N/A';
    return typeof num === 'number' ? num.toFixed(4) : num;
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg p-6 border border-gray-200">
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-gray-200 rounded w-1/3"></div>
          <div className="h-32 bg-gray-200 rounded"></div>
          <div className="grid grid-cols-2 gap-4">
            <div className="h-16 bg-gray-200 rounded"></div>
            <div className="h-16 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-white rounded-lg p-6 border border-red-200">
        <div className="text-red-600 text-center">
          <Activity className="w-8 h-8 mx-auto mb-2" />
          <p>Failed to load training metrics</p>
          <p className="text-sm mt-1">{error.message}</p>
        </div>
      </div>
    );
  }

  const chartData = metrics?.history || [];
  const currentMetrics = metrics?.current || {};

  return (
    <div className="bg-white rounded-lg p-6 border border-gray-200 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <Activity className="w-5 h-5 text-green-600" />
          Training Progress
        </h3>
        <div className="flex gap-2">
          <button
            onClick={() => setSelectedMetric('loss')}
            className={`px-3 py-1 rounded text-sm transition-colors ${
              selectedMetric === 'loss'
                ? 'bg-green-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            Loss
          </button>
          <button
            onClick={() => setSelectedMetric('accuracy')}
            className={`px-3 py-1 rounded text-sm transition-colors ${
              selectedMetric === 'accuracy'
                ? 'bg-green-600 text-white'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            Accuracy
          </button>
        </div>
      </div>

      {/* Current Metrics Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gradient-to-r from-green-50 to-green-100 p-4 rounded-lg border border-green-200">
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className="w-4 h-4 text-green-600" />
            <span className="text-sm font-medium text-green-800">Current Loss</span>
          </div>
          <p className="text-2xl font-bold text-green-900">{formatNumber(currentMetrics.loss)}</p>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-blue-100 p-4 rounded-lg border border-blue-200">
          <div className="flex items-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-800">Learning Rate</span>
          </div>
          <p className="text-2xl font-bold text-blue-900">{formatNumber(currentMetrics.learning_rate)}</p>
        </div>

        <div className="bg-gradient-to-r from-purple-50 to-purple-100 p-4 rounded-lg border border-purple-200">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-4 h-4 text-purple-600" />
            <span className="text-sm font-medium text-purple-800">Epoch</span>
          </div>
          <p className="text-2xl font-bold text-purple-900">
            {currentMetrics.epoch || 0}/{currentMetrics.total_epochs || 0}
          </p>
        </div>

        <div className="bg-gradient-to-r from-orange-50 to-orange-100 p-4 rounded-lg border border-orange-200">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-4 h-4 text-orange-600" />
            <span className="text-sm font-medium text-orange-800">ETA</span>
          </div>
          <p className="text-lg font-bold text-orange-900">{formatTime(currentMetrics.eta)}</p>
        </div>
      </div>

      {/* Progress Chart */}
      <div className="space-y-4">
        <h4 className="font-medium text-gray-900 capitalize">{selectedMetric} Over Time</h4>
        <div className="h-64">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height="100%">
              {selectedMetric === 'loss' ? (
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis 
                    dataKey="step" 
                    stroke="#666"
                    fontSize={12}
                  />
                  <YAxis 
                    stroke="#666"
                    fontSize={12}
                    tickFormatter={(value) => value.toFixed(3)}
                  />
                  <Tooltip 
                    formatter={(value) => [value.toFixed(4), 'Loss']}
                    labelFormatter={(label) => `Step ${label}`}
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="#dc2626" 
                    fill="#fef2f2"
                    strokeWidth={2}
                  />
                </AreaChart>
              ) : (
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis 
                    dataKey="step" 
                    stroke="#666"
                    fontSize={12}
                  />
                  <YAxis 
                    stroke="#666"
                    fontSize={12}
                    tickFormatter={(value) => `${(value * 100).toFixed(1)}%`}
                  />
                  <Tooltip 
                    formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Accuracy']}
                    labelFormatter={(label) => `Step ${label}`}
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="accuracy" 
                    stroke="#16a34a" 
                    strokeWidth={2}
                    dot={{ fill: '#16a34a', strokeWidth: 2, r: 3 }}
                  />
                </LineChart>
              )}
            </ResponsiveContainer>
          ) : (
            <div className="h-full flex items-center justify-center bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
              <div className="text-center text-gray-500">
                <Activity className="w-8 h-8 mx-auto mb-2" />
                <p>No training data available yet</p>
                <p className="text-sm">Metrics will appear once training starts</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Training Statistics */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-medium text-gray-900 mb-3">Training Statistics</h4>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-600">Total Steps</span>
            <p className="font-medium text-gray-900">{currentMetrics.total_steps || 0}</p>
          </div>
          <div>
            <span className="text-gray-600">Samples/sec</span>
            <p className="font-medium text-gray-900">{formatNumber(currentMetrics.samples_per_second)}</p>
          </div>
          <div>
            <span className="text-gray-600">GPU Memory</span>
            <p className="font-medium text-gray-900">{currentMetrics.gpu_memory || 'N/A'}</p>
          </div>
          <div>
            <span className="text-gray-600">Best Loss</span>
            <p className="font-medium text-gray-900">{formatNumber(currentMetrics.best_loss)}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProgressMonitor;