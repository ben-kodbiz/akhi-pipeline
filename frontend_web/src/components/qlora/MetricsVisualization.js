import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, RadialBarChart, RadialBar } from 'recharts';
import { BarChart3, PieChart as PieChartIcon, Activity, Cpu, HardDrive, Zap } from 'lucide-react';
import { useTrainingMetrics } from '../../hooks/qlora/useTrainingMetrics';

const MetricsVisualization = ({ jobId }) => {
  const { metrics, isLoading, error } = useTrainingMetrics(jobId);
  const [activeTab, setActiveTab] = useState('performance');

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg p-6 border border-gray-200">
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-gray-200 rounded w-1/3"></div>
          <div className="h-32 bg-gray-200 rounded"></div>
          <div className="grid grid-cols-3 gap-4">
            <div className="h-16 bg-gray-200 rounded"></div>
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
          <BarChart3 className="w-8 h-8 mx-auto mb-2" />
          <p>Failed to load metrics visualization</p>
          <p className="text-sm mt-1">{error.message}</p>
        </div>
      </div>
    );
  }

  const resourceData = [
    {
      name: 'GPU',
      usage: metrics?.current?.gpu_usage || 0,
      color: '#10b981',
      icon: <Cpu className="w-4 h-4" />
    },
    {
      name: 'CPU',
      usage: metrics?.current?.cpu_usage || 0,
      color: '#3b82f6',
      icon: <Activity className="w-4 h-4" />
    },
    {
      name: 'Memory',
      usage: metrics?.current?.memory_usage || 0,
      color: '#f59e0b',
      icon: <HardDrive className="w-4 h-4" />
    }
  ];

  const performanceData = metrics?.performance_history || [];
  const lossDistribution = metrics?.loss_distribution || [];

  const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'];

  const tabs = [
    { id: 'performance', label: 'Performance', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'resources', label: 'Resources', icon: <Cpu className="w-4 h-4" /> },
    { id: 'distribution', label: 'Distribution', icon: <PieChartIcon className="w-4 h-4" /> }
  ];

  return (
    <div className="bg-white rounded-lg p-6 border border-gray-200 space-y-6">
      {/* Header with Tabs */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-blue-600" />
          Training Analytics
        </h3>
        <div className="flex bg-gray-100 rounded-lg p-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Tab Content */}
      <div className="min-h-[300px]">
        {activeTab === 'performance' && (
          <div className="space-y-4">
            <h4 className="font-medium text-gray-900">Training Performance Metrics</h4>
            <div className="h-64">
              {performanceData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis 
                      dataKey="epoch" 
                      stroke="#666"
                      fontSize={12}
                    />
                    <YAxis 
                      stroke="#666"
                      fontSize={12}
                    />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#fff',
                        border: '1px solid #e5e7eb',
                        borderRadius: '8px',
                        boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                      }}
                    />
                    <Bar dataKey="throughput" fill="#10b981" name="Samples/sec" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                  <div className="text-center text-gray-500">
                    <BarChart3 className="w-8 h-8 mx-auto mb-2" />
                    <p>No performance data available</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'resources' && (
          <div className="space-y-6">
            <h4 className="font-medium text-gray-900">Resource Utilization</h4>
            
            {/* Resource Usage Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {resourceData.map((resource, index) => (
                <div key={resource.name} className="bg-gradient-to-r from-gray-50 to-gray-100 p-4 rounded-lg border border-gray-200">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div style={{ color: resource.color }}>
                        {resource.icon}
                      </div>
                      <span className="font-medium text-gray-900">{resource.name}</span>
                    </div>
                    <span className="text-2xl font-bold" style={{ color: resource.color }}>
                      {resource.usage}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="h-2 rounded-full transition-all duration-300"
                      style={{ 
                        width: `${resource.usage}%`,
                        backgroundColor: resource.color
                      }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>

            {/* Radial Chart */}
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <RadialBarChart cx="50%" cy="50%" innerRadius="20%" outerRadius="80%" data={resourceData}>
                  <RadialBar 
                    dataKey="usage" 
                    cornerRadius={10} 
                    fill={(entry, index) => COLORS[index % COLORS.length]}
                  />
                  <Tooltip 
                    formatter={(value, name) => [`${value}%`, name]}
                    contentStyle={{
                      backgroundColor: '#fff',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px',
                      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                    }}
                  />
                </RadialBarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {activeTab === 'distribution' && (
          <div className="space-y-4">
            <h4 className="font-medium text-gray-900">Loss Distribution Analysis</h4>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Pie Chart */}
              <div className="h-64">
                {lossDistribution.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={lossDistribution}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {lossDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip 
                        formatter={(value, name) => [value.toFixed(4), name]}
                        contentStyle={{
                          backgroundColor: '#fff',
                          border: '1px solid #e5e7eb',
                          borderRadius: '8px',
                          boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
                    <div className="text-center text-gray-500">
                      <PieChartIcon className="w-8 h-8 mx-auto mb-2" />
                      <p>No distribution data available</p>
                    </div>
                  </div>
                )}
              </div>

              {/* Statistics */}
              <div className="space-y-4">
                <h5 className="font-medium text-gray-900">Training Statistics</h5>
                <div className="space-y-3">
                  <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg border border-green-200">
                    <span className="text-green-800 font-medium">Best Loss</span>
                    <span className="text-green-900 font-bold">
                      {metrics?.current?.best_loss?.toFixed(4) || 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-blue-50 rounded-lg border border-blue-200">
                    <span className="text-blue-800 font-medium">Average Loss</span>
                    <span className="text-blue-900 font-bold">
                      {metrics?.current?.avg_loss?.toFixed(4) || 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-purple-50 rounded-lg border border-purple-200">
                    <span className="text-purple-800 font-medium">Loss Variance</span>
                    <span className="text-purple-900 font-bold">
                      {metrics?.current?.loss_variance?.toFixed(6) || 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between items-center p-3 bg-orange-50 rounded-lg border border-orange-200">
                    <span className="text-orange-800 font-medium">Convergence Rate</span>
                    <span className="text-orange-900 font-bold">
                      {metrics?.current?.convergence_rate || 'Calculating...'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Real-time Status */}
      <div className="bg-gray-50 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm font-medium text-gray-900">Live Metrics</span>
          </div>
          <div className="text-sm text-gray-600">
            Last updated: {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetricsVisualization;