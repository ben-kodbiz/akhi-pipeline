import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Play, Pause, Square, Settings, Activity, Clock, Cpu } from 'lucide-react';
import { useQLoRATraining } from '../../hooks/qlora/useQLoRATraining';
import ProgressMonitor from './ProgressMonitor';
import MetricsVisualization from './MetricsVisualization';
import toast from 'react-hot-toast';

const TrainingDashboard = () => {
  const { trainingJobs, isLoading, error, startTraining, stopTraining, refetchJobs } = useQLoRATraining();
  const [selectedJob, setSelectedJob] = useState(null);

  useEffect(() => {
    // Refresh jobs every 5 seconds
    const interval = setInterval(refetchJobs, 5000);
    return () => clearInterval(interval);
  }, [refetchJobs]);

  const handleStartTraining = async (config) => {
    try {
      await startTraining(config);
      toast.success('Training started successfully!');
    } catch (error) {
      toast.error('Failed to start training: ' + error.message);
    }
  };

  const handleStopTraining = async (jobId) => {
    try {
      await stopTraining(jobId);
      toast.success('Training stopped successfully!');
    } catch (error) {
      toast.error('Failed to stop training: ' + error.message);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'text-green-600 bg-green-100';
      case 'completed': return 'text-blue-600 bg-blue-100';
      case 'failed': return 'text-red-600 bg-red-100';
      case 'paused': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running': return <Activity className="w-4 h-4" />;
      case 'completed': return <Square className="w-4 h-4" />;
      case 'failed': return <Square className="w-4 h-4" />;
      case 'paused': return <Pause className="w-4 h-4" />;
      default: return <Clock className="w-4 h-4" />;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800">Error loading training jobs: {error.message}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">QLoRA Training Dashboard</h2>
          <p className="text-gray-600 mt-1">Monitor and manage your Islamic AI model training</p>
        </div>
        <button
          onClick={() => setSelectedJob('new')}
          className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg flex items-center gap-2 transition-colors"
        >
          <Play className="w-4 h-4" />
          Start New Training
        </button>
      </div>

      {/* Training Jobs Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {trainingJobs.map((job) => (
          <motion.div
            key={job.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-lg shadow-md border border-gray-200 overflow-hidden hover:shadow-lg transition-shadow"
          >
            {/* Job Header */}
            <div className="p-4 border-b border-gray-100">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-gray-900 truncate">{job.name}</h3>
                <span className={`px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1 ${getStatusColor(job.status)}`}>
                  {getStatusIcon(job.status)}
                  {job.status}
                </span>
              </div>
              <p className="text-sm text-gray-600 mt-1">{job.model_name}</p>
            </div>

            {/* Job Details */}
            <div className="p-4 space-y-3">
              {/* Progress Bar */}
              {job.status === 'running' && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Progress</span>
                    <span className="font-medium">{job.progress}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${job.progress}%` }}
                    ></div>
                  </div>
                </div>
              )}

              {/* Metrics */}
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-600">Epoch</span>
                  <p className="font-medium">{job.current_epoch}/{job.total_epochs}</p>
                </div>
                <div>
                  <span className="text-gray-600">Loss</span>
                  <p className="font-medium">{job.current_loss?.toFixed(4) || 'N/A'}</p>
                </div>
                <div>
                  <span className="text-gray-600">ETA</span>
                  <p className="font-medium">{job.eta || 'Calculating...'}</p>
                </div>
                <div>
                  <span className="text-gray-600">GPU Usage</span>
                  <p className="font-medium flex items-center gap-1">
                    <Cpu className="w-3 h-3" />
                    {job.gpu_usage || 0}%
                  </p>
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-2 pt-2">
                {job.status === 'running' && (
                  <button
                    onClick={() => handleStopTraining(job.id)}
                    className="flex-1 bg-red-600 hover:bg-red-700 text-white px-3 py-2 rounded text-sm transition-colors"
                  >
                    Stop
                  </button>
                )}
                <button
                  onClick={() => setSelectedJob(job.id)}
                  className="flex-1 bg-gray-600 hover:bg-gray-700 text-white px-3 py-2 rounded text-sm transition-colors flex items-center justify-center gap-1"
                >
                  <Settings className="w-3 h-3" />
                  Details
                </button>
              </div>
            </div>
          </motion.div>
        ))}

        {/* Empty State */}
        {trainingJobs.length === 0 && (
          <div className="col-span-full bg-gray-50 border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
            <Activity className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Training Jobs</h3>
            <p className="text-gray-600 mb-4">Start your first QLoRA training to see it here</p>
            <button
              onClick={() => setSelectedJob('new')}
              className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              Start Training
            </button>
          </div>
        )}
      </div>

      {/* Detailed View Modal/Panel */}
      {selectedJob && selectedJob !== 'new' && (
        <div className="mt-8">
          <div className="bg-white rounded-lg shadow-lg border border-gray-200">
            <div className="p-6 border-b border-gray-200">
              <h3 className="text-xl font-semibold text-gray-900">Training Details</h3>
            </div>
            <div className="p-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ProgressMonitor jobId={selectedJob} />
                <MetricsVisualization jobId={selectedJob} />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingDashboard;