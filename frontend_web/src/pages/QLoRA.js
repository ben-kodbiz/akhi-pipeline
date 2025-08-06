import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, Settings, Database, BarChart3, 
  Cpu, Zap, AlertCircle, CheckCircle,
  RefreshCw, Bell, HelpCircle
} from 'lucide-react';
import { useQLoRATraining } from '../hooks/qlora/useQLoRATraining';
import { useModelLibrary } from '../hooks/qlora/useModelLibrary';
import TrainingDashboard from '../components/qlora/TrainingDashboard';
import ProgressMonitor from '../components/qlora/ProgressMonitor';
import MetricsVisualization from '../components/qlora/MetricsVisualization';
import ConfigurationPanel from '../components/qlora/ConfigurationPanel';
import ModelLibrary from '../components/qlora/ModelLibrary';
import toast from 'react-hot-toast';

const QLoRA = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [showNotifications, setShowNotifications] = useState(false);
  const [systemStatus, setSystemStatus] = useState('healthy');
  
  const {
    trainingJobs,
    isLoadingJobs,
    isConnected,
    getCurrentTrainingStatus,
    getTrainingStatistics
  } = useQLoRATraining();
  
  const {
    models,
    isLoading: isLoadingModels,
    getModelStatistics
  } = useModelLibrary();

  // Get current status and statistics
  const trainingStatus = getCurrentTrainingStatus();
  const trainingStats = getTrainingStatistics();
  const modelStats = getModelStatistics();

  // Check system health
  useEffect(() => {
    const checkSystemHealth = () => {
      if (!isConnected) {
        setSystemStatus('disconnected');
      } else if (trainingStatus.failedJobs > 0) {
        setSystemStatus('warning');
      } else {
        setSystemStatus('healthy');
      }
    };

    checkSystemHealth();
  }, [isConnected, trainingStatus.failedJobs]);

  // Show notifications for important events
  useEffect(() => {
    if (trainingStatus.hasActiveJobs && trainingStats?.lossTrend === 'increasing') {
      toast.error('Training loss is increasing. Consider adjusting parameters.');
    }
  }, [trainingStatus.hasActiveJobs, trainingStats?.lossTrend]);

  const tabs = [
    {
      id: 'dashboard',
      name: 'Training Dashboard',
      icon: Play,
      description: 'Monitor and manage training jobs'
    },
    {
      id: 'models',
      name: 'Model Library',
      icon: Database,
      description: 'Manage trained models and deployments'
    },
    {
      id: 'config',
      name: 'Configuration',
      icon: Settings,
      description: 'Training parameters and settings'
    },
    {
      id: 'metrics',
      name: 'Analytics',
      icon: BarChart3,
      description: 'Training metrics and performance'
    }
  ];

  const getStatusIcon = () => {
    switch (systemStatus) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'warning':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      case 'disconnected':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <RefreshCw className="w-5 h-5 text-gray-500 animate-spin" />;
    }
  };

  const getStatusText = () => {
    switch (systemStatus) {
      case 'healthy':
        return 'System Healthy';
      case 'warning':
        return 'System Warning';
      case 'disconnected':
        return 'Disconnected';
      default:
        return 'Checking...';
    }
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return (
          <div className="space-y-6">
            <TrainingDashboard />
            {trainingStatus.hasActiveJobs && (
              <ProgressMonitor jobId={trainingStatus.activeJobs[0]?.id} />
            )}
          </div>
        );
      case 'models':
        return <ModelLibrary />;
      case 'config':
        return (
          <ConfigurationPanel 
            onSave={(config) => {
              console.log('Configuration saved:', config);
              toast.success('Configuration saved successfully!');
            }}
          />
        );
      case 'metrics':
        return (
          <div className="space-y-6">
            <MetricsVisualization />
            {/* Additional analytics components can be added here */}
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Title and Status */}
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Zap className="w-8 h-8 text-blue-600" />
                <div>
                  <h1 className="text-xl font-bold text-gray-900">QLoRA Training</h1>
                  <p className="text-sm text-gray-500">Islamic AI Model Fine-tuning</p>
                </div>
              </div>
              
              <div className="flex items-center space-x-2 px-3 py-1 bg-gray-50 rounded-full">
                {getStatusIcon()}
                <span className="text-sm font-medium text-gray-700">
                  {getStatusText()}
                </span>
              </div>
            </div>

            {/* Stats and Actions */}
            <div className="flex items-center space-x-6">
              {/* Quick Stats */}
              <div className="hidden md:flex items-center space-x-6 text-sm">
                <div className="text-center">
                  <div className="font-semibold text-gray-900">
                    {trainingStatus.activeJobsCount}
                  </div>
                  <div className="text-gray-500">Active Jobs</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-gray-900">
                    {modelStats.total || 0}
                  </div>
                  <div className="text-gray-500">Models</div>
                </div>
                <div className="text-center">
                  <div className="font-semibold text-gray-900">
                    {modelStats.deployed || 0}
                  </div>
                  <div className="text-gray-500">Deployed</div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex items-center space-x-3">
                <button
                  onClick={() => setShowNotifications(!showNotifications)}
                  className="relative p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100"
                >
                  <Bell className="w-5 h-5" />
                  {trainingStatus.failedJobs > 0 && (
                    <span className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full"></span>
                  )}
                </button>
                
                <button className="p-2 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100">
                  <HelpCircle className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                    isActive
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{tab.name}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
          >
            {renderTabContent()}
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Notifications Panel */}
      <AnimatePresence>
        {showNotifications && (
          <motion.div
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
            className="fixed top-16 right-4 w-80 bg-white rounded-lg shadow-lg border border-gray-200 z-50"
          >
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold text-gray-900">Notifications</h3>
                <button
                  onClick={() => setShowNotifications(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ×
                </button>
              </div>
            </div>
            <div className="p-4 space-y-3 max-h-96 overflow-y-auto">
              {trainingStatus.failedJobs > 0 && (
                <div className="flex items-start space-x-3 p-3 bg-red-50 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-500 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-red-800">
                      Training Jobs Failed
                    </p>
                    <p className="text-sm text-red-600">
                      {trainingStatus.failedJobs} training job(s) have failed. Check logs for details.
                    </p>
                  </div>
                </div>
              )}
              
              {!isConnected && (
                <div className="flex items-start space-x-3 p-3 bg-yellow-50 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-yellow-500 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-yellow-800">
                      Connection Lost
                    </p>
                    <p className="text-sm text-yellow-600">
                      Real-time updates are unavailable. Trying to reconnect...
                    </p>
                  </div>
                </div>
              )}
              
              {trainingStatus.hasActiveJobs && isConnected && (
                <div className="flex items-start space-x-3 p-3 bg-blue-50 rounded-lg">
                  <Cpu className="w-5 h-5 text-blue-500 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-blue-800">
                      Training in Progress
                    </p>
                    <p className="text-sm text-blue-600">
                      {trainingStatus.activeJobsCount} training job(s) are currently running.
                    </p>
                  </div>
                </div>
              )}
              
              {trainingStatus.totalJobs === 0 && (
                <div className="text-center py-8">
                  <Bell className="w-8 h-8 text-gray-300 mx-auto mb-2" />
                  <p className="text-sm text-gray-500">No notifications</p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Loading Overlay */}
      {(isLoadingJobs || isLoadingModels) && (
        <div className="fixed inset-0 bg-black bg-opacity-20 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 flex items-center space-x-3">
            <RefreshCw className="w-5 h-5 animate-spin text-blue-600" />
            <span className="text-gray-700">Loading QLoRA data...</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default QLoRA;