import { useState, useEffect, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { qloraApi } from '../../services/qloraApi';

export const useTrainingStatus = (jobId) => {
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);
  const [eta, setEta] = useState(null);
  const [metrics, setMetrics] = useState({
    loss: 0,
    accuracy: 0,
    learningRate: 0
  });

  // Fetch training status
  const { data: trainingData, isLoading, error } = useQuery({
    queryKey: ['training-status', jobId],
    queryFn: () => qloraApi.getTrainingJob(jobId),
    enabled: !!jobId,
    refetchInterval: status === 'running' ? 2000 : false,
    staleTime: 1000
  });

  // Update local state when data changes
  useEffect(() => {
    if (trainingData) {
      setStatus(trainingData.status || 'idle');
      setProgress(trainingData.progress || 0);
      setCurrentEpoch(trainingData.current_epoch || 0);
      setTotalEpochs(trainingData.total_epochs || 0);
      setEta(trainingData.eta || null);
      
      if (trainingData.metrics) {
        setMetrics({
          loss: trainingData.metrics.loss || 0,
          accuracy: trainingData.metrics.accuracy || 0,
          learningRate: trainingData.metrics.learning_rate || 0
        });
      }
    }
  }, [trainingData]);

  // Calculate progress percentage
  const progressPercentage = useCallback(() => {
    if (totalEpochs === 0) return 0;
    return Math.round((currentEpoch / totalEpochs) * 100);
  }, [currentEpoch, totalEpochs]);

  // Format ETA
  const formatEta = useCallback(() => {
    if (!eta) return 'Unknown';
    
    const hours = Math.floor(eta / 3600);
    const minutes = Math.floor((eta % 3600) / 60);
    const seconds = Math.floor(eta % 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${seconds}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    } else {
      return `${seconds}s`;
    }
  }, [eta]);

  // Check if training is active
  const isActive = useCallback(() => {
    return ['running', 'starting', 'resuming'].includes(status);
  }, [status]);

  // Check if training is completed
  const isCompleted = useCallback(() => {
    return status === 'completed';
  }, [status]);

  // Check if training has failed
  const hasFailed = useCallback(() => {
    return status === 'failed';
  }, [status]);

  // Get status color for UI
  const getStatusColor = useCallback(() => {
    switch (status) {
      case 'running':
      case 'starting':
      case 'resuming':
        return 'blue';
      case 'completed':
        return 'green';
      case 'failed':
      case 'error':
        return 'red';
      case 'paused':
      case 'stopped':
        return 'yellow';
      default:
        return 'gray';
    }
  }, [status]);

  // Get status icon
  const getStatusIcon = useCallback(() => {
    switch (status) {
      case 'running':
        return 'play';
      case 'starting':
      case 'resuming':
        return 'refresh';
      case 'completed':
        return 'check';
      case 'failed':
      case 'error':
        return 'x';
      case 'paused':
      case 'stopped':
        return 'pause';
      default:
        return 'circle';
    }
  }, [status]);

  return {
    // Status data
    status,
    progress,
    currentEpoch,
    totalEpochs,
    eta,
    metrics,
    
    // Loading states
    isLoading,
    error,
    
    // Computed values
    progressPercentage: progressPercentage(),
    formattedEta: formatEta(),
    
    // Status checks
    isActive: isActive(),
    isCompleted: isCompleted(),
    hasFailed: hasFailed(),
    
    // UI helpers
    statusColor: getStatusColor(),
    statusIcon: getStatusIcon(),
    
    // Raw data
    trainingData
  };
};

export default useTrainingStatus;