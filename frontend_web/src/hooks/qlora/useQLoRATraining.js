import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import io from 'socket.io-client';
import toast from 'react-hot-toast';
import { qloraApi } from '../../services/qloraApi';

export const useQLoRATraining = () => {
  const [socket, setSocket] = useState(null);
  const [trainingMetrics, setTrainingMetrics] = useState({
    loss: [],
    accuracy: [],
    learning_rate: [],
    epoch: 0,
    step: 0,
    eta: null,
    throughput: null
  });
  const [isConnected, setIsConnected] = useState(false);
  const queryClient = useQueryClient();

  // Initialize WebSocket connection
  useEffect(() => {
    const socketInstance = io(process.env.REACT_APP_WS_URL || 'ws://localhost:8000', {
      path: '/ws/qlora',
      transports: ['websocket']
    });

    socketInstance.on('connect', () => {
      setIsConnected(true);
      console.log('Connected to QLoRA training WebSocket');
    });

    socketInstance.on('disconnect', () => {
      setIsConnected(false);
      console.log('Disconnected from QLoRA training WebSocket');
    });

    socketInstance.on('training_progress', (data) => {
      setTrainingMetrics(prev => ({
        ...prev,
        loss: [...prev.loss.slice(-99), { step: data.step, value: data.loss }],
        accuracy: [...prev.accuracy.slice(-99), { step: data.step, value: data.accuracy }],
        learning_rate: [...prev.learning_rate.slice(-99), { step: data.step, value: data.learning_rate }],
        epoch: data.epoch,
        step: data.step,
        eta: data.eta,
        throughput: data.throughput
      }));
    });

    socketInstance.on('training_status', (data) => {
      queryClient.invalidateQueries(['training-jobs']);
      
      if (data.status === 'completed') {
        toast.success(`Training job ${data.job_id} completed successfully!`);
      } else if (data.status === 'failed') {
        toast.error(`Training job ${data.job_id} failed: ${data.error}`);
      }
    });

    socketInstance.on('error', (error) => {
      console.error('WebSocket error:', error);
      toast.error('Connection error: ' + error.message);
    });

    setSocket(socketInstance);

    return () => {
      socketInstance.disconnect();
    };
  }, [queryClient]);

  // Fetch training jobs
  const {
    data: trainingJobs = [],
    isLoading: isLoadingJobs,
    error: jobsError
  } = useQuery({
    queryKey: ['training-jobs'],
    queryFn: qloraApi.getTrainingJobs,
    refetchInterval: 5000, // Refetch every 5 seconds
    staleTime: 1000 // Consider data stale after 1 second
  });

  // Fetch training job details - this should be called at component level
  const getTrainingJob = useCallback((jobId) => {
    // Return a function that can be used to fetch job details
    return () => qloraApi.getTrainingJob(jobId);
  }, []);

  // Start training mutation
  const startTrainingMutation = useMutation({
    mutationFn: qloraApi.startTraining,
    onSuccess: (data) => {
      queryClient.invalidateQueries(['training-jobs']);
      toast.success(`Training job ${data.job_id} started successfully!`);
      
      // Join the training room for real-time updates
      if (socket) {
        socket.emit('join_training', { job_id: data.job_id });
      }
    },
    onError: (error) => {
      toast.error('Failed to start training: ' + error.message);
    }
  });

  // Stop training mutation
  const stopTrainingMutation = useMutation({
    mutationFn: qloraApi.stopTraining,
    onSuccess: (data) => {
      queryClient.invalidateQueries(['training-jobs']);
      toast.success(`Training job ${data.job_id} stopped successfully!`);
      
      // Leave the training room
      if (socket) {
        socket.emit('leave_training', { job_id: data.job_id });
      }
    },
    onError: (error) => {
      toast.error('Failed to stop training: ' + error.message);
    }
  });

  // Resume training mutation
  const resumeTrainingMutation = useMutation({
    mutationFn: qloraApi.resumeTraining,
    onSuccess: (data) => {
      queryClient.invalidateQueries(['training-jobs']);
      toast.success(`Training job ${data.job_id} resumed successfully!`);
      
      // Rejoin the training room
      if (socket) {
        socket.emit('join_training', { job_id: data.job_id });
      }
    },
    onError: (error) => {
      toast.error('Failed to resume training: ' + error.message);
    }
  });

  // Delete training job mutation
  const deleteTrainingMutation = useMutation({
    mutationFn: qloraApi.deleteTrainingJob,
    onSuccess: () => {
      queryClient.invalidateQueries(['training-jobs']);
      toast.success('Training job deleted successfully!');
    },
    onError: (error) => {
      toast.error('Failed to delete training job: ' + error.message);
    }
  });

  // Get training logs
  const getTrainingLogs = useCallback(async (jobId, lines = 100) => {
    try {
      return await qloraApi.getTrainingLogs(jobId, lines);
    } catch (error) {
      toast.error('Failed to fetch training logs: ' + error.message);
      throw error;
    }
  }, []);

  // Export training results
  const exportTrainingResults = useCallback(async (jobId, format = 'json') => {
    try {
      const blob = await qloraApi.exportTrainingResults(jobId, format);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `training_results_${jobId}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success('Training results exported successfully!');
    } catch (error) {
      toast.error('Failed to export training results: ' + error.message);
    }
  }, []);

  // Join training room for real-time updates
  const joinTrainingRoom = useCallback((jobId) => {
    if (socket && isConnected) {
      socket.emit('join_training', { job_id: jobId });
    }
  }, [socket, isConnected]);

  // Leave training room
  const leaveTrainingRoom = useCallback((jobId) => {
    if (socket && isConnected) {
      socket.emit('leave_training', { job_id: jobId });
    }
  }, [socket, isConnected]);

  // Get current training status
  const getCurrentTrainingStatus = useCallback(() => {
    const activeJobs = trainingJobs.filter(job => 
      ['running', 'starting', 'resuming'].includes(job.status)
    );
    
    return {
      hasActiveJobs: activeJobs.length > 0,
      activeJobsCount: activeJobs.length,
      activeJobs,
      totalJobs: trainingJobs.length,
      completedJobs: trainingJobs.filter(job => job.status === 'completed').length,
      failedJobs: trainingJobs.filter(job => job.status === 'failed').length
    };
  }, [trainingJobs]);

  // Calculate training statistics
  const getTrainingStatistics = useCallback(() => {
    if (trainingMetrics.loss.length === 0) return null;

    const latestLoss = trainingMetrics.loss[trainingMetrics.loss.length - 1]?.value;
    const latestAccuracy = trainingMetrics.accuracy[trainingMetrics.accuracy.length - 1]?.value;
    const latestLR = trainingMetrics.learning_rate[trainingMetrics.learning_rate.length - 1]?.value;

    // Calculate loss trend (last 10 points)
    const recentLoss = trainingMetrics.loss.slice(-10).map(point => point.value);
    const lossTrend = recentLoss.length > 1 
      ? recentLoss[recentLoss.length - 1] - recentLoss[0]
      : 0;

    return {
      currentLoss: latestLoss,
      currentAccuracy: latestAccuracy,
      currentLearningRate: latestLR,
      currentEpoch: trainingMetrics.epoch,
      currentStep: trainingMetrics.step,
      eta: trainingMetrics.eta,
      throughput: trainingMetrics.throughput,
      lossTrend: lossTrend < 0 ? 'decreasing' : lossTrend > 0 ? 'increasing' : 'stable',
      totalDataPoints: trainingMetrics.loss.length
    };
  }, [trainingMetrics]);

  return {
    // Data
    trainingJobs,
    trainingMetrics,
    
    // Loading states
    isLoadingJobs,
    isConnected,
    
    // Errors
    jobsError,
    
    // Actions
    startTraining: startTrainingMutation.mutateAsync,
    stopTraining: stopTrainingMutation.mutateAsync,
    resumeTraining: resumeTrainingMutation.mutateAsync,
    deleteTraining: deleteTrainingMutation.mutateAsync,
    
    // Utilities
    getTrainingJob,
    getTrainingLogs,
    exportTrainingResults,
    joinTrainingRoom,
    leaveTrainingRoom,
    getCurrentTrainingStatus,
    getTrainingStatistics,
    
    // Mutation states
    isStarting: startTrainingMutation.isPending,
    isStopping: stopTrainingMutation.isPending,
    isResuming: resumeTrainingMutation.isPending,
    isDeleting: deleteTrainingMutation.isPending
  };
};