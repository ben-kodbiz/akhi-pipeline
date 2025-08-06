import { useState, useEffect, useCallback, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { qloraApi } from '../../services/qloraApi';

export const useTrainingMetrics = (jobId, options = {}) => {
  const {
    realTime = true,
    maxDataPoints = 1000,
    refreshInterval = 2000,
    enableAggregation = true
  } = options;

  const [metricsHistory, setMetricsHistory] = useState({
    loss: [],
    accuracy: [],
    learning_rate: [],
    gradient_norm: [],
    throughput: [],
    memory_usage: [],
    gpu_utilization: []
  });

  const [aggregatedMetrics, setAggregatedMetrics] = useState({
    loss: { min: null, max: null, avg: null, current: null },
    accuracy: { min: null, max: null, avg: null, current: null },
    learning_rate: { min: null, max: null, avg: null, current: null },
    gradient_norm: { min: null, max: null, avg: null, current: null },
    throughput: { min: null, max: null, avg: null, current: null },
    memory_usage: { min: null, max: null, avg: null, current: null },
    gpu_utilization: { min: null, max: null, avg: null, current: null }
  });

  const [performanceStats, setPerformanceStats] = useState({
    totalSteps: 0,
    currentEpoch: 0,
    totalEpochs: 0,
    trainingTime: 0,
    estimatedTimeRemaining: null,
    averageStepTime: null,
    stepsPerSecond: null
  });

  // Fetch training metrics
  const {
    data: currentMetrics,
    isLoading,
    error
  } = useQuery({
    queryKey: ['training-metrics', jobId],
    queryFn: () => qloraApi.getTrainingMetrics(jobId),
    enabled: !!jobId && realTime,
    refetchInterval: realTime ? refreshInterval : false,
    staleTime: 1000
  });

  // Update metrics history when new data arrives
  useEffect(() => {
    if (!currentMetrics) return;

    setMetricsHistory(prev => {
      const newHistory = { ...prev };
      
      Object.keys(currentMetrics).forEach(metric => {
        if (Array.isArray(currentMetrics[metric])) {
          // Handle array data (time series)
          newHistory[metric] = [
            ...prev[metric],
            ...currentMetrics[metric]
          ].slice(-maxDataPoints); // Keep only last N points
        } else if (typeof currentMetrics[metric] === 'number') {
          // Handle single value data
          const timestamp = Date.now();
          newHistory[metric] = [
            ...prev[metric],
            { timestamp, value: currentMetrics[metric], step: currentMetrics.step || 0 }
          ].slice(-maxDataPoints);
        }
      });
      
      return newHistory;
    });

    // Update performance stats
    if (currentMetrics.step !== undefined) {
      setPerformanceStats(prev => ({
        ...prev,
        totalSteps: currentMetrics.step,
        currentEpoch: currentMetrics.epoch || prev.currentEpoch,
        totalEpochs: currentMetrics.total_epochs || prev.totalEpochs,
        trainingTime: currentMetrics.training_time || prev.trainingTime,
        estimatedTimeRemaining: currentMetrics.eta || prev.estimatedTimeRemaining,
        averageStepTime: currentMetrics.avg_step_time || prev.averageStepTime,
        stepsPerSecond: currentMetrics.steps_per_second || prev.stepsPerSecond
      }));
    }
  }, [currentMetrics, maxDataPoints]);

  // Calculate aggregated metrics
  useEffect(() => {
    if (!enableAggregation) return;

    const newAggregated = {};
    
    Object.keys(metricsHistory).forEach(metric => {
      const data = metricsHistory[metric];
      if (data.length === 0) {
        newAggregated[metric] = { min: null, max: null, avg: null, current: null };
        return;
      }

      const values = data.map(point => point.value || point).filter(v => typeof v === 'number');
      if (values.length === 0) {
        newAggregated[metric] = { min: null, max: null, avg: null, current: null };
        return;
      }

      newAggregated[metric] = {
        min: Math.min(...values),
        max: Math.max(...values),
        avg: values.reduce((sum, val) => sum + val, 0) / values.length,
        current: values[values.length - 1]
      };
    });
    
    setAggregatedMetrics(newAggregated);
  }, [metricsHistory, enableAggregation]);

  // Get metrics for specific time range
  const getMetricsForTimeRange = useCallback((metric, startTime, endTime) => {
    const data = metricsHistory[metric] || [];
    return data.filter(point => {
      const timestamp = point.timestamp || point.step;
      return timestamp >= startTime && timestamp <= endTime;
    });
  }, [metricsHistory]);

  // Get metrics for specific step range
  const getMetricsForStepRange = useCallback((metric, startStep, endStep) => {
    const data = metricsHistory[metric] || [];
    return data.filter(point => {
      const step = point.step || 0;
      return step >= startStep && step <= endStep;
    });
  }, [metricsHistory]);

  // Calculate moving average
  const getMovingAverage = useCallback((metric, windowSize = 10) => {
    const data = metricsHistory[metric] || [];
    if (data.length < windowSize) return data;

    const result = [];
    for (let i = windowSize - 1; i < data.length; i++) {
      const window = data.slice(i - windowSize + 1, i + 1);
      const avg = window.reduce((sum, point) => sum + (point.value || point), 0) / windowSize;
      result.push({
        ...data[i],
        value: avg,
        original_value: data[i].value || data[i]
      });
    }
    return result;
  }, [metricsHistory]);

  // Calculate trend (slope of linear regression)
  const getTrend = useCallback((metric, windowSize = 50) => {
    const data = metricsHistory[metric] || [];
    if (data.length < 2) return 0;

    const recentData = data.slice(-windowSize);
    const n = recentData.length;
    
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    
    recentData.forEach((point, index) => {
      const x = index;
      const y = point.value || point;
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumXX += x * x;
    });
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    return slope;
  }, [metricsHistory]);

  // Detect anomalies using z-score
  const detectAnomalies = useCallback((metric, threshold = 2) => {
    const data = metricsHistory[metric] || [];
    if (data.length < 10) return [];

    const values = data.map(point => point.value || point);
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);

    return data.filter(point => {
      const value = point.value || point;
      const zScore = Math.abs((value - mean) / stdDev);
      return zScore > threshold;
    });
  }, [metricsHistory]);

  // Get performance insights
  const getPerformanceInsights = useCallback(() => {
    const insights = [];
    
    // Loss trend analysis
    const lossTrend = getTrend('loss', 20);
    if (lossTrend > 0.001) {
      insights.push({
        type: 'warning',
        metric: 'loss',
        message: 'Loss is increasing. Consider reducing learning rate.',
        severity: 'medium'
      });
    } else if (lossTrend < -0.001) {
      insights.push({
        type: 'success',
        metric: 'loss',
        message: 'Loss is decreasing steadily. Training is progressing well.',
        severity: 'low'
      });
    }
    
    // Learning rate analysis
    const currentLR = aggregatedMetrics.learning_rate?.current;
    if (currentLR && currentLR < 1e-6) {
      insights.push({
        type: 'warning',
        metric: 'learning_rate',
        message: 'Learning rate is very low. Training might be too slow.',
        severity: 'medium'
      });
    }
    
    // GPU utilization analysis
    const avgGPUUtil = aggregatedMetrics.gpu_utilization?.avg;
    if (avgGPUUtil && avgGPUUtil < 0.5) {
      insights.push({
        type: 'info',
        metric: 'gpu_utilization',
        message: 'GPU utilization is low. Consider increasing batch size.',
        severity: 'low'
      });
    }
    
    // Memory usage analysis
    const maxMemory = aggregatedMetrics.memory_usage?.max;
    if (maxMemory && maxMemory > 0.9) {
      insights.push({
        type: 'error',
        metric: 'memory_usage',
        message: 'Memory usage is very high. Risk of out-of-memory errors.',
        severity: 'high'
      });
    }
    
    // Gradient norm analysis
    const gradientAnomalies = detectAnomalies('gradient_norm', 3);
    if (gradientAnomalies.length > 0) {
      insights.push({
        type: 'warning',
        metric: 'gradient_norm',
        message: 'Gradient norm spikes detected. Consider gradient clipping.',
        severity: 'medium'
      });
    }
    
    return insights;
  }, [aggregatedMetrics, getTrend, detectAnomalies]);

  // Export metrics data
  const exportMetrics = useCallback((format = 'json') => {
    const exportData = {
      jobId,
      timestamp: new Date().toISOString(),
      metricsHistory,
      aggregatedMetrics,
      performanceStats,
      insights: getPerformanceInsights()
    };
    
    let content, filename, mimeType;
    
    switch (format) {
      case 'csv':
        // Convert to CSV format
        const csvRows = [];
        const headers = ['timestamp', 'step', 'metric', 'value'];
        csvRows.push(headers.join(','));
        
        Object.keys(metricsHistory).forEach(metric => {
          metricsHistory[metric].forEach(point => {
            csvRows.push([
              point.timestamp || '',
              point.step || '',
              metric,
              point.value || point
            ].join(','));
          });
        });
        
        content = csvRows.join('\n');
        filename = `training_metrics_${jobId}.csv`;
        mimeType = 'text/csv';
        break;
        
      case 'json':
      default:
        content = JSON.stringify(exportData, null, 2);
        filename = `training_metrics_${jobId}.json`;
        mimeType = 'application/json';
        break;
    }
    
    const blob = new Blob([content], { type: mimeType });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }, [jobId, metricsHistory, aggregatedMetrics, performanceStats, getPerformanceInsights]);

  // Memoized computed values
  const computedMetrics = useMemo(() => {
    return {
      totalDataPoints: Object.values(metricsHistory).reduce((sum, data) => sum + data.length, 0),
      metricsAvailable: Object.keys(metricsHistory).filter(key => metricsHistory[key].length > 0),
      latestTimestamp: Math.max(
        ...Object.values(metricsHistory)
          .flat()
          .map(point => point.timestamp || 0)
          .filter(t => t > 0)
      ),
      trainingProgress: performanceStats.totalEpochs > 0 
        ? (performanceStats.currentEpoch / performanceStats.totalEpochs) * 100 
        : 0
    };
  }, [metricsHistory, performanceStats]);

  return {
    // Raw data
    metricsHistory,
    aggregatedMetrics,
    performanceStats,
    currentMetrics,
    
    // Computed values
    computedMetrics,
    
    // Loading states
    isLoading,
    error,
    
    // Analysis functions
    getMetricsForTimeRange,
    getMetricsForStepRange,
    getMovingAverage,
    getTrend,
    detectAnomalies,
    getPerformanceInsights,
    
    // Utilities
    exportMetrics,
    
    // Control functions
    clearHistory: () => setMetricsHistory({
      loss: [],
      accuracy: [],
      learning_rate: [],
      gradient_norm: [],
      throughput: [],
      memory_usage: [],
      gpu_utilization: []
    })
  };
};