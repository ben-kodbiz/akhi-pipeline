import { useState, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { qloraApi } from '../../services/qloraApi';

export const useModelLibrary = () => {
  const [uploadProgress, setUploadProgress] = useState(new Map());
  const [downloadProgress, setDownloadProgress] = useState(new Map());
  const queryClient = useQueryClient();

  // Fetch all models
  const {
    data: models = [],
    isLoading,
    error,
    refetch: refreshModels
  } = useQuery({
    queryKey: ['qlora-models'],
    queryFn: qloraApi.getModels,
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 60 * 1000 // Refetch every minute
  });

  // Fetch model details - this should be called at component level
  const getModelDetails = useCallback((modelId) => {
    // Return a function that can be used to fetch model details
    return () => qloraApi.getModel(modelId);
  }, []);

  // Upload model mutation
  const uploadModelMutation = useMutation({
    mutationFn: async ({ file, metadata, onProgress }) => {
      const formData = new FormData();
      formData.append('model_file', file);
      formData.append('metadata', JSON.stringify(metadata));
      
      return qloraApi.uploadModel(formData, (progressEvent) => {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setUploadProgress(prev => new Map(prev.set(file.name, progress)));
        if (onProgress) onProgress(progress);
      });
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries(['qlora-models']);
      toast.success(`Model "${data.name}" uploaded successfully!`);
      setUploadProgress(prev => {
        const newMap = new Map(prev);
        newMap.delete(data.original_filename);
        return newMap;
      });
    },
    onError: (error, variables) => {
      toast.error('Failed to upload model: ' + error.message);
      setUploadProgress(prev => {
        const newMap = new Map(prev);
        newMap.delete(variables.file.name);
        return newMap;
      });
    }
  });

  // Download model mutation
  const downloadModelMutation = useMutation({
    mutationFn: async ({ modelId, onProgress }) => {
      const response = await qloraApi.downloadModel(modelId, (progressEvent) => {
        const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        setDownloadProgress(prev => new Map(prev.set(modelId, progress)));
        if (onProgress) onProgress(progress);
      });
      
      return response;
    },
    onSuccess: (blob, variables) => {
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `model_${variables.modelId}.tar.gz`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast.success('Model downloaded successfully!');
      setDownloadProgress(prev => {
        const newMap = new Map(prev);
        newMap.delete(variables.modelId);
        return newMap;
      });
    },
    onError: (error, variables) => {
      toast.error('Failed to download model: ' + error.message);
      setDownloadProgress(prev => {
        const newMap = new Map(prev);
        newMap.delete(variables.modelId);
        return newMap;
      });
    }
  });

  // Delete model mutation
  const deleteModelMutation = useMutation({
    mutationFn: qloraApi.deleteModel,
    onSuccess: () => {
      queryClient.invalidateQueries(['qlora-models']);
      toast.success('Model deleted successfully!');
    },
    onError: (error) => {
      toast.error('Failed to delete model: ' + error.message);
    }
  });

  // Deploy model mutation
  const deployModelMutation = useMutation({
    mutationFn: ({ modelId, deploymentConfig }) => 
      qloraApi.deployModel(modelId, deploymentConfig),
    onSuccess: (data) => {
      queryClient.invalidateQueries(['qlora-models']);
      queryClient.invalidateQueries(['model-deployments']);
      toast.success(`Model deployed successfully! Endpoint: ${data.endpoint}`);
    },
    onError: (error) => {
      toast.error('Failed to deploy model: ' + error.message);
    }
  });

  // Undeploy model mutation
  const undeployModelMutation = useMutation({
    mutationFn: qloraApi.undeployModel,
    onSuccess: () => {
      queryClient.invalidateQueries(['qlora-models']);
      queryClient.invalidateQueries(['model-deployments']);
      toast.success('Model undeployed successfully!');
    },
    onError: (error) => {
      toast.error('Failed to undeploy model: ' + error.message);
    }
  });

  // Archive model mutation
  const archiveModelMutation = useMutation({
    mutationFn: qloraApi.archiveModel,
    onSuccess: () => {
      queryClient.invalidateQueries(['qlora-models']);
      toast.success('Model archived successfully!');
    },
    onError: (error) => {
      toast.error('Failed to archive model: ' + error.message);
    }
  });

  // Restore model mutation
  const restoreModelMutation = useMutation({
    mutationFn: qloraApi.restoreModel,
    onSuccess: () => {
      queryClient.invalidateQueries(['qlora-models']);
      toast.success('Model restored successfully!');
    },
    onError: (error) => {
      toast.error('Failed to restore model: ' + error.message);
    }
  });

  // Update model metadata mutation
  const updateModelMutation = useMutation({
    mutationFn: ({ modelId, metadata }) => qloraApi.updateModel(modelId, metadata),
    onSuccess: (data) => {
      queryClient.invalidateQueries(['qlora-models']);
      queryClient.invalidateQueries(['qlora-model', data.id]);
      toast.success('Model updated successfully!');
    },
    onError: (error) => {
      toast.error('Failed to update model: ' + error.message);
    }
  });

  // Get model metrics
  const getModelMetrics = useCallback(async (modelId) => {
    try {
      return await qloraApi.getModelMetrics(modelId);
    } catch (error) {
      toast.error('Failed to fetch model metrics: ' + error.message);
      throw error;
    }
  }, []);

  // Test model inference
  const testModelInference = useCallback(async (modelId, input) => {
    try {
      return await qloraApi.testModelInference(modelId, input);
    } catch (error) {
      toast.error('Failed to test model inference: ' + error.message);
      throw error;
    }
  }, []);

  // Compare models
  const compareModels = useCallback(async (modelIds) => {
    try {
      return await qloraApi.compareModels(modelIds);
    } catch (error) {
      toast.error('Failed to compare models: ' + error.message);
      throw error;
    }
  }, []);

  // Get model usage statistics
  const getModelUsage = useCallback(async (modelId, timeRange = '7d') => {
    try {
      return await qloraApi.getModelUsage(modelId, timeRange);
    } catch (error) {
      toast.error('Failed to fetch model usage: ' + error.message);
      throw error;
    }
  }, []);

  // Export model
  const exportModel = useCallback(async (modelId, format = 'onnx') => {
    try {
      const blob = await qloraApi.exportModel(modelId, format);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `model_${modelId}.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success(`Model exported as ${format.toUpperCase()}`);
    } catch (error) {
      toast.error('Failed to export model: ' + error.message);
    }
  }, []);

  // Clone model
  const cloneModel = useCallback(async (modelId, newName) => {
    try {
      const clonedModel = await qloraApi.cloneModel(modelId, newName);
      queryClient.invalidateQueries(['qlora-models']);
      toast.success(`Model cloned as "${clonedModel.name}"`);
      return clonedModel;
    } catch (error) {
      toast.error('Failed to clone model: ' + error.message);
      throw error;
    }
  }, [queryClient]);

  // Get model versions - this should be called at component level
  const getModelVersions = useCallback((modelId) => {
    // Return a function that can be used to fetch model versions
    return () => qloraApi.getModelVersions(modelId);
  }, []);

  // Create model version
  const createModelVersion = useCallback(async (modelId, versionData) => {
    try {
      const version = await qloraApi.createModelVersion(modelId, versionData);
      queryClient.invalidateQueries(['qlora-model-versions', modelId]);
      toast.success(`Model version ${version.version} created`);
      return version;
    } catch (error) {
      toast.error('Failed to create model version: ' + error.message);
      throw error;
    }
  }, [queryClient]);

  // Get deployment status
  const getDeploymentStatus = useCallback(async (modelId) => {
    try {
      return await qloraApi.getDeploymentStatus(modelId);
    } catch (error) {
      toast.error('Failed to fetch deployment status: ' + error.message);
      throw error;
    }
  }, []);

  // Batch operations
  const batchDeleteModels = useCallback(async (modelIds) => {
    try {
      const results = await Promise.allSettled(
        modelIds.map(id => qloraApi.deleteModel(id))
      );
      
      const successful = results.filter(r => r.status === 'fulfilled').length;
      const failed = results.filter(r => r.status === 'rejected').length;
      
      queryClient.invalidateQueries(['qlora-models']);
      
      if (failed === 0) {
        toast.success(`Successfully deleted ${successful} models`);
      } else {
        toast.warning(`Deleted ${successful} models, ${failed} failed`);
      }
      
      return { successful, failed };
    } catch (error) {
      toast.error('Batch delete failed: ' + error.message);
      throw error;
    }
  }, [queryClient]);

  const batchArchiveModels = useCallback(async (modelIds) => {
    try {
      const results = await Promise.allSettled(
        modelIds.map(id => qloraApi.archiveModel(id))
      );
      
      const successful = results.filter(r => r.status === 'fulfilled').length;
      const failed = results.filter(r => r.status === 'rejected').length;
      
      queryClient.invalidateQueries(['qlora-models']);
      
      if (failed === 0) {
        toast.success(`Successfully archived ${successful} models`);
      } else {
        toast.warning(`Archived ${successful} models, ${failed} failed`);
      }
      
      return { successful, failed };
    } catch (error) {
      toast.error('Batch archive failed: ' + error.message);
      throw error;
    }
  }, [queryClient]);

  // Helper functions
  const getModelsByStatus = useCallback((status) => {
    return models.filter(model => model.status === status);
  }, [models]);

  const getModelsByTag = useCallback((tag) => {
    return models.filter(model => model.tags && model.tags.includes(tag));
  }, [models]);

  const getTotalStorageUsed = useCallback(() => {
    return models.reduce((total, model) => total + (model.size || 0), 0);
  }, [models]);

  const getModelStatistics = useCallback(() => {
    const stats = {
      total: models.length,
      ready: 0,
      training: 0,
      deployed: 0,
      archived: 0,
      error: 0,
      totalSize: 0
    };
    
    models.forEach(model => {
      stats[model.status] = (stats[model.status] || 0) + 1;
      stats.totalSize += model.size || 0;
    });
    
    return stats;
  }, [models]);

  return {
    // Data
    models,
    uploadProgress,
    downloadProgress,
    
    // Loading states
    isLoading,
    
    // Errors
    error,
    
    // Actions
    uploadModel: uploadModelMutation.mutateAsync,
    downloadModel: downloadModelMutation.mutateAsync,
    deleteModel: deleteModelMutation.mutateAsync,
    deployModel: deployModelMutation.mutateAsync,
    undeployModel: undeployModelMutation.mutateAsync,
    archiveModel: archiveModelMutation.mutateAsync,
    restoreModel: restoreModelMutation.mutateAsync,
    updateModel: updateModelMutation.mutateAsync,
    
    // Utilities
    refreshModels,
    getModelDetails,
    getModelMetrics,
    testModelInference,
    compareModels,
    getModelUsage,
    exportModel,
    cloneModel,
    getModelVersions,
    createModelVersion,
    getDeploymentStatus,
    
    // Batch operations
    batchDeleteModels,
    batchArchiveModels,
    
    // Helper functions
    getModelsByStatus,
    getModelsByTag,
    getTotalStorageUsed,
    getModelStatistics,
    
    // Mutation states
    isUploading: uploadModelMutation.isPending,
    isDownloading: downloadModelMutation.isPending,
    isDeleting: deleteModelMutation.isPending,
    isDeploying: deployModelMutation.isPending,
    isUndeploying: undeployModelMutation.isPending,
    isArchiving: archiveModelMutation.isPending,
    isRestoring: restoreModelMutation.isPending,
    isUpdating: updateModelMutation.isPending
  };
};