import { useState, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { qloraApi } from '../../services/qloraApi';

export const useQLoRAConfig = () => {
  const [validationCache, setValidationCache] = useState(new Map());
  const queryClient = useQueryClient();

  // Predefined configuration templates
  const templates = {
    islamic_qa: {
      model_name: 'islamic-qa-model',
      base_model: 'microsoft/DialoGPT-medium',
      lora_r: 16,
      lora_alpha: 32,
      lora_dropout: 0.1,
      lora_target_modules: ['q_proj', 'v_proj'],
      num_epochs: 3,
      batch_size: 4,
      learning_rate: 0.0002,
      warmup_steps: 100,
      weight_decay: 0.01,
      gradient_accumulation_steps: 4,
      max_grad_norm: 1.0,
      save_steps: 500,
      eval_steps: 500,
      logging_steps: 10,
      enable_content_validation: true,
      validation_threshold: 0.85,
      islamic_keywords_weight: 1.3
    },
    hadith_study: {
      model_name: 'hadith-study-model',
      base_model: 'microsoft/DialoGPT-large',
      lora_r: 32,
      lora_alpha: 64,
      lora_dropout: 0.05,
      lora_target_modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
      num_epochs: 5,
      batch_size: 2,
      learning_rate: 0.0001,
      warmup_steps: 200,
      weight_decay: 0.005,
      gradient_accumulation_steps: 8,
      max_grad_norm: 0.5,
      save_steps: 250,
      eval_steps: 250,
      logging_steps: 5,
      enable_content_validation: true,
      validation_threshold: 0.9,
      islamic_keywords_weight: 1.5
    },
    quran_tafsir: {
      model_name: 'quran-tafsir-model',
      base_model: 'Qwen/Qwen2.5-3B-Instruct',
      lora_r: 64,
      lora_alpha: 128,
      lora_dropout: 0.1,
      lora_target_modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
      num_epochs: 4,
      batch_size: 1,
      learning_rate: 0.00005,
      warmup_steps: 300,
      weight_decay: 0.01,
      gradient_accumulation_steps: 16,
      max_grad_norm: 1.0,
      save_steps: 100,
      eval_steps: 100,
      logging_steps: 1,
      enable_content_validation: true,
      validation_threshold: 0.95,
      islamic_keywords_weight: 2.0
    },
    general_islamic: {
      model_name: 'general-islamic-model',
      base_model: 'microsoft/DialoGPT-medium',
      lora_r: 24,
      lora_alpha: 48,
      lora_dropout: 0.1,
      lora_target_modules: ['q_proj', 'v_proj', 'k_proj'],
      num_epochs: 3,
      batch_size: 4,
      learning_rate: 0.0003,
      warmup_steps: 150,
      weight_decay: 0.01,
      gradient_accumulation_steps: 4,
      max_grad_norm: 1.0,
      save_steps: 400,
      eval_steps: 400,
      logging_steps: 20,
      enable_content_validation: true,
      validation_threshold: 0.8,
      islamic_keywords_weight: 1.2
    }
  };

  // Fetch saved configurations
  const {
    data: savedConfigs = [],
    isLoading: isLoadingConfigs,
    error: configsError
  } = useQuery({
    queryKey: ['qlora-configs'],
    queryFn: qloraApi.getConfigurations,
    staleTime: 5 * 60 * 1000 // 5 minutes
  });

  // Fetch configuration by ID - this should be called at component level
  const getConfiguration = useCallback((configId) => {
    // Return a function that can be used to fetch configuration
    return () => qloraApi.getConfiguration(configId);
  }, []);

  // Save configuration mutation
  const saveConfigMutation = useMutation({
    mutationFn: qloraApi.saveConfiguration,
    onSuccess: (data) => {
      queryClient.invalidateQueries(['qlora-configs']);
      toast.success(`Configuration "${data.name}" saved successfully!`);
    },
    onError: (error) => {
      toast.error('Failed to save configuration: ' + error.message);
    }
  });

  // Update configuration mutation
  const updateConfigMutation = useMutation({
    mutationFn: ({ configId, config }) => qloraApi.updateConfiguration(configId, config),
    onSuccess: (data) => {
      queryClient.invalidateQueries(['qlora-configs']);
      queryClient.invalidateQueries(['qlora-config', data.id]);
      toast.success(`Configuration "${data.name}" updated successfully!`);
    },
    onError: (error) => {
      toast.error('Failed to update configuration: ' + error.message);
    }
  });

  // Delete configuration mutation
  const deleteConfigMutation = useMutation({
    mutationFn: qloraApi.deleteConfiguration,
    onSuccess: () => {
      queryClient.invalidateQueries(['qlora-configs']);
      toast.success('Configuration deleted successfully!');
    },
    onError: (error) => {
      toast.error('Failed to delete configuration: ' + error.message);
    }
  });

  // Validate configuration
  const validateConfig = useCallback(async (config) => {
    const configKey = JSON.stringify(config);
    
    // Check cache first
    if (validationCache.has(configKey)) {
      return validationCache.get(configKey);
    }

    try {
      const result = await qloraApi.validateConfiguration(config);
      
      // Cache the result
      setValidationCache(prev => {
        const newCache = new Map(prev);
        newCache.set(configKey, result);
        
        // Limit cache size to 50 entries
        if (newCache.size > 50) {
          const firstKey = newCache.keys().next().value;
          newCache.delete(firstKey);
        }
        
        return newCache;
      });
      
      return result;
    } catch (error) {
      const errorResult = {
        valid: false,
        messages: [error.message || 'Validation failed'],
        warnings: [],
        suggestions: []
      };
      
      setValidationCache(prev => {
        const newCache = new Map(prev);
        newCache.set(configKey, errorResult);
        return newCache;
      });
      
      return errorResult;
    }
  }, [validationCache]);

  // Get optimal configuration suggestions
  const getOptimalConfig = useCallback(async (requirements) => {
    try {
      const suggestions = await qloraApi.getOptimalConfiguration(requirements);
      return suggestions;
    } catch (error) {
      toast.error('Failed to get optimal configuration: ' + error.message);
      throw error;
    }
  }, []);

  // Estimate training time and resources
  const estimateTraining = useCallback(async (config) => {
    try {
      const estimation = await qloraApi.estimateTraining(config);
      return estimation;
    } catch (error) {
      toast.error('Failed to estimate training: ' + error.message);
      throw error;
    }
  }, []);

  // Export configuration
  const exportConfig = useCallback((config, format = 'json') => {
    try {
      let content, filename, mimeType;
      
      switch (format) {
        case 'yaml':
          // Simple YAML export (you might want to use a proper YAML library)
          content = Object.entries(config)
            .map(([key, value]) => {
              if (Array.isArray(value)) {
                return `${key}:\n${value.map(v => `  - ${v}`).join('\n')}`;
              }
              return `${key}: ${value}`;
            })
            .join('\n');
          filename = `qlora_config_${config.model_name || 'unnamed'}.yaml`;
          mimeType = 'text/yaml';
          break;
        case 'json':
        default:
          content = JSON.stringify(config, null, 2);
          filename = `qlora_config_${config.model_name || 'unnamed'}.json`;
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
      
      toast.success(`Configuration exported as ${format.toUpperCase()}`);
    } catch (error) {
      toast.error('Failed to export configuration: ' + error.message);
    }
  }, []);

  // Import configuration
  const importConfig = useCallback((file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      
      reader.onload = (e) => {
        try {
          const content = e.target.result;
          let config;
          
          if (file.name.endsWith('.json')) {
            config = JSON.parse(content);
          } else if (file.name.endsWith('.yaml') || file.name.endsWith('.yml')) {
            // Simple YAML parsing (you might want to use a proper YAML library)
            config = {};
            content.split('\n').forEach(line => {
              const [key, ...valueParts] = line.split(':');
              if (key && valueParts.length > 0) {
                const value = valueParts.join(':').trim();
                if (value.startsWith('[') && value.endsWith(']')) {
                  config[key.trim()] = JSON.parse(value);
                } else if (!isNaN(value)) {
                  config[key.trim()] = parseFloat(value);
                } else if (value === 'true' || value === 'false') {
                  config[key.trim()] = value === 'true';
                } else {
                  config[key.trim()] = value;
                }
              }
            });
          } else {
            throw new Error('Unsupported file format. Please use JSON or YAML.');
          }
          
          toast.success('Configuration imported successfully!');
          resolve(config);
        } catch (error) {
          toast.error('Failed to import configuration: ' + error.message);
          reject(error);
        }
      };
      
      reader.onerror = () => {
        const error = new Error('Failed to read file');
        toast.error(error.message);
        reject(error);
      };
      
      reader.readAsText(file);
    });
  }, []);

  // Get configuration recommendations based on dataset
  const getDatasetRecommendations = useCallback(async (datasetPath) => {
    try {
      const recommendations = await qloraApi.analyzeDataset(datasetPath);
      return recommendations;
    } catch (error) {
      toast.error('Failed to analyze dataset: ' + error.message);
      throw error;
    }
  }, []);

  // Clear validation cache
  const clearValidationCache = useCallback(() => {
    setValidationCache(new Map());
    toast.success('Validation cache cleared');
  }, []);

  // Get configuration history - this should be called at component level
  const getConfigHistory = useCallback((configId) => {
    // Return a function that can be used to fetch configuration history
    return () => qloraApi.getConfigurationHistory(configId);
  }, []);

  return {
    // Data
    templates,
    savedConfigs,
    
    // Loading states
    isLoading: isLoadingConfigs,
    
    // Errors
    error: configsError,
    
    // Actions
    saveConfig: saveConfigMutation.mutateAsync,
    updateConfig: updateConfigMutation.mutateAsync,
    deleteConfig: deleteConfigMutation.mutateAsync,
    
    // Utilities
    getConfiguration,
    validateConfig,
    getOptimalConfig,
    estimateTraining,
    exportConfig,
    importConfig,
    getDatasetRecommendations,
    clearValidationCache,
    getConfigHistory,
    
    // Mutation states
    isSaving: saveConfigMutation.isPending,
    isUpdating: updateConfigMutation.isPending,
    isDeleting: deleteConfigMutation.isPending,
    
    // Cache info
    validationCacheSize: validationCache.size
  };
};