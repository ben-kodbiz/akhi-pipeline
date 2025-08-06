import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json'
  }
});

// Request interceptor for authentication
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    
    const message = error.response?.data?.detail || 
                   error.response?.data?.message || 
                   error.message || 
                   'An unexpected error occurred';
    
    return Promise.reject(new Error(message));
  }
);

export const qloraApi = {
  // Training Jobs
  async getTrainingJobs() {
    const response = await api.get('/api/qlora/training/jobs');
    return response.data?.jobs || [];
  },

  async getTrainingJob(jobId) {
    return api.get(`/api/qlora/training/jobs/${jobId}`);
  },

  async startTraining(config) {
    return api.post('/api/qlora/training/start', config);
  },

  async stopTraining(jobId) {
    return api.post(`/api/qlora/training/jobs/${jobId}/stop`);
  },

  async resumeTraining(jobId) {
    return api.post(`/api/qlora/training/jobs/${jobId}/resume`);
  },

  async deleteTrainingJob(jobId) {
    return api.delete(`/api/qlora/training/jobs/${jobId}`);
  },

  async getTrainingLogs(jobId, lines = 100) {
    return api.get(`/api/qlora/training/jobs/${jobId}/logs`, {
      params: { lines }
    });
  },

  async getTrainingMetrics(jobId) {
    return api.get(`/api/qlora/training/jobs/${jobId}/metrics`);
  },

  async exportTrainingResults(jobId, format = 'json') {
    const response = await api.get(`/api/qlora/training/jobs/${jobId}/export`, {
      params: { format },
      responseType: 'blob'
    });
    return response;
  },

  // Configuration Management
  async getConfigurations() {
    return api.get('/api/qlora/configs');
  },

  async getConfiguration(configId) {
    return api.get(`/api/qlora/configs/${configId}`);
  },

  async saveConfiguration(config) {
    return api.post('/api/qlora/configs', config);
  },

  async updateConfiguration(configId, config) {
    return api.put(`/api/qlora/configs/${configId}`, config);
  },

  async deleteConfiguration(configId) {
    return api.delete(`/api/qlora/configs/${configId}`);
  },

  async validateConfiguration(config) {
    return api.post('/api/qlora/configs/validate', config);
  },

  async getOptimalConfiguration(requirements) {
    return api.post('/api/qlora/configs/optimize', requirements);
  },

  async estimateTraining(config) {
    return api.post('/api/qlora/training/estimate', config);
  },

  async getConfigurationHistory(configId) {
    return api.get(`/api/qlora/configs/${configId}/history`);
  },

  async analyzeDataset(datasetPath) {
    return api.post('/api/qlora/dataset/analyze', { dataset_path: datasetPath });
  },

  // Model Management
  async getModels() {
    try {
      const response = await api.get('/api/qlora/models');
      return response.data?.models || [];
    } catch (error) {
      // Return empty array if endpoint doesn't exist
      if (error.message.includes('Not Found') || error.message.includes('404')) {
        return [];
      }
      throw error;
    }
  },

  async getModel(modelId) {
    return api.get(`/api/qlora/models/${modelId}`);
  },

  async uploadModel(formData, onProgress) {
    return api.post('/api/qlora/models/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      onUploadProgress: onProgress
    });
  },

  async downloadModel(modelId, onProgress) {
    const response = await api.get(`/api/qlora/models/${modelId}/download`, {
      responseType: 'blob',
      onDownloadProgress: onProgress
    });
    return response;
  },

  async deleteModel(modelId) {
    return api.delete(`/api/qlora/models/${modelId}`);
  },

  async updateModel(modelId, metadata) {
    return api.put(`/api/qlora/models/${modelId}`, metadata);
  },

  async deployModel(modelId, deploymentConfig = {}) {
    return api.post(`/api/qlora/models/${modelId}/deploy`, deploymentConfig);
  },

  async undeployModel(modelId) {
    return api.post(`/api/qlora/models/${modelId}/undeploy`);
  },

  async getDeploymentStatus(modelId) {
    return api.get(`/api/qlora/models/${modelId}/deployment/status`);
  },

  async archiveModel(modelId) {
    return api.post(`/api/qlora/models/${modelId}/archive`);
  },

  async restoreModel(modelId) {
    return api.post(`/api/qlora/models/${modelId}/restore`);
  },

  async getModelMetrics(modelId) {
    return api.get(`/api/qlora/models/${modelId}/metrics`);
  },

  async testModelInference(modelId, input) {
    return api.post(`/api/qlora/models/${modelId}/inference`, { input });
  },

  async compareModels(modelIds) {
    return api.post('/api/qlora/models/compare', { model_ids: modelIds });
  },

  async getModelUsage(modelId, timeRange = '7d') {
    return api.get(`/api/qlora/models/${modelId}/usage`, {
      params: { time_range: timeRange }
    });
  },

  async exportModel(modelId, format = 'onnx') {
    const response = await api.get(`/api/qlora/models/${modelId}/export`, {
      params: { format },
      responseType: 'blob'
    });
    return response;
  },

  async cloneModel(modelId, newName) {
    return api.post(`/api/qlora/models/${modelId}/clone`, { name: newName });
  },

  async getModelVersions(modelId) {
    return api.get(`/api/qlora/models/${modelId}/versions`);
  },

  async createModelVersion(modelId, versionData) {
    return api.post(`/api/qlora/models/${modelId}/versions`, versionData);
  },

  // Dataset Management
  async getDatasets() {
    return api.get('/api/qlora/datasets');
  },

  async getDataset(datasetId) {
    return api.get(`/api/qlora/datasets/${datasetId}`);
  },

  async uploadDataset(formData, onProgress) {
    return api.post('/api/qlora/datasets/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      onUploadProgress: onProgress
    });
  },

  async validateDataset(datasetId) {
    return api.post(`/api/qlora/datasets/${datasetId}/validate`);
  },

  async preprocessDataset(datasetId, config) {
    return api.post(`/api/qlora/datasets/${datasetId}/preprocess`, config);
  },

  async getDatasetStatistics(datasetId) {
    return api.get(`/api/qlora/datasets/${datasetId}/statistics`);
  },

  async splitDataset(datasetId, splitConfig) {
    return api.post(`/api/qlora/datasets/${datasetId}/split`, splitConfig);
  },

  // Islamic Content Validation
  async validateIslamicContent(text) {
    return api.post('/api/qlora/islamic/validate', { text });
  },

  async getIslamicKeywords() {
    return api.get('/api/qlora/islamic/keywords');
  },

  async updateIslamicKeywords(keywords) {
    return api.put('/api/qlora/islamic/keywords', { keywords });
  },

  async getContentValidationRules() {
    return api.get('/api/qlora/islamic/rules');
  },

  async updateContentValidationRules(rules) {
    return api.put('/api/qlora/islamic/rules', { rules });
  },

  // System Monitoring
  async getSystemStatus() {
    return api.get('/api/qlora/system/status');
  },

  async getResourceUsage() {
    return api.get('/api/qlora/system/resources');
  },

  async getSystemLogs(lines = 100) {
    return api.get('/api/qlora/system/logs', {
      params: { lines }
    });
  },

  async getSystemMetrics(timeRange = '1h') {
    return api.get('/api/qlora/system/metrics', {
      params: { time_range: timeRange }
    });
  },

  // Deployment Management
  async getDeployments() {
    return api.get('/api/qlora/deployments');
  },

  async getDeployment(deploymentId) {
    return api.get(`/api/qlora/deployments/${deploymentId}`);
  },

  async createDeployment(config) {
    return api.post('/api/qlora/deployments', config);
  },

  async updateDeployment(deploymentId, config) {
    return api.put(`/api/qlora/deployments/${deploymentId}`, config);
  },

  async deleteDeployment(deploymentId) {
    return api.delete(`/api/qlora/deployments/${deploymentId}`);
  },

  async scaleDeployment(deploymentId, replicas) {
    return api.post(`/api/qlora/deployments/${deploymentId}/scale`, { replicas });
  },

  async getDeploymentLogs(deploymentId, lines = 100) {
    return api.get(`/api/qlora/deployments/${deploymentId}/logs`, {
      params: { lines }
    });
  },

  async getDeploymentMetrics(deploymentId) {
    return api.get(`/api/qlora/deployments/${deploymentId}/metrics`);
  },

  // Evaluation and Testing
  async createEvaluation(config) {
    return api.post('/api/qlora/evaluations', config);
  },

  async getEvaluations() {
    return api.get('/api/qlora/evaluations');
  },

  async getEvaluation(evaluationId) {
    return api.get(`/api/qlora/evaluations/${evaluationId}`);
  },

  async getEvaluationResults(evaluationId) {
    return api.get(`/api/qlora/evaluations/${evaluationId}/results`);
  },

  async runBenchmark(modelId, benchmarkConfig) {
    return api.post(`/api/qlora/models/${modelId}/benchmark`, benchmarkConfig);
  },

  // Utilities
  async healthCheck() {
    return api.get('/api/qlora/health');
  },

  async getVersion() {
    return api.get('/api/qlora/version');
  },

  async getCapabilities() {
    return api.get('/api/qlora/capabilities');
  },

  // WebSocket connection helper
  getWebSocketUrl(path = '') {
    const baseUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';
    return `${baseUrl}/ws/qlora${path}`;
  }
};

export default qloraApi;