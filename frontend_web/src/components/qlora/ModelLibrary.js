import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Download, Upload, Trash2, Eye, Play, Pause, 
  Star, Clock, HardDrive, Cpu, MoreVertical,
  Filter, Search, RefreshCw, Archive, ExternalLink
} from 'lucide-react';
import { useModelLibrary } from '../../hooks/qlora/useModelLibrary';
import toast from 'react-hot-toast';

const ModelLibrary = () => {
  const {
    models,
    isLoading,
    downloadModel,
    uploadModel,
    deleteModel,
    deployModel,
    refreshModels
  } = useModelLibrary();

  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState('all');
  const [sortBy, setSortBy] = useState('created_at');
  const [selectedModels, setSelectedModels] = useState(new Set());
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'list'

  // Filter and sort models
  const filteredModels = models
    .filter(model => {
      const matchesSearch = model.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           model.description?.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesFilter = filterStatus === 'all' || model.status === filterStatus;
      return matchesSearch && matchesFilter;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'size':
          return b.size - a.size;
        case 'created_at':
          return new Date(b.created_at) - new Date(a.created_at);
        case 'performance':
          return (b.metrics?.accuracy || 0) - (a.metrics?.accuracy || 0);
        default:
          return 0;
      }
    });

  const handleSelectModel = (modelId) => {
    const newSelected = new Set(selectedModels);
    if (newSelected.has(modelId)) {
      newSelected.delete(modelId);
    } else {
      newSelected.add(modelId);
    }
    setSelectedModels(newSelected);
  };

  const handleBulkAction = async (action) => {
    const selectedModelIds = Array.from(selectedModels);
    if (selectedModelIds.length === 0) {
      toast.error('No models selected');
      return;
    }

    try {
      switch (action) {
        case 'delete':
          await Promise.all(selectedModelIds.map(id => deleteModel(id)));
          toast.success(`Deleted ${selectedModelIds.length} models`);
          break;
        case 'archive':
          // Implementation for archiving models
          toast.success(`Archived ${selectedModelIds.length} models`);
          break;
        default:
          break;
      }
      setSelectedModels(new Set());
    } catch (error) {
      toast.error(`Bulk action failed: ${error.message}`);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'ready': return 'text-green-600 bg-green-100';
      case 'training': return 'text-blue-600 bg-blue-100';
      case 'deploying': return 'text-yellow-600 bg-yellow-100';
      case 'deployed': return 'text-purple-600 bg-purple-100';
      case 'error': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const ModelCard = ({ model }) => (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className="bg-white rounded-lg border border-gray-200 hover:border-blue-300 hover:shadow-md transition-all duration-200 overflow-hidden"
    >
      {/* Card Header */}
      <div className="p-4 border-b border-gray-100">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <input
              type="checkbox"
              checked={selectedModels.has(model.id)}
              onChange={() => handleSelectModel(model.id)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <div>
              <h3 className="font-semibold text-gray-900 text-sm">{model.name}</h3>
              <p className="text-xs text-gray-500 mt-1">{model.base_model}</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(model.status)}`}>
              {model.status}
            </span>
            <div className="relative">
              <button className="p-1 hover:bg-gray-100 rounded">
                <MoreVertical className="w-4 h-4 text-gray-400" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Card Content */}
      <div className="p-4 space-y-3">
        {/* Description */}
        {model.description && (
          <p className="text-sm text-gray-600 line-clamp-2">{model.description}</p>
        )}

        {/* Metrics */}
        {model.metrics && (
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-gray-500">Accuracy:</span>
              <span className="font-medium">{(model.metrics.accuracy * 100).toFixed(1)}%</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-500">Loss:</span>
              <span className="font-medium">{model.metrics.loss?.toFixed(3)}</span>
            </div>
          </div>
        )}

        {/* Model Info */}
        <div className="grid grid-cols-2 gap-2 text-xs text-gray-500">
          <div className="flex items-center space-x-1">
            <HardDrive className="w-3 h-3" />
            <span>{formatFileSize(model.size)}</span>
          </div>
          <div className="flex items-center space-x-1">
            <Clock className="w-3 h-3" />
            <span>{new Date(model.created_at).toLocaleDateString()}</span>
          </div>
        </div>

        {/* Tags */}
        {model.tags && model.tags.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {model.tags.slice(0, 3).map(tag => (
              <span key={tag} className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded">
                {tag}
              </span>
            ))}
            {model.tags.length > 3 && (
              <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded">
                +{model.tags.length - 3}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Card Actions */}
      <div className="px-4 py-3 bg-gray-50 border-t border-gray-100">
        <div className="flex items-center justify-between">
          <div className="flex space-x-2">
            <button
              onClick={() => deployModel(model.id)}
              disabled={model.status === 'deployed' || model.status === 'deploying'}
              className="flex items-center space-x-1 px-2 py-1 text-xs font-medium text-blue-600 hover:text-blue-700 disabled:text-gray-400 disabled:cursor-not-allowed"
            >
              <Play className="w-3 h-3" />
              <span>{model.status === 'deployed' ? 'Deployed' : 'Deploy'}</span>
            </button>
            <button
              onClick={() => downloadModel(model.id)}
              className="flex items-center space-x-1 px-2 py-1 text-xs font-medium text-gray-600 hover:text-gray-700"
            >
              <Download className="w-3 h-3" />
              <span>Download</span>
            </button>
          </div>
          <div className="flex space-x-1">
            <button
              onClick={() => {/* View model details */}}
              className="p-1 text-gray-400 hover:text-gray-600 rounded"
            >
              <Eye className="w-3 h-3" />
            </button>
            <button
              onClick={() => deleteModel(model.id)}
              className="p-1 text-gray-400 hover:text-red-600 rounded"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );

  const ModelListItem = ({ model }) => (
    <motion.div
      layout
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="bg-white border border-gray-200 rounded-lg p-4 hover:border-blue-300 hover:shadow-sm transition-all duration-200"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <input
            type="checkbox"
            checked={selectedModels.has(model.id)}
            onChange={() => handleSelectModel(model.id)}
            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
          <div className="flex-1">
            <div className="flex items-center space-x-3">
              <h3 className="font-semibold text-gray-900">{model.name}</h3>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(model.status)}`}>
                {model.status}
              </span>
            </div>
            <div className="flex items-center space-x-4 mt-1 text-sm text-gray-500">
              <span>{model.base_model}</span>
              <span>{formatFileSize(model.size)}</span>
              <span>{new Date(model.created_at).toLocaleDateString()}</span>
              {model.metrics && (
                <span>Accuracy: {(model.metrics.accuracy * 100).toFixed(1)}%</span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => deployModel(model.id)}
            disabled={model.status === 'deployed' || model.status === 'deploying'}
            className="flex items-center space-x-1 px-3 py-1 text-sm font-medium text-blue-600 hover:text-blue-700 disabled:text-gray-400 disabled:cursor-not-allowed"
          >
            <Play className="w-4 h-4" />
            <span>{model.status === 'deployed' ? 'Deployed' : 'Deploy'}</span>
          </button>
          <button
            onClick={() => downloadModel(model.id)}
            className="p-2 text-gray-400 hover:text-gray-600 rounded"
          >
            <Download className="w-4 h-4" />
          </button>
          <button
            onClick={() => {/* View model details */}}
            className="p-2 text-gray-400 hover:text-gray-600 rounded"
          >
            <Eye className="w-4 h-4" />
          </button>
          <button
            onClick={() => deleteModel(model.id)}
            className="p-2 text-gray-400 hover:text-red-600 rounded"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>
    </motion.div>
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Model Library</h2>
          <p className="text-gray-600 mt-1">
            Manage your trained QLoRA models and deployments
          </p>
        </div>
        <div className="flex space-x-3">
          <button
            onClick={() => setShowUploadModal(true)}
            className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium"
          >
            <Upload className="w-4 h-4" />
            <span>Upload Model</span>
          </button>
          <button
            onClick={refreshModels}
            disabled={isLoading}
            className="flex items-center space-x-2 border border-gray-300 hover:bg-gray-50 px-4 py-2 rounded-lg font-medium disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="Search models..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>

          {/* Filters */}
          <div className="flex space-x-3">
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="all">All Status</option>
              <option value="ready">Ready</option>
              <option value="training">Training</option>
              <option value="deployed">Deployed</option>
              <option value="error">Error</option>
            </select>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="created_at">Date Created</option>
              <option value="name">Name</option>
              <option value="size">Size</option>
              <option value="performance">Performance</option>
            </select>

            <div className="flex border border-gray-300 rounded-lg overflow-hidden">
              <button
                onClick={() => setViewMode('grid')}
                className={`px-3 py-2 text-sm font-medium ${
                  viewMode === 'grid'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                Grid
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`px-3 py-2 text-sm font-medium border-l border-gray-300 ${
                  viewMode === 'list'
                    ? 'bg-blue-600 text-white'
                    : 'bg-white text-gray-700 hover:bg-gray-50'
                }`}
              >
                List
              </button>
            </div>
          </div>
        </div>

        {/* Bulk Actions */}
        {selectedModels.size > 0 && (
          <div className="mt-4 pt-4 border-t border-gray-200">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">
                {selectedModels.size} model{selectedModels.size !== 1 ? 's' : ''} selected
              </span>
              <div className="flex space-x-2">
                <button
                  onClick={() => handleBulkAction('archive')}
                  className="flex items-center space-x-1 px-3 py-1 text-sm font-medium text-gray-600 hover:text-gray-700 border border-gray-300 rounded"
                >
                  <Archive className="w-4 h-4" />
                  <span>Archive</span>
                </button>
                <button
                  onClick={() => handleBulkAction('delete')}
                  className="flex items-center space-x-1 px-3 py-1 text-sm font-medium text-red-600 hover:text-red-700 border border-red-300 rounded"
                >
                  <Trash2 className="w-4 h-4" />
                  <span>Delete</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Models Display */}
      <div className="space-y-4">
        {isLoading ? (
          <div className="text-center py-12">
            <RefreshCw className="w-8 h-8 animate-spin text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600">Loading models...</p>
          </div>
        ) : filteredModels.length === 0 ? (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
              <Cpu className="w-8 h-8 text-gray-400" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No models found</h3>
            <p className="text-gray-600 mb-4">
              {searchTerm || filterStatus !== 'all'
                ? 'Try adjusting your search or filters'
                : 'Upload your first model to get started'}
            </p>
            {!searchTerm && filterStatus === 'all' && (
              <button
                onClick={() => setShowUploadModal(true)}
                className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg font-medium"
              >
                Upload Model
              </button>
            )}
          </div>
        ) : (
          <AnimatePresence>
            {viewMode === 'grid' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredModels.map(model => (
                  <ModelCard key={model.id} model={model} />
                ))}
              </div>
            ) : (
              <div className="space-y-3">
                {filteredModels.map(model => (
                  <ModelListItem key={model.id} model={model} />
                ))}
              </div>
            )}
          </AnimatePresence>
        )}
      </div>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Upload Model</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Model File
                </label>
                <input
                  type="file"
                  accept=".bin,.safetensors,.pt,.pth"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Model Name
                </label>
                <input
                  type="text"
                  placeholder="Enter model name"
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Description (Optional)
                </label>
                <textarea
                  placeholder="Describe your model"
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowUploadModal(false)}
                className="px-4 py-2 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  // Handle upload
                  setShowUploadModal(false);
                  toast.success('Model upload started');
                }}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
              >
                Upload
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelLibrary;