import React, { useState, useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { Settings, Save, RotateCcw, CheckCircle, AlertCircle, Info } from 'lucide-react';
import { useQLoRAConfig } from '../../hooks/qlora/useQLoRAConfig';
import toast from 'react-hot-toast';

// Validation schema
const configSchema = z.object({
  model_name: z.string().min(1, 'Model name is required'),
  base_model: z.string().min(1, 'Base model is required'),
  dataset_path: z.string().min(1, 'Dataset path is required'),
  output_dir: z.string().min(1, 'Output directory is required'),
  
  // LoRA parameters
  lora_r: z.number().min(1).max(512),
  lora_alpha: z.number().min(1).max(1024),
  lora_dropout: z.number().min(0).max(1),
  lora_target_modules: z.array(z.string()).min(1, 'At least one target module required'),
  
  // Training parameters
  num_epochs: z.number().min(1).max(100),
  batch_size: z.number().min(1).max(128),
  learning_rate: z.number().min(0.00001).max(0.1),
  warmup_steps: z.number().min(0),
  weight_decay: z.number().min(0).max(1),
  
  // Advanced settings
  gradient_accumulation_steps: z.number().min(1).max(32),
  max_grad_norm: z.number().min(0).max(10),
  save_steps: z.number().min(1),
  eval_steps: z.number().min(1),
  logging_steps: z.number().min(1),
  
  // Islamic content validation
  enable_content_validation: z.boolean(),
  validation_threshold: z.number().min(0).max(1),
  islamic_keywords_weight: z.number().min(0).max(2)
});

const ConfigurationPanel = ({ onSave, initialConfig = null }) => {
  const { templates, validateConfig, isLoading } = useQLoRAConfig();
  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [validationResult, setValidationResult] = useState(null);
  const [isValidating, setIsValidating] = useState(false);

  const {
    register,
    handleSubmit,
    watch,
    setValue,
    reset,
    formState: { errors, isDirty }
  } = useForm({
    defaultValues: {
      model_name: '',
      base_model: 'microsoft/DialoGPT-medium',
      dataset_path: '',
      output_dir: './models/qlora-output',
      
      // LoRA defaults
      lora_r: 16,
      lora_alpha: 32,
      lora_dropout: 0.1,
      lora_target_modules: ['q_proj', 'v_proj'],
      
      // Training defaults
      num_epochs: 3,
      batch_size: 4,
      learning_rate: 0.0002,
      warmup_steps: 100,
      weight_decay: 0.01,
      
      // Advanced defaults
      gradient_accumulation_steps: 4,
      max_grad_norm: 1.0,
      save_steps: 500,
      eval_steps: 500,
      logging_steps: 10,
      
      // Islamic validation defaults
      enable_content_validation: true,
      validation_threshold: 0.8,
      islamic_keywords_weight: 1.2,
      
      ...initialConfig
    }
  });

  const watchedValues = watch();

  useEffect(() => {
    if (selectedTemplate && templates[selectedTemplate]) {
      const template = templates[selectedTemplate];
      Object.keys(template).forEach(key => {
        setValue(key, template[key]);
      });
      toast.success(`Applied ${selectedTemplate} template`);
    }
  }, [selectedTemplate, templates, setValue]);

  const handleValidateConfig = async () => {
    setIsValidating(true);
    try {
      const result = await validateConfig(watchedValues);
      setValidationResult(result);
      if (result.valid) {
        toast.success('Configuration is valid!');
      } else {
        toast.error('Configuration has issues');
      }
    } catch (error) {
      toast.error('Validation failed: ' + error.message);
    } finally {
      setIsValidating(false);
    }
  };

  const onSubmit = async (data) => {
    try {
      // Validate with Zod
      configSchema.parse(data);
      
      // Additional validation
      await handleValidateConfig();
      
      if (validationResult?.valid !== false) {
        await onSave(data);
        toast.success('Configuration saved successfully!');
      }
    } catch (error) {
      if (error instanceof z.ZodError) {
        error.errors.forEach(err => {
          toast.error(`${err.path.join('.')}: ${err.message}`);
        });
      } else {
        toast.error('Failed to save configuration: ' + error.message);
      }
    }
  };

  const targetModuleOptions = [
    'q_proj', 'v_proj', 'k_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj',
    'embed_tokens', 'lm_head'
  ];

  const baseModelOptions = [
    { value: 'microsoft/DialoGPT-medium', label: 'DialoGPT Medium' },
    { value: 'microsoft/DialoGPT-large', label: 'DialoGPT Large' },
    { value: 'Qwen/Qwen2.5-1.5B-Instruct', label: 'Qwen 2.5 1.5B' },
    { value: 'Qwen/Qwen2.5-3B-Instruct', label: 'Qwen 2.5 3B' },
    { value: 'meta-llama/Llama-2-7b-chat-hf', label: 'Llama 2 7B Chat' },
    { value: 'mistralai/Mistral-7B-Instruct-v0.1', label: 'Mistral 7B Instruct' }
  ];

  return (
    <div className="bg-white rounded-lg p-6 border border-gray-200 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
          <Settings className="w-5 h-5 text-blue-600" />
          Training Configuration
        </h3>
        <div className="flex gap-2">
          <select
            value={selectedTemplate}
            onChange={(e) => setSelectedTemplate(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">Select Template</option>
            <option value="islamic_qa">Islamic Q&A</option>
            <option value="hadith_study">Hadith Study</option>
            <option value="quran_tafsir">Quran Tafsir</option>
            <option value="general_islamic">General Islamic</option>
          </select>
          <button
            type="button"
            onClick={() => reset()}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm hover:bg-gray-50 flex items-center gap-1"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        {/* Basic Configuration */}
        <div className="space-y-4">
          <h4 className="font-medium text-gray-900 border-b border-gray-200 pb-2">Basic Configuration</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model Name *
              </label>
              <input
                {...register('model_name', { required: 'Model name is required' })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="my-islamic-model"
              />
              {errors.model_name && (
                <p className="text-red-600 text-sm mt-1">{errors.model_name.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Base Model *
              </label>
              <select
                {...register('base_model', { required: 'Base model is required' })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                {baseModelOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              {errors.base_model && (
                <p className="text-red-600 text-sm mt-1">{errors.base_model.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Dataset Path *
              </label>
              <input
                {...register('dataset_path', { required: 'Dataset path is required' })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="./data/islamic_dataset.json"
              />
              {errors.dataset_path && (
                <p className="text-red-600 text-sm mt-1">{errors.dataset_path.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Output Directory *
              </label>
              <input
                {...register('output_dir', { required: 'Output directory is required' })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="./models/output"
              />
              {errors.output_dir && (
                <p className="text-red-600 text-sm mt-1">{errors.output_dir.message}</p>
              )}
            </div>
          </div>
        </div>

        {/* LoRA Parameters */}
        <div className="space-y-4">
          <h4 className="font-medium text-gray-900 border-b border-gray-200 pb-2">LoRA Parameters</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                LoRA Rank (r)
                <Info className="w-3 h-3 inline ml-1 text-gray-400" title="Lower values = fewer parameters, faster training" />
              </label>
              <input
                type="number"
                {...register('lora_r', { valueAsNumber: true, min: 1, max: 512 })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              {errors.lora_r && (
                <p className="text-red-600 text-sm mt-1">{errors.lora_r.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                LoRA Alpha
                <Info className="w-3 h-3 inline ml-1 text-gray-400" title="Scaling factor, typically 2x rank" />
              </label>
              <input
                type="number"
                {...register('lora_alpha', { valueAsNumber: true, min: 1, max: 1024 })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              {errors.lora_alpha && (
                <p className="text-red-600 text-sm mt-1">{errors.lora_alpha.message}</p>
              )}
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                LoRA Dropout
                <Info className="w-3 h-3 inline ml-1 text-gray-400" title="Regularization, 0.1 is typical" />
              </label>
              <input
                type="number"
                step="0.01"
                {...register('lora_dropout', { valueAsNumber: true, min: 0, max: 1 })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
              {errors.lora_dropout && (
                <p className="text-red-600 text-sm mt-1">{errors.lora_dropout.message}</p>
              )}
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Target Modules
              <Info className="w-3 h-3 inline ml-1 text-gray-400" title="Which model layers to apply LoRA to" />
            </label>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {targetModuleOptions.map(module => (
                <label key={module} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    value={module}
                    {...register('lora_target_modules')}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-700">{module}</span>
                </label>
              ))}
            </div>
            {errors.lora_target_modules && (
              <p className="text-red-600 text-sm mt-1">{errors.lora_target_modules.message}</p>
            )}
          </div>
        </div>

        {/* Training Parameters */}
        <div className="space-y-4">
          <h4 className="font-medium text-gray-900 border-b border-gray-200 pb-2">Training Parameters</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Epochs
              </label>
              <input
                type="number"
                {...register('num_epochs', { valueAsNumber: true, min: 1, max: 100 })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Batch Size
              </label>
              <input
                type="number"
                {...register('batch_size', { valueAsNumber: true, min: 1, max: 128 })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Learning Rate
              </label>
              <input
                type="number"
                step="0.00001"
                {...register('learning_rate', { valueAsNumber: true, min: 0.00001, max: 0.1 })}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </div>
        </div>

        {/* Islamic Content Validation */}
        <div className="space-y-4">
          <h4 className="font-medium text-gray-900 border-b border-gray-200 pb-2">Islamic Content Validation</h4>
          
          <div className="space-y-4">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                {...register('enable_content_validation')}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-sm font-medium text-gray-700">
                Enable Islamic content validation during training
              </span>
            </label>

            {watchedValues.enable_content_validation && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 ml-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Validation Threshold
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    {...register('validation_threshold', { valueAsNumber: true, min: 0, max: 1 })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Islamic Keywords Weight
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    {...register('islamic_keywords_weight', { valueAsNumber: true, min: 0, max: 2 })}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Validation Results */}
        {validationResult && (
          <div className={`p-4 rounded-lg border ${
            validationResult.valid 
              ? 'bg-green-50 border-green-200' 
              : 'bg-red-50 border-red-200'
          }`}>
            <div className="flex items-center gap-2 mb-2">
              {validationResult.valid ? (
                <CheckCircle className="w-5 h-5 text-green-600" />
              ) : (
                <AlertCircle className="w-5 h-5 text-red-600" />
              )}
              <span className={`font-medium ${
                validationResult.valid ? 'text-green-800' : 'text-red-800'
              }`}>
                {validationResult.valid ? 'Configuration Valid' : 'Configuration Issues'}
              </span>
            </div>
            {validationResult.messages && (
              <ul className={`text-sm space-y-1 ${
                validationResult.valid ? 'text-green-700' : 'text-red-700'
              }`}>
                {validationResult.messages.map((message, index) => (
                  <li key={index}>• {message}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Actions */}
        <div className="flex gap-3 pt-4 border-t border-gray-200">
          <button
            type="button"
            onClick={handleValidateConfig}
            disabled={isValidating}
            className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isValidating ? 'Validating...' : 'Validate'}
          </button>
          <button
            type="submit"
            disabled={isLoading || !isDirty}
            className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white px-4 py-2 rounded-md text-sm font-medium flex items-center justify-center gap-2 disabled:cursor-not-allowed"
          >
            <Save className="w-4 h-4" />
            {isLoading ? 'Saving...' : 'Save Configuration'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ConfigurationPanel;