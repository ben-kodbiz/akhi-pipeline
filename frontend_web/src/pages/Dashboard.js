import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { getStatus, downloadVideos, transcribeVideos, generateJson, resetData, getTranscriptionStatus, getJsonGenerationStatus } from '../services/api';

export default function Dashboard() {
  const [status, setStatus] = useState({ clips: 0, transcripts: 0, json_exists: false, json_count: 0 });
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState('');
  const [videoLinks, setVideoLinks] = useState('');
  const [transcriptionStatus, setTranscriptionStatus] = useState({
    is_running: false,
    current_file: null,
    progress: 0,
    total_files: 0,
    completed_files: 0,
    error: null,
    last_updated: null
  });
  const [jsonGenerationStatus, setJsonGenerationStatus] = useState({
    is_running: false,
    current_step: null,
    progress: 0,
    total_files: 0,
    processed_files: 0,
    error: null,
    last_updated: null
  });
  const [processing, setProcessing] = useState({
    download: false,
    transcribe: false,
    json: false,
    reset: false
  });

  useEffect(() => {
    fetchStatus();
  }, []);

  // Poll transcription status when transcription is running
  useEffect(() => {
    let interval;
    if (transcriptionStatus.is_running || processing.transcribe) {
      interval = setInterval(fetchTranscriptionStatus, 2000); // Poll every 2 seconds
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [transcriptionStatus.is_running, processing.transcribe]);

  // Poll JSON generation status when JSON generation is running
  useEffect(() => {
    let interval;
    if (jsonGenerationStatus.is_running || processing.json) {
      interval = setInterval(fetchJsonGenerationStatus, 2000); // Poll every 2 seconds
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [jsonGenerationStatus.is_running, processing.json]);

  const fetchStatus = async () => {
    try {
      setLoading(true);
      const data = await getStatus();
      setStatus(data);
      
      // Also fetch transcription status
      const transcriptionData = await getTranscriptionStatus();
      setTranscriptionStatus(transcriptionData);
      
      // Also fetch JSON generation status
      const jsonGenerationData = await getJsonGenerationStatus();
      setJsonGenerationStatus(jsonGenerationData);
      
      setLoading(false);
    } catch (error) {
      console.error('Error fetching status:', error);
      setLoading(false);
    }
  };

  const fetchTranscriptionStatus = async () => {
    try {
      const transcriptionData = await getTranscriptionStatus();
      setTranscriptionStatus(transcriptionData);
    } catch (error) {
      console.error('Error fetching transcription status:', error);
    }
  };

  const fetchJsonGenerationStatus = async () => {
    try {
      const jsonGenerationData = await getJsonGenerationStatus();
      setJsonGenerationStatus(jsonGenerationData);
    } catch (error) {
      console.error('Error fetching JSON generation status:', error);
    }
  };

  const handleDownload = async (e) => {
    e.preventDefault();
    if (!videoLinks.trim()) {
      setMessage('Please enter at least one YouTube URL');
      return;
    }

    try {
      setProcessing({ ...processing, download: true });
      setMessage('');
      
      // Parse links from textarea
      const links = videoLinks
        .split('\n')
        .map(link => link.trim())
        .filter(link => link)
        .map(url => ({ url }));
      
      console.log('Dashboard sending links:', { links });
      
      // Make sure we're sending the correct format
      const response = await downloadVideos({ links });
      console.log('Dashboard received response:', response);
      
      // Handle the response message
      setMessage(response.message || 'Videos downloaded successfully');
      setVideoLinks('');
      
      // Wait a bit and refresh status
      setTimeout(fetchStatus, 2000);
    } catch (error) {
      console.error('Dashboard error:', error);
      let errorMessage = 'Error downloading videos: ';
      
      if (error.response?.data?.detail) {
        // Server returned an error with details
        errorMessage += error.response.data.detail;
        console.error('Error details:', error.response.data.detail);
      } else if (error.message) {
        // Network or other error with message
        errorMessage += error.message;
        console.error('Error message:', error.message);
      } else {
        // Fallback error message
        errorMessage += 'Unknown error occurred';
      }
      
      setMessage(errorMessage);
    } finally {
      setProcessing({ ...processing, download: false });
    }
  };

  const handleTranscribe = async () => {
    setProcessing(prev => ({ ...prev, transcribe: true }));
    setMessage('');
    try {
      const result = await transcribeVideos();
      setMessage(result.msg);
      // Start polling for transcription status
      fetchTranscriptionStatus();
      fetchStatus();
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.detail || error.message}`);
      setProcessing(prev => ({ ...prev, transcribe: false }));
    }
  };

  // Update processing state when transcription completes
  useEffect(() => {
    if (!transcriptionStatus.is_running && processing.transcribe) {
      setProcessing(prev => ({ ...prev, transcribe: false }));
      if (transcriptionStatus.error) {
        setMessage(`Transcription Error: ${transcriptionStatus.error}`);
      } else if (transcriptionStatus.completed_files > 0) {
        setMessage(`Transcription completed! Processed ${transcriptionStatus.completed_files} files.`);
        fetchStatus(); // Refresh overall status
      }
    }
  }, [transcriptionStatus.is_running, processing.transcribe, transcriptionStatus.error, transcriptionStatus.completed_files]);

  // Update processing state when JSON generation completes
  useEffect(() => {
    if (!jsonGenerationStatus.is_running && processing.json) {
      setProcessing(prev => ({ ...prev, json: false }));
      if (jsonGenerationStatus.error) {
        setMessage(`JSON Generation Error: ${jsonGenerationStatus.error}`);
      } else if (jsonGenerationStatus.processed_files > 0) {
        setMessage(`JSON generation completed! Processed ${jsonGenerationStatus.processed_files} files.`);
        fetchStatus(); // Refresh overall status
      }
    }
  }, [jsonGenerationStatus.is_running, processing.json, jsonGenerationStatus.error, jsonGenerationStatus.processed_files]);

  const handleGenerateJson = async () => {
    setProcessing(prev => ({ ...prev, json: true }));
    setMessage('');
    try {
      const result = await generateJson();
      setMessage(result.msg);
      // Start polling for JSON generation status
      fetchJsonGenerationStatus();
      fetchStatus();
    } catch (error) {
      setMessage(`Error: ${error.response?.data?.detail || error.message}`);
      setProcessing(prev => ({ ...prev, json: false }));
    }
  };

  const handleReset = async () => {
    if (!window.confirm('Are you sure you want to reset all data? This action cannot be undone.')) {
      return;
    }
    
    try {
      setProcessing({ ...processing, reset: true });
      setMessage('');
      
      const response = await resetData();
      setMessage(response.msg);
      
      // Refresh status
      fetchStatus();
    } catch (error) {
      console.error('Error resetting data:', error);
      setMessage('Error resetting data: ' + (error.response?.data?.detail || error.message));
    } finally {
      setProcessing({ ...processing, reset: false });
    }
  };

  return (
    <div className="space-y-6">
      {/* Status Cards */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
        <div className="overflow-hidden rounded-lg bg-white shadow">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-6 w-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
                </svg>
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="truncate text-sm font-medium text-gray-500">Audio Clips</dt>
                  <dd>
                    <div className="text-lg font-medium text-gray-900">{loading ? '...' : status.clips}</div>
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <div className="overflow-hidden rounded-lg bg-white shadow">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-6 w-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="truncate text-sm font-medium text-gray-500">Transcripts</dt>
                  <dd>
                    <div className="text-lg font-medium text-gray-900">{loading ? '...' : status.transcripts}</div>
                  </dd>
                </dl>
              </div>
            </div>
          </div>
          <div className="bg-gray-50 px-5 py-3">
            <div className="text-sm">
              <Link to="/transcripts" className="font-medium text-primary-600 hover:text-primary-500">
                View all
              </Link>
            </div>
          </div>
        </div>

        <div className="overflow-hidden rounded-lg bg-white shadow">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-6 w-6 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                </svg>
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="truncate text-sm font-medium text-gray-500">JSON Entries</dt>
                  <dd>
                    <div className="text-lg font-medium text-gray-900">
                      {loading ? '...' : (status.json_exists ? status.json_count : 'Not generated')}
                    </div>
                  </dd>
                </dl>
              </div>
            </div>
          </div>
          {status.json_exists && (
            <div className="bg-gray-50 px-5 py-3">
              <div className="text-sm">
                <Link to="/json" className="font-medium text-primary-600 hover:text-primary-500">
                  View JSON
                </Link>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Message Alert */}
      {message && (
        <div className={`rounded-md ${message.includes('Error') ? 'bg-red-50' : 'bg-green-50'} p-4`}>
          <div className="flex">
            <div className="flex-shrink-0">
              {message.includes('Error') ? (
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              ) : (
                <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              )}
            </div>
            <div className="ml-3">
              <p className={`text-sm font-medium ${message.includes('Error') ? 'text-red-800' : 'text-green-800'}`}>{message}</p>
            </div>
          </div>
        </div>
      )}

      {/* Pipeline Controls */}
      <div className="bg-white shadow sm:rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h3 className="text-lg font-medium leading-6 text-gray-900">Pipeline Controls</h3>
          <div className="mt-2 max-w-xl text-sm text-gray-500">
            <p>Run the data pipeline in steps or all at once.</p>
          </div>
          
          {/* Step 1: Download Videos */}
          <div className="mt-5">
            <h4 className="text-md font-medium text-gray-900">Step 1: Download YouTube Videos</h4>
            <form onSubmit={handleDownload} className="mt-3">
              <div>
                <label htmlFor="video-links" className="block text-sm font-medium text-gray-700">
                  YouTube URLs (one per line)
                </label>
                <div className="mt-1">
                  <textarea
                    id="video-links"
                    name="video-links"
                    rows={3}
                    className="block w-full rounded-md border-gray-300 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                    placeholder="https://youtube.com/watch?v=..."
                    value={videoLinks}
                    onChange={(e) => setVideoLinks(e.target.value)}
                  />
                </div>
              </div>
              <button
                type="submit"
                disabled={processing.download}
                className="mt-3 inline-flex items-center rounded-md border border-transparent bg-primary-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50"
              >
                {processing.download ? 'Downloading...' : 'Download Videos'}
              </button>
            </form>
          </div>

          <div className="mt-5 border-t border-gray-200 pt-5">
            <h4 className="text-md font-medium text-gray-900">Step 2: Transcribe Audio</h4>
            <div className="mt-3">
              <button
                type="button"
                onClick={handleTranscribe}
                disabled={processing.transcribe || status.clips === 0}
                className="inline-flex items-center rounded-md border border-transparent bg-primary-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50"
              >
                {processing.transcribe ? 'Transcribing...' : 'Transcribe Audio'}
              </button>
              {status.clips === 0 && (
                <p className="mt-2 text-sm text-gray-500">Download videos first before transcribing.</p>
              )}
              
              {/* Transcription Progress Indicator */}
              {(transcriptionStatus.is_running || processing.transcribe) && (
                <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="flex items-center justify-between mb-2">
                    <h5 className="text-sm font-medium text-blue-900">Transcription Progress</h5>
                    <span className="text-sm text-blue-700">
                      {transcriptionStatus.completed_files} / {transcriptionStatus.total_files} files
                    </span>
                  </div>
                  
                  {/* Progress Bar */}
                  <div className="w-full bg-blue-200 rounded-full h-2 mb-3">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${transcriptionStatus.progress}%` }}
                    ></div>
                  </div>
                  
                  {/* Current File */}
                  {transcriptionStatus.current_file && (
                    <p className="text-sm text-blue-800">
                      Currently processing: <span className="font-medium">{transcriptionStatus.current_file}</span>
                    </p>
                  )}
                  
                  {/* Progress Percentage */}
                  <p className="text-sm text-blue-700 mt-1">
                    {Math.round(transcriptionStatus.progress)}% complete
                  </p>
                </div>
              )}
              
              {/* Transcription Error */}
              {transcriptionStatus.error && !transcriptionStatus.is_running && (
                <div className="mt-4 p-4 bg-red-50 rounded-lg border border-red-200">
                  <div className="flex items-center">
                    <svg className="h-5 w-5 text-red-400 mr-2" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                    <h5 className="text-sm font-medium text-red-900">Transcription Error</h5>
                  </div>
                  <p className="text-sm text-red-800 mt-1">{transcriptionStatus.error}</p>
                </div>
              )}
            </div>
          </div>

          <div className="mt-5 border-t border-gray-200 pt-5">
            <h4 className="text-md font-medium text-gray-900">Step 3: Generate JSON</h4>
            <div className="mt-3">
              <button
                type="button"
                onClick={handleGenerateJson}
                disabled={processing.json || status.transcripts === 0}
                className="inline-flex items-center rounded-md border border-transparent bg-primary-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50"
              >
                {processing.json ? 'Generating...' : 'Generate JSON'}
              </button>
              {status.transcripts === 0 && (
                <p className="mt-2 text-sm text-gray-500">Transcribe audio first before generating JSON.</p>
              )}
              
              {/* JSON Generation Progress Indicator */}
              {(jsonGenerationStatus.is_running || processing.json) && (
                <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <div className="flex items-center justify-between mb-2">
                    <h5 className="text-sm font-medium text-blue-900">JSON Generation Progress</h5>
                    <span className="text-sm text-blue-700">
                      {jsonGenerationStatus.processed_files} / {jsonGenerationStatus.total_files} files
                    </span>
                  </div>
                  
                  {/* Progress Bar */}
                  <div className="w-full bg-blue-200 rounded-full h-2 mb-3">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${jsonGenerationStatus.progress}%` }}
                    ></div>
                  </div>
                  
                  {/* Current Step */}
                  {jsonGenerationStatus.current_step && (
                    <p className="text-sm text-blue-800">
                      Current step: <span className="font-medium">{jsonGenerationStatus.current_step}</span>
                    </p>
                  )}
                  
                  {/* Progress Percentage */}
                  <p className="text-sm text-blue-700 mt-1">
                    {Math.round(jsonGenerationStatus.progress)}% complete
                  </p>
                </div>
              )}
              
              {/* JSON Generation Error */}
              {jsonGenerationStatus.error && !jsonGenerationStatus.is_running && (
                <div className="mt-4 p-4 bg-red-50 rounded-lg border border-red-200">
                  <div className="flex items-center">
                    <svg className="h-5 w-5 text-red-400 mr-2" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                    </svg>
                    <h5 className="text-sm font-medium text-red-900">JSON Generation Error</h5>
                  </div>
                  <p className="text-sm text-red-800 mt-1">{jsonGenerationStatus.error}</p>
                </div>
              )}
            </div>
          </div>

          <div className="mt-5 border-t border-gray-200 pt-5">
            <h4 className="text-md font-medium text-gray-900">Reset Data</h4>
            <div className="mt-3">
              <button
                type="button"
                onClick={handleReset}
                disabled={processing.reset}
                className="inline-flex items-center rounded-md border border-transparent bg-red-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 disabled:opacity-50"
              >
                {processing.reset ? 'Resetting...' : 'Reset All Data'}
              </button>
              <p className="mt-2 text-sm text-gray-500">
                This will delete all downloaded audio, transcripts, and generated JSON files.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}