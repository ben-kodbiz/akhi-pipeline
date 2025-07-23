import axios from 'axios';

const API_URL = 'http://localhost:8001';

const api = axios.create({
  baseURL: API_URL,
});

export const getStatus = async () => {
  const response = await api.get('/api/status');
  return response.data;
};

export const downloadVideos = async (linksData) => {
  try {
    console.log('API service sending:', linksData);
    const response = await api.post('/api/videos/download', linksData);
    console.log('API service received:', response.data);
    return response.data;
  } catch (error) {
    console.error('API service error:', error);
    // Rethrow with more detailed error information
    if (error.response) {
      // The request was made and the server responded with a status code
      // that falls out of the range of 2xx
      console.error('Response error data:', error.response.data);
      throw error;
    } else if (error.request) {
      // The request was made but no response was received
      throw new Error('Network error - no response received from server');
    } else {
      // Something happened in setting up the request that triggered an Error
      throw new Error('Error setting up request: ' + error.message);
    }
  }
};

export const transcribeVideos = async () => {
  const response = await api.post('/api/transcribe');
  return response.data;
};

export const generateJson = async () => {
  const response = await api.post('/api/generate-json');
  return response.data;
};

export const getTranscripts = async () => {
  const response = await api.get('/api/transcripts');
  return response.data;
};

export const getTranscript = async (fileName) => {
  const response = await api.get(`/api/transcripts/${fileName}`);
  return response.data;
};

export const updateTranscript = async (fileName, content) => {
  const response = await api.post(`/api/transcripts/${fileName}`, {
    file_name: fileName,
    content,
  });
  return response.data;
};

export const getJson = async () => {
  const response = await api.get('/api/json');
  return response.data;
};

export const getTranscriptionStatus = async () => {
  const response = await api.get('/api/transcription/status');
  return response.data;
};

export const getJsonGenerationStatus = async () => {
  const response = await api.get('/api/json-generation/status');
  return response.data;
};

export const resetData = async () => {
  const response = await api.delete('/api/reset');
  return response.data;
};