# Akhi Data Builder - Frontend Web

A React + Tailwind CSS web interface for the Akhi Data Builder system. This frontend connects to the FastAPI backend to provide a user-friendly interface for managing the Islamic dataset pipeline with real-time progress tracking.

## Features

### Core Functionality
- **Real-time Dashboard**: Live pipeline status and progress tracking
- **YouTube Integration**: Video URL submission with validation
- **Transcript Management**: Viewing, editing, and quality control
- **QLoRA Data Export**: JSON preview and export for training
- **Error Handling**: Comprehensive error display and recovery

### Advanced Features
- **Progress Monitoring**: Real-time progress bars for all pipeline stages
- **File Management**: Browse and manage generated files
- **Status Polling**: Automatic updates every 2 seconds
- **Responsive Design**: Modern UI with Tailwind CSS
- **API Integration**: Full REST API connectivity with error handling

## Tech Stack

- React 18
- React Router for navigation
- Tailwind CSS for styling
- Axios for API communication

## Getting Started

### Prerequisites

- Node.js 14+ and npm

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The application will be available at http://localhost:3000

### Building for Production

```bash
npm run build
```

## Connecting to Backend

By default, the frontend connects to the backend at `http://localhost:8001`. If your backend is running on a different URL, update the `API_URL` constant in `src/services/api.js`.

## API Endpoints

The frontend communicates with these backend endpoints:

- `GET /status` - Pipeline status and progress
- `POST /process` - Start pipeline processing
- `GET /files` - List generated files
- `GET /download/{filename}` - Download files
- `POST /upload` - Upload video URLs

## Project Structure

```
src/
├── components/     # Reusable UI components
├── pages/          # Page components
├── services/       # API services
├── App.js          # Main application component with routing
└── index.js        # Application entry point
```
