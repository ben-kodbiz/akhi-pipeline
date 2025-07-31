# Akhi Data Builder - Frontend Web

A React + Tailwind CSS web interface for the Akhi Data Builder system. This frontend connects to the FastAPI backend to provide a user-friendly interface for managing the Islamic dataset pipeline.

## Features

- Dashboard with pipeline status and controls
- YouTube video URL submission
- Transcript viewing and editing
- JSON preview and export

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

By default, the frontend connects to the backend at `http://localhost:8000`. If your backend is running on a different URL, update the `API_URL` constant in `src/services/api.js`.

## Project Structure

```
src/
├── components/     # Reusable UI components
├── pages/          # Page components
├── services/       # API services
├── App.js          # Main application component with routing
└── index.js        # Application entry point
```
