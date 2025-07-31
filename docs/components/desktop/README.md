# Akhi Data Builder - Desktop App

A Tauri-based cross-platform desktop application for the Akhi Data Builder system. This desktop app provides a native interface for managing the Islamic dataset pipeline with direct integration to the agentic pipeline.

## Features

### Core Functionality
- **YouTube Integration**: Upload and manage video links
- **Pipeline Control**: Start and monitor transcription jobs
- **File Management**: Browse, preview, and export generated files
- **Native Experience**: Cross-platform desktop application

### Advanced Features
- **Real-time Progress**: Live pipeline status monitoring
- **QLoRA Integration**: Direct access to training data generation
- **Error Handling**: Comprehensive error display and recovery
- **Offline Capability**: Local processing without internet dependency
- **Performance Monitoring**: Resource usage and processing statistics

## Tech Stack

- Tauri (Rust + Web Technologies)
- React for UI
- Tailwind CSS for styling

## Development Setup

### Prerequisites

- Rust and Cargo
- Node.js and npm
- Tauri CLI

### Installation

```bash
# Install Tauri CLI
npm install -g @tauri-apps/cli

# Install dependencies
npm install

# Start development server
npm run tauri dev
```

### Building for Production

```bash
npm run tauri build
```

## Project Structure

```
src/                # React frontend code
src-tauri/          # Rust backend code
  ├── src/          # Rust source files
  └── tauri.conf.json # Tauri configuration
```

## Implementation Notes

The desktop app integrates with the agentic pipeline in the `/pipeline` directory using Tauri's command API. This provides:

- **Direct Pipeline Access**: Calls to `run_pipeline.py` for processing
- **State Management**: Integration with `status_tracker.json` for progress
- **File System Access**: Direct access to generated files and logs
- **Native Performance**: Rust backend for optimal performance

## Running the Desktop App

### Quick Start

```bash
# Navigate to desktop app directory
cd desktop_app

# Install dependencies
npm install

# Run in development mode
npm run tauri dev
```

### Production Build

```bash
# Build for current platform
npm run tauri build

# The built application will be in src-tauri/target/release/
```

## Configuration

The desktop app automatically detects the pipeline configuration from:
- `../pipeline/config.yaml` - Pipeline settings
- `../pipeline/db/status_tracker.json` - Processing state
- `../akhi_crewai/config/` - Agent configurations
