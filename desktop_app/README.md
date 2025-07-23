# Akhi Data Builder - Desktop App

A Tauri-based desktop application for the Akhi Data Builder system. This desktop app provides a native interface for managing the Islamic dataset pipeline by calling the Python CLI tools directly.

## Features

- Upload/download YouTube video links
- Start transcription jobs
- Browse and export JSONs
- Native desktop experience

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

The desktop app calls the Python CLI tools in the `/pipeline` directory directly using Tauri's command API. This allows for a native desktop experience while leveraging the existing Python pipeline.
