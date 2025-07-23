// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::process::Command;
use std::path::Path;

// Define the pipeline directory path
const PIPELINE_DIR: &str = "../../pipeline";

// Command to download videos from YouTube
#[tauri::command]
fn download_videos(links: Vec<String>) -> Result<String, String> {
    // Create a temporary file with the links
    let temp_file = Path::new(PIPELINE_DIR).join("temp_links.txt");
    std::fs::write(&temp_file, links.join("\n"))
        .map_err(|e| format!("Failed to write links file: {}", e))?;

    // Run yt-dlp command
    let output = Command::new("yt-dlp")
        .args([
            "-a", temp_file.to_str().unwrap(),
            "--extract-audio",
            "--audio-format", "mp3",
            "-o", "output/clips/%(title)s.%(ext)s",
        ])
        .current_dir(PIPELINE_DIR)
        .output()
        .map_err(|e| format!("Failed to execute yt-dlp: {}", e))?;

    // Clean up the temporary file
    std::fs::remove_file(temp_file).ok();

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

// Command to transcribe audio files
#[tauri::command]
fn transcribe_audio() -> Result<String, String> {
    // Run the transcription script
    let output = Command::new("bash")
        .args(["-c", "for f in output/clips/*.mp3; do faster-whisper \"$f\" --output_format txt --output_dir output/transcripts; done"])
        .current_dir(PIPELINE_DIR)
        .output()
        .map_err(|e| format!("Failed to execute transcription: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

// Command to generate JSON
#[tauri::command]
fn generate_json() -> Result<String, String> {
    // Run the Python script
    let output = Command::new("python3")
        .args(["scripts/make_quran_lora_json.py", "output/transcripts"])
        .current_dir(PIPELINE_DIR)
        .output()
        .map_err(|e| format!("Failed to execute JSON generation: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

// Command to get pipeline status
#[tauri::command]
fn get_status() -> Result<serde_json::Value, String> {
    let clips_dir = Path::new(PIPELINE_DIR).join("output/clips");
    let transcripts_dir = Path::new(PIPELINE_DIR).join("output/transcripts");
    let json_file = Path::new(PIPELINE_DIR).join("output/json/akhi_lora.json");

    // Count MP3 files
    let clips_count = std::fs::read_dir(clips_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|e| {
                    e.path().extension().map_or(false, |ext| ext == "mp3")
                })
                .count()
        })
        .unwrap_or(0);

    // Count transcript files
    let transcripts_count = std::fs::read_dir(transcripts_dir)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|e| {
                    e.path().extension().map_or(false, |ext| ext == "txt")
                })
                .count()
        })
        .unwrap_or(0);

    // Check if JSON exists and count entries
    let json_exists = json_file.exists();
    let json_count = if json_exists {
        match std::fs::read_to_string(&json_file) {
            Ok(content) => {
                match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(json) => {
                        if let Some(array) = json.as_array() {
                            array.len()
                        } else {
                            0
                        }
                    },
                    Err(_) => 0
                }
            },
            Err(_) => 0
        }
    } else {
        0
    };

    // Create the status JSON
    let status = serde_json::json!({
        "clips": clips_count,
        "transcripts": transcripts_count,
        "json_exists": json_exists,
        "json_count": json_count
    });

    Ok(status)
}

// Command to get list of transcripts
#[tauri::command]
fn get_transcripts() -> Result<serde_json::Value, String> {
    let transcripts_dir = Path::new(PIPELINE_DIR).join("output/transcripts");
    let mut transcripts = Vec::new();

    if let Ok(entries) = std::fs::read_dir(transcripts_dir) {
        for entry in entries.filter_map(Result::ok) {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "txt") {
                if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        let word_count = content.split_whitespace().count();
                        let preview = if content.len() > 200 {
                            format!("{}{}", &content[..200], "...")
                        } else {
                            content.clone()
                        };

                        transcripts.push(serde_json::json!({
                            "file_name": file_name,
                            "word_count": word_count,
                            "preview": preview
                        }));
                    }
                }
            }
        }
    }

    Ok(serde_json::json!({ "transcripts": transcripts }))
}

// Command to get a specific transcript
#[tauri::command]
fn get_transcript(file_name: String) -> Result<serde_json::Value, String> {
    let file_path = Path::new(PIPELINE_DIR).join("output/transcripts").join(&file_name);
    
    if !file_path.exists() {
        return Err(format!("Transcript not found: {}", file_name));
    }

    let content = std::fs::read_to_string(&file_path)
        .map_err(|e| format!("Failed to read transcript: {}", e))?;

    Ok(serde_json::json!({
        "file_name": file_name,
        "content": content
    }))
}

// Command to update a transcript
#[tauri::command]
fn update_transcript(file_name: String, content: String) -> Result<(), String> {
    let file_path = Path::new(PIPELINE_DIR).join("output/transcripts").join(&file_name);
    
    if !file_path.exists() {
        return Err(format!("Transcript not found: {}", file_name));
    }

    std::fs::write(&file_path, content)
        .map_err(|e| format!("Failed to write transcript: {}", e))?;

    Ok(())
}

// Command to get the JSON data
#[tauri::command]
fn get_json() -> Result<serde_json::Value, String> {
    let json_file = Path::new(PIPELINE_DIR).join("output/json/akhi_lora.json");
    
    if !json_file.exists() {
        return Err("JSON file not found".to_string());
    }

    let content = std::fs::read_to_string(&json_file)
        .map_err(|e| format!("Failed to read JSON: {}", e))?;

    let data = serde_json::from_str::<serde_json::Value>(&content)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;

    Ok(serde_json::json!({ "data": data }))
}

// Command to reset all data
#[tauri::command]
fn reset_data() -> Result<(), String> {
    let clips_dir = Path::new(PIPELINE_DIR).join("output/clips");
    let transcripts_dir = Path::new(PIPELINE_DIR).join("output/transcripts");
    let json_dir = Path::new(PIPELINE_DIR).join("output/json");

    // Clear clips directory
    if let Ok(entries) = std::fs::read_dir(&clips_dir) {
        for entry in entries.filter_map(Result::ok) {
            if entry.path().is_file() {
                std::fs::remove_file(entry.path()).ok();
            }
        }
    }

    // Clear transcripts directory
    if let Ok(entries) = std::fs::read_dir(&transcripts_dir) {
        for entry in entries.filter_map(Result::ok) {
            if entry.path().is_file() {
                std::fs::remove_file(entry.path()).ok();
            }
        }
    }

    // Clear JSON directory
    if let Ok(entries) = std::fs::read_dir(&json_dir) {
        for entry in entries.filter_map(Result::ok) {
            if entry.path().is_file() {
                std::fs::remove_file(entry.path()).ok();
            }
        }
    }

    Ok(())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            download_videos,
            transcribe_audio,
            generate_json,
            get_status,
            get_transcripts,
            get_transcript,
            update_transcript,
            get_json,
            reset_data
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}