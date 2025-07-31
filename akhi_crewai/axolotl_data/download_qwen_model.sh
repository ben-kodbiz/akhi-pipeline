#!/bin/bash
# download_qwen_model.sh - Download Qwen 2.5 1.5B Instruct model

set -e

echo "📥 Qwen Model Download Script"
echo "============================"
echo "Model: Qwen/Qwen2.5-1.5B-Instruct"
echo "Size: ~3GB"
echo "Purpose: Islamic content fine-tuning base model"
echo ""

# Configuration
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
MODELS_DIR="/data/work/dev/akhi_data_builder/models"
LOCAL_MODEL_DIR="$MODELS_DIR/qwen-2.5-1.5b-instruct"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if model already exists
if [ -d "$LOCAL_MODEL_DIR" ] && [ -f "$LOCAL_MODEL_DIR/config.json" ]; then
    print_warning "Model already exists at: $LOCAL_MODEL_DIR"
    echo ""
    read -p "Re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Using existing model"
        exit 0
    fi
    print_status "Removing existing model..."
    rm -rf "$LOCAL_MODEL_DIR"
fi

# Check available disk space
print_status "Checking disk space..."
AVAILABLE_SPACE=$(df "$MODELS_DIR" 2>/dev/null | awk 'NR==2 {print $4}' || echo "0")
REQUIRED_SPACE=3145728  # 3GB in KB

if [ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]; then
    print_error "Insufficient disk space"
    echo "Required: 3GB, Available: $((AVAILABLE_SPACE / 1024 / 1024))GB"
    exit 1
fi
print_success "Sufficient disk space available"

# Check Python and required packages
print_status "Checking dependencies..."

if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found"
    exit 1
fi

if ! python3 -c "import huggingface_hub" &> /dev/null; then
    print_warning "huggingface_hub not found, installing..."
    pip install huggingface_hub
fi

print_success "Dependencies ready"

# Create models directory
print_status "Creating models directory..."
mkdir -p "$MODELS_DIR"
mkdir -p "$LOCAL_MODEL_DIR"
print_success "Directory created: $LOCAL_MODEL_DIR"

# Download model
print_status "Starting download..."
echo "This may take 5-15 minutes depending on your internet connection"
echo ""

# Create download script
cat > /tmp/download_qwen.py << EOF
import os
import sys
from huggingface_hub import snapshot_download
from tqdm import tqdm

def download_with_progress():
    try:
        print(f"📥 Downloading {os.environ['MODEL_NAME']}...")
        print(f"📁 Destination: {os.environ['LOCAL_MODEL_DIR']}")
        print("")
        
        snapshot_download(
            repo_id=os.environ['MODEL_NAME'],
            local_dir=os.environ['LOCAL_MODEL_DIR'],
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print("")
        print("✅ Download completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    success = download_with_progress()
    sys.exit(0 if success else 1)
EOF

# Set environment variables and run download
export MODEL_NAME="$MODEL_NAME"
export LOCAL_MODEL_DIR="$LOCAL_MODEL_DIR"

if python3 /tmp/download_qwen.py; then
    print_success "Model downloaded successfully!"
    
    # Verify download
    print_status "Verifying download..."
    
    REQUIRED_FILES=("config.json" "tokenizer.json" "tokenizer_config.json")
    ALL_FOUND=true
    
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "$LOCAL_MODEL_DIR/$file" ]; then
            print_success "$file ✓"
        else
            print_error "$file ✗"
            ALL_FOUND=false
        fi
    done
    
    if [ "$ALL_FOUND" = true ]; then
        print_success "All required files found"
        
        # Display model info
        echo ""
        print_status "Model Information:"
        echo "  Location: $LOCAL_MODEL_DIR"
        echo "  Size: $(du -sh "$LOCAL_MODEL_DIR" | cut -f1)"
        echo "  Files: $(find "$LOCAL_MODEL_DIR" -type f | wc -l)"
        
        # Test model loading
        print_status "Testing model loading..."
        cat > /tmp/test_qwen.py << 'EOF'
import os
try:
    from transformers import AutoTokenizer, AutoConfig
    
    model_path = os.environ['LOCAL_MODEL_DIR']
    
    # Test config loading
    config = AutoConfig.from_pretrained(model_path)
    print(f"✅ Config loaded: {config.model_type}")
    
    # Test tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"✅ Tokenizer loaded: {len(tokenizer)} tokens")
    
    print("✅ Model ready for training!")
    
except Exception as e:
    print(f"❌ Model test failed: {e}")
    exit(1)
EOF
        
        if python3 /tmp/test_qwen.py; then
            echo ""
            print_success "Model download and verification complete!"
            print_status "Ready for QLoRA training"
            
            echo ""
            print_status "Next steps:"
            echo "  1. Run training: ./train_qwen_qlora.sh"
            echo "  2. Monitor training progress"
            echo "  3. Test fine-tuned model"
            
        else
            print_error "Model verification failed"
            exit 1
        fi
        
    else
        print_error "Download incomplete - missing required files"
        exit 1
    fi
    
else
    print_error "Download failed"
    print_warning "Possible issues:"
    echo "  - Network connectivity"
    echo "  - Hugging Face Hub access"
    echo "  - Insufficient disk space"
    exit 1
fi

# Cleanup
rm -f /tmp/download_qwen.py /tmp/test_qwen.py

echo ""
print_success "Qwen 2.5 1.5B Instruct model ready! 🚀"