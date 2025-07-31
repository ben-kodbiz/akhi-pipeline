#!/bin/bash
# setup_qwen_pipeline.sh - Complete Qwen 3 1.7B QLoRA Pipeline Setup

set -e

echo "🚀 Qwen 3 1.7B Islamic Content Training Pipeline"
echo "==============================================="
echo "This script will set up the complete training pipeline:"
echo "  1. Install dependencies"
echo "  2. Download Qwen 2.5 1.5B Instruct model"
echo "  3. Prepare training environment"
echo "  4. Validate dataset"
echo "  5. Run QLoRA training"
echo "  6. Test trained model"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_header() {
    echo -e "\n${PURPLE}=== $1 ===${NC}"
}

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

# Configuration
PROJECT_ROOT="/data/work/dev/akhi_data_builder"
AXOLOTL_DIR="$PROJECT_ROOT/akhi_crewai/axolotl_data"
DATA_FILE="$PROJECT_ROOT/pipeline/output/json/akhi_lora.json"
MODELS_DIR="$PROJECT_ROOT/models"
BASE_MODEL_DIR="$MODELS_DIR/qwen-2.5-1.5b-instruct"
OUTPUT_DIR="$AXOLOTL_DIR/qwen-1.7b-islamic-qlora"

# Change to axolotl directory
cd "$AXOLOTL_DIR"

print_header "STEP 1: DEPENDENCY CHECK"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found"
    exit 1
fi
print_success "Python 3: $(python3 --version)"

# Check pip
if ! command -v pip &> /dev/null; then
    print_error "pip not found"
    exit 1
fi
print_success "pip: $(pip --version | cut -d' ' -f2)"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_success "GPU: $GPU_INFO"
    
    # Check VRAM
    VRAM=$(echo $GPU_INFO | cut -d',' -f2 | tr -d ' ')
    if [ "$VRAM" -lt 6000 ]; then
        print_warning "GPU has less than 6GB VRAM. Training may be slow or fail."
        echo "Consider using a cloud GPU or reducing batch size."
    fi
else
    print_warning "No GPU detected. Training will be very slow on CPU."
fi

print_header "STEP 2: INSTALL DEPENDENCIES"

# Required packages
REQUIRED_PACKAGES=(
    "torch>=2.0.0"
    "transformers>=4.34.0"
    "bitsandbytes>=0.41.0"
    "peft>=0.5.0"
    "accelerate>=0.21.0"
    "datasets>=2.14.0"
    "huggingface_hub>=0.16.0"
    "tqdm"
)

print_status "Installing required packages..."
for package in "${REQUIRED_PACKAGES[@]}"; do
    print_status "Installing $package..."
    pip install "$package" --quiet
done

# Install Axolotl
print_status "Installing Axolotl..."
if pip show axolotl &> /dev/null; then
    print_success "Axolotl already installed"
else
    pip install "axolotl[flash-attn,deepspeed]" --quiet
    print_success "Axolotl installed"
fi

print_header "STEP 3: VALIDATE DATASET"

# Check if dataset exists
if [ ! -f "$DATA_FILE" ]; then
    print_error "Dataset not found: $DATA_FILE"
    print_status "Generating dataset using QLoRA formatter..."
    
    cd "$PROJECT_ROOT/akhi_crewai"
    python3 -c "
from tools.qlora_formatter import QLoRAFormatterTool
tool = QLoRAFormatterTool()
result = tool._run(
    '/data/work/dev/akhi_data_builder/pipeline_output/transcripts',
    '/data/work/dev/akhi_data_builder/pipeline/output/json/akhi_lora.json'
)
print(result)
"
    
    cd "$AXOLOTL_DIR"
    
    if [ ! -f "$DATA_FILE" ]; then
        print_error "Failed to generate dataset"
        exit 1
    fi
fi

# Validate dataset format
EXAMPLES=$(python3 -c "import json; data=json.load(open('$DATA_FILE')); print(len(data))" 2>/dev/null || echo "0")
if [ "$EXAMPLES" -eq "0" ]; then
    print_error "Dataset is empty or invalid"
    exit 1
fi

print_success "Dataset validated: $EXAMPLES training examples"

# Show sample
print_status "Sample training example:"
python3 -c "
import json
data = json.load(open('$DATA_FILE'))
if data:
    sample = data[0]
    print(f'Instruction: {sample.get(\"instruction\", \"\")[:100]}...')
    print(f'Output: {sample.get(\"output\", \"\")[:100]}...')
"

print_header "STEP 4: DOWNLOAD MODEL"

if [ -d "$BASE_MODEL_DIR" ] && [ -f "$BASE_MODEL_DIR/config.json" ]; then
    print_success "Model already downloaded: $BASE_MODEL_DIR"
else
    print_status "Downloading Qwen 2.5 1.5B Instruct model..."
    ./download_qwen_model.sh
fi

print_header "STEP 5: PREPARE TRAINING ENVIRONMENT"

# Create DeepSpeed config
DEEPSPEED_DIR="deepspeed_configs"
DEEPSPEED_CONFIG="$DEEPSPEED_DIR/zero2.json"

if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    print_status "Creating DeepSpeed configuration..."
    mkdir -p "$DEEPSPEED_DIR"
    cat > "$DEEPSPEED_CONFIG" << 'EOF'
{
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
EOF
    print_success "DeepSpeed config created"
fi

# Validate Axolotl config
if [ ! -f "axolotl_config_qwen.yml" ]; then
    print_error "Axolotl config not found: axolotl_config_qwen.yml"
    exit 1
fi

print_success "Training environment ready"

print_header "STEP 6: TRAINING CONFIGURATION"

echo "Training Parameters:"
echo "  Model: Qwen/Qwen2.5-1.5B-Instruct"
echo "  Method: QLoRA (4-bit quantization)"
echo "  Dataset: $EXAMPLES examples"
echo "  LoRA rank: 16, alpha: 32, dropout: 0.1"
echo "  Batch size: 1 (micro) × 8 (accumulation) = 8 (effective)"
echo "  Learning rate: 0.0002 (cosine schedule)"
echo "  Epochs: 3"
echo "  Output: $OUTPUT_DIR"
echo ""

# Estimate training time
if [ "$EXAMPLES" -gt 0 ]; then
    ESTIMATED_MINUTES=$((EXAMPLES * 3 * 2 / 60))  # 3 epochs, ~2 seconds per example
    print_status "Estimated training time: ~$ESTIMATED_MINUTES minutes"
fi

echo ""
read -p "Start training now? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Training skipped. Run manually with: ./train_qwen_qlora.sh"
    exit 0
fi

print_header "STEP 7: STARTING TRAINING"

# Run training
if ./train_qwen_qlora.sh; then
    print_success "Training completed successfully!"
else
    print_error "Training failed"
    exit 1
fi

print_header "STEP 8: POST-TRAINING VALIDATION"

# Check if model was saved
if [ -d "$OUTPUT_DIR" ]; then
    print_success "Model saved to: $OUTPUT_DIR"
    
    # List output files
    print_status "Output files:"
    ls -la "$OUTPUT_DIR" | head -10
    
    # Check for key files
    if [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
        print_success "LoRA adapter configuration found"
    fi
    
    if [ -f "$OUTPUT_DIR/adapter_model.safetensors" ] || [ -f "$OUTPUT_DIR/adapter_model.bin" ]; then
        print_success "LoRA adapter weights found"
    fi
    
else
    print_error "Training output directory not found"
    exit 1
fi

print_header "STEP 9: MODEL TESTING"

print_status "Running quick model test..."
if python3 inference_qwen.py --test; then
    print_success "Model test passed"
else
    print_warning "Model test failed - check inference_qwen.py"
fi

print_header "SETUP COMPLETE! 🎉"

print_success "Qwen 3 1.7B Islamic Content Pipeline Ready!"
echo ""
print_status "What's been set up:"
echo "  ✅ Dependencies installed"
echo "  ✅ Qwen 2.5 1.5B model downloaded"
echo "  ✅ Training dataset validated ($EXAMPLES examples)"
echo "  ✅ QLoRA training completed"
echo "  ✅ Fine-tuned model saved"
echo "  ✅ Inference script ready"
echo ""
print_status "Next steps:"
echo "  1. Test the model: ./inference_qwen.py"
echo "  2. Integrate with CrewAI pipeline"
echo "  3. Expand dataset for better performance"
echo "  4. Deploy model for production use"
echo ""
print_status "Usage examples:"
echo "  # Interactive chat"
echo "  ./inference_qwen.py"
echo ""
echo "  # Single prompt"
echo "  ./inference_qwen.py --prompt 'What are the five pillars of Islam?'"
echo ""
echo "  # Batch testing"
echo "  ./inference_qwen.py --test"
echo ""
print_success "Happy Islamic content generation! 🕌"