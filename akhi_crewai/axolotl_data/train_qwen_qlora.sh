#!/bin/bash
# train_qwen_qlora.sh - Qwen 3 1.7B QLoRA Training Script

set -e

echo "🚀 Starting Qwen 1.7B QLoRA Training"
echo "===================================="
echo "Model: Qwen/Qwen2.5-1.5B-Instruct"
echo "Method: QLoRA (4-bit quantization)"
echo "Dataset: Islamic content from transcripts"
echo ""

# Configuration
CONFIG_FILE="axolotl_config_qwen.yml"
OUTPUT_DIR="./qwen-1.7b-islamic-qlora"
DATA_FILE="/data/work/dev/akhi_data_builder/pipeline/output/json/akhi_lora.json"
BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Pre-flight checks
print_status "Running pre-flight checks..."

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi
print_success "Config file found: $CONFIG_FILE"

# Check if dataset exists
if [ ! -f "$DATA_FILE" ]; then
    print_error "Dataset not found: $DATA_FILE"
    print_warning "Please run QLoRA formatter first:"
    echo "  cd /data/work/dev/akhi_data_builder/akhi_crewai"
    echo "  python -c \"from tools.qlora_formatter import QLoRAFormatterTool; tool = QLoRAFormatterTool(); result = tool._run('/data/work/dev/akhi_data_builder/pipeline_output/transcripts', '/data/work/dev/akhi_data_builder/pipeline/output/json/akhi_lora.json'); print(result)\""
    exit 1
fi
print_success "Dataset found: $DATA_FILE"

# Count training examples
EXAMPLES=$(python3 -c "import json; data=json.load(open('$DATA_FILE')); print(len(data))" 2>/dev/null || echo "0")
if [ "$EXAMPLES" -eq "0" ]; then
    print_error "No training examples found in dataset"
    exit 1
fi
print_success "Training examples: $EXAMPLES"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
    print_success "GPU detected: $GPU_INFO"
else
    print_warning "nvidia-smi not found. Training will use CPU (very slow)"
fi

# Check Python packages
print_status "Checking required packages..."

check_package() {
    if python3 -c "import $1" &> /dev/null; then
        print_success "$1 ✓"
    else
        print_error "$1 not found"
        echo "Install with: pip install $2"
        exit 1
    fi
}

check_package "axolotl" "axolotl[flash-attn,deepspeed]"
check_package "transformers" "transformers>=4.34.0"
check_package "torch" "torch>=2.0.0"
check_package "bitsandbytes" "bitsandbytes>=0.41.0"

# Create output directory
print_status "Creating output directory..."
mkdir -p "$OUTPUT_DIR"
print_success "Output directory: $OUTPUT_DIR"

# Create DeepSpeed config if not exists
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
    print_success "DeepSpeed config created: $DEEPSPEED_CONFIG"
fi

# Display training configuration
echo ""
print_status "Training Configuration:"
echo "  Model: $BASE_MODEL"
echo "  Dataset: $DATA_FILE ($EXAMPLES examples)"
echo "  Output: $OUTPUT_DIR"
echo "  Config: $CONFIG_FILE"
echo "  Method: QLoRA (4-bit quantization)"
echo "  LoRA rank: 16, alpha: 32, dropout: 0.1"
echo "  Batch size: 1 (micro) × 8 (accumulation) = 8 (effective)"
echo "  Learning rate: 0.0002 (cosine schedule)"
echo "  Epochs: 3"
echo ""

# Estimate training time
if [ "$EXAMPLES" -gt 0 ]; then
    # Rough estimation: ~1-2 seconds per example with QLoRA
    ESTIMATED_MINUTES=$((EXAMPLES * 3 * 2 / 60))  # 3 epochs, 2 seconds per example
    print_status "Estimated training time: ~$ESTIMATED_MINUTES minutes"
fi

echo ""
read -p "Continue with training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_warning "Training cancelled by user"
    exit 0
fi

# Start training
echo ""
print_status "Starting Axolotl training..."
echo "=" * 50

# Set environment variables for optimization
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true  # Disable wandb for now

# Training command
TRAIN_CMD="axolotl train $CONFIG_FILE"

if [ -f "$DEEPSPEED_CONFIG" ]; then
    TRAIN_CMD="$TRAIN_CMD --deepspeed $DEEPSPEED_CONFIG"
fi

print_status "Executing: $TRAIN_CMD"
echo ""

# Run training with error handling
if $TRAIN_CMD; then
    echo ""
    print_success "Training completed successfully!"
    print_success "Model saved to: $OUTPUT_DIR"
    
    # Display output structure
    if [ -d "$OUTPUT_DIR" ]; then
        echo ""
        print_status "Output structure:"
        ls -la "$OUTPUT_DIR" | head -10
        
        # Check for adapter files
        if [ -f "$OUTPUT_DIR/adapter_model.bin" ] || [ -f "$OUTPUT_DIR/adapter_model.safetensors" ]; then
            print_success "LoRA adapter files found"
        fi
        
        if [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
            print_success "Adapter configuration found"
        fi
    fi
    
    echo ""
    print_status "Next steps:"
    echo "  1. Test the model with inference script"
    echo "  2. Integrate with CrewAI pipeline"
    echo "  3. Evaluate on Islamic content tasks"
    echo ""
    echo "Example inference:"
    echo "  python inference_qwen.py"
    
else
    echo ""
    print_error "Training failed!"
    print_warning "Check the error messages above"
    print_warning "Common issues:"
    echo "  - Insufficient GPU memory (try reducing batch size)"
    echo "  - Missing dependencies (check package installation)"
    echo "  - Dataset format issues (verify JSON structure)"
    exit 1
fi

echo ""
print_success "Qwen 1.7B QLoRA training pipeline completed!"
echo "Model ready for Islamic content generation 🕌"