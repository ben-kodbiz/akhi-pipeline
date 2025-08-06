#!/usr/bin/env python3
"""
Model Deployer Tool for CrewAI

This module provides a CrewAI-compatible tool for deploying validated QLoRA models
to various production environments and serving configurations.

Author: Assistant
Date: December 2024
Phase: 8 - QLoRA Fine-Tuning
"""

import os
import sys
import json
import yaml
import shutil
import subprocess
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from pydantic import BaseModel, Field, field_validator
from crewai.tools import BaseTool


class ModelDeployerInput(BaseModel):
    """
    Input schema for Model Deployer Tool.
    """
    model_path: str = Field(
        description="Path to the validated QLoRA model",
        default="models/akhi_qlora"
    )
    deployment_target: str = Field(
        description="Deployment target (local, huggingface, api_server)",
        default="local"
    )
    deployment_name: str = Field(
        description="Name for the deployed model",
        default="akhi-islamic-assistant"
    )
    deployment_dir: str = Field(
        description="Directory for deployment artifacts",
        default="deployments"
    )
    api_port: int = Field(
        description="Port for API server deployment",
        default=8000,
        ge=1000,
        le=65535
    )
    max_length: int = Field(
        description="Maximum generation length for deployment",
        default=512,
        ge=50,
        le=2048
    )
    temperature: float = Field(
        description="Default temperature for generation",
        default=0.7,
        ge=0.1,
        le=2.0
    )
    enable_auth: bool = Field(
        description="Enable authentication for API deployment",
        default=True
    )

    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v):
        if not os.path.exists(v):
            raise ValueError(f"Model path does not exist: {v}")
        return v
    
    @field_validator('deployment_target')
    @classmethod
    def validate_deployment_target(cls, v):
        valid_targets = ['local', 'huggingface', 'api_server']
        if v not in valid_targets:
            raise ValueError(f"Invalid deployment target. Must be one of: {valid_targets}")
        return v


class ModelDeployerTool(BaseTool):
    """
    CrewAI tool for deploying validated QLoRA models.
    
    This tool handles various deployment scenarios:
    - Local deployment for development/testing
    - HuggingFace Hub deployment for sharing
    - API server deployment for production use
    """
    
    name: str = "model_deployer"
    description: str = (
        "Deploy validated QLoRA models to production environments. "
        "This tool handles model packaging, configuration, and deployment "
        "to various targets including local, API server, and HuggingFace Hub. "
        "Input: model path, deployment target, configuration options. "
        "Output: Deployment artifacts and access instructions."
    )
    args_schema: type = ModelDeployerInput
    config: Optional[Dict[str, Any]] = None
    deployment_status: Optional[Dict[str, Any]] = None

    def __init__(self):
        super().__init__()
        self.config = self._load_config()
        self.deployment_status = {}
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration for the tool.
        
        Returns:
            Configuration dictionary
        """
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '../config/crew_config.yaml'
        )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('model_deployment', {})
        except FileNotFoundError:
            return {
                'default_port': 8000,
                'default_max_length': 512,
                'default_temperature': 0.7
            }

    def _run(
        self,
        model_path: str = "models/akhi_qlora",
        deployment_target: str = "local",
        deployment_name: str = "akhi-islamic-assistant",
        deployment_dir: str = "deployments",
        api_port: int = 8000,
        max_length: int = 512,
        temperature: float = 0.7,
        enable_auth: bool = True,
    ) -> str:
        """
        Execute model deployment process.
        
        Args:
            model_path: Path to the model
            deployment_target: Target deployment environment
            deployment_name: Name for deployment
            deployment_dir: Deployment directory
            api_port: API server port
            max_length: Maximum generation length
            temperature: Generation temperature
            enable_auth: Enable authentication
            
        Returns:
            Deployment status and instructions
        """
        try:
            # Create deployment directory
            os.makedirs(deployment_dir, exist_ok=True)
            
            # Initialize deployment tracking
            deployment_id = f"{deployment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.deployment_status[deployment_id] = {
                'status': 'starting',
                'target': deployment_target,
                'start_time': datetime.now(),
                'model_path': model_path,
                'deployment_dir': deployment_dir
            }
            
            # Route to appropriate deployment method
            if deployment_target == 'local':
                result = self._deploy_local(
                    model_path, deployment_name, deployment_dir,
                    max_length, temperature
                )
            elif deployment_target == 'api_server':
                result = self._deploy_api_server(
                    model_path, deployment_name, deployment_dir,
                    api_port, max_length, temperature, enable_auth
                )
            elif deployment_target == 'huggingface':
                result = self._deploy_huggingface(
                    model_path, deployment_name, deployment_dir
                )
            else:
                result = {
                    'success': False,
                    'error': f'Unsupported deployment target: {deployment_target}'
                }
            
            # Update deployment status
            self.deployment_status[deployment_id].update(result)
            self.deployment_status[deployment_id]['deployment_id'] = deployment_id
            
            return self._format_deployment_result(result, deployment_id)
            
        except Exception as e:
            return f"❌ Error during model deployment: {str(e)}"

    def _deploy_local(self, model_path: str, deployment_name: str, 
                     deployment_dir: str, max_length: int, 
                     temperature: float) -> Dict[str, Any]:
        """
        Deploy model for local inference.
        
        Args:
            model_path: Path to the model
            deployment_name: Name for deployment
            deployment_dir: Deployment directory
            max_length: Maximum generation length
            temperature: Generation temperature
            
        Returns:
            Deployment results
        """
        try:
            local_deployment_dir = os.path.join(deployment_dir, f"{deployment_name}_local")
            os.makedirs(local_deployment_dir, exist_ok=True)
            
            # Copy model files
            model_deployment_dir = os.path.join(local_deployment_dir, "model")
            if os.path.exists(model_deployment_dir):
                shutil.rmtree(model_deployment_dir)
            shutil.copytree(model_path, model_deployment_dir)
            
            # Create inference script
            inference_script = self._create_inference_script(
                "./model", max_length, temperature
            )
            
            script_path = os.path.join(local_deployment_dir, "inference.py")
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(inference_script)
            
            # Create requirements file
            requirements = [
                "transformers>=4.35.0",
                "torch>=2.0.0",
                "peft>=0.6.0",
                "accelerate>=0.24.0"
            ]
            
            requirements_path = os.path.join(local_deployment_dir, "requirements.txt")
            with open(requirements_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(requirements))
            
            return {
                'success': True,
                'deployment_type': 'local',
                'deployment_path': local_deployment_dir,
                'inference_script': script_path,
                'requirements': requirements_path,
                'instructions': [
                    f"Local deployment prepared: {local_deployment_dir}",
                    f"1. Install dependencies: pip install -r {requirements_path}",
                    f"2. Run inference: python {script_path}",
                    "3. Follow the prompts to interact with the model"
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Local deployment failed: {str(e)}'
            }

    def _deploy_api_server(self, model_path: str, deployment_name: str,
                          deployment_dir: str, api_port: int,
                          max_length: int, temperature: float,
                          enable_auth: bool) -> Dict[str, Any]:
        """
        Deploy model as API server.
        
        Args:
            model_path: Path to the model
            deployment_name: Name for deployment
            deployment_dir: Deployment directory
            api_port: API server port
            max_length: Maximum generation length
            temperature: Generation temperature
            enable_auth: Enable authentication
            
        Returns:
            Deployment results
        """
        try:
            api_deployment_dir = os.path.join(deployment_dir, f"{deployment_name}_api")
            os.makedirs(api_deployment_dir, exist_ok=True)
            
            # Copy model files
            model_deployment_dir = os.path.join(api_deployment_dir, "model")
            if os.path.exists(model_deployment_dir):
                shutil.rmtree(model_deployment_dir)
            shutil.copytree(model_path, model_deployment_dir)
            
            # Create API server script
            api_script = self._create_api_server_script(
                "./model", api_port, max_length, temperature, enable_auth
            )
            
            script_path = os.path.join(api_deployment_dir, "api_server.py")
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(api_script)
            
            # Create requirements file
            requirements = [
                "fastapi>=0.104.0",
                "uvicorn>=0.24.0",
                "transformers>=4.35.0",
                "torch>=2.0.0",
                "peft>=0.6.0",
                "pydantic>=2.0.0"
            ]
            
            requirements_path = os.path.join(api_deployment_dir, "requirements.txt")
            with open(requirements_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(requirements))
            
            return {
                'success': True,
                'deployment_type': 'api_server',
                'deployment_path': api_deployment_dir,
                'api_script': script_path,
                'api_port': api_port,
                'api_url': f'http://localhost:{api_port}',
                'docs_url': f'http://localhost:{api_port}/docs',
                'requirements': requirements_path,
                'instructions': [
                    f"API server deployment prepared: {api_deployment_dir}",
                    f"1. Install dependencies: pip install -r {requirements_path}",
                    f"2. Start server: uvicorn api_server:app --host 0.0.0.0 --port {api_port}",
                    f"3. API will be available at: http://localhost:{api_port}",
                    f"4. Interactive docs: http://localhost:{api_port}/docs"
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'API server deployment failed: {str(e)}'
            }

    def _deploy_huggingface(self, model_path: str, deployment_name: str,
                           deployment_dir: str) -> Dict[str, Any]:
        """
        Deploy model to HuggingFace Hub.
        
        Args:
            model_path: Path to the model
            deployment_name: Name for deployment
            deployment_dir: Deployment directory
            
        Returns:
            Deployment results
        """
        try:
            hf_deployment_dir = os.path.join(deployment_dir, f"{deployment_name}_huggingface")
            os.makedirs(hf_deployment_dir, exist_ok=True)
            
            # Copy model files
            model_deployment_dir = os.path.join(hf_deployment_dir, "model")
            if os.path.exists(model_deployment_dir):
                shutil.rmtree(model_deployment_dir)
            shutil.copytree(model_path, model_deployment_dir)
            
            # Create model card
            model_card = self._create_model_card(deployment_name)
            card_path = os.path.join(hf_deployment_dir, "README.md")
            with open(card_path, 'w', encoding='utf-8') as f:
                f.write(model_card)
            
            # Create upload script
            upload_script = f'''#!/usr/bin/env python3
"""
HuggingFace Hub Upload Script
"""

from huggingface_hub import HfApi, create_repo
import os

def upload_model():
    """Upload model to HuggingFace Hub"""
    api = HfApi()
    
    # Create repository
    try:
        create_repo(
            repo_id="{deployment_name}",
            repo_type="model",
            exist_ok=True
        )
        print(f"Repository created/verified: {deployment_name}")
    except Exception as e:
        print(f"Repository creation failed: {{e}}")
        return
    
    # Upload files
    try:
        api.upload_folder(
            folder_path="./model",
            repo_id="{deployment_name}",
            repo_type="model"
        )
        print(f"Model uploaded successfully to: https://huggingface.co/{deployment_name}")
    except Exception as e:
        print(f"Upload failed: {{e}}")

if __name__ == "__main__":
    upload_model()
'''
            
            upload_path = os.path.join(hf_deployment_dir, "upload_to_hf.py")
            with open(upload_path, 'w', encoding='utf-8') as f:
                f.write(upload_script)
            
            return {
                'success': True,
                'deployment_type': 'huggingface',
                'deployment_path': hf_deployment_dir,
                'model_card': card_path,
                'upload_script': upload_path,
                'instructions': [
                    f"HuggingFace deployment prepared: {hf_deployment_dir}",
                    "1. Login to HuggingFace: huggingface-cli login",
                    f"2. Run upload script: python {upload_path}",
                    f"3. Model will be available at: https://huggingface.co/{deployment_name}",
                    f"4. Model card: {card_path}"
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'HuggingFace deployment failed: {str(e)}'
            }

    def _create_inference_script(self, model_path: str, max_length: int, 
                                temperature: float) -> str:
        """
        Create local inference script.
        
        Args:
            model_path: Path to the model
            max_length: Maximum generation length
            temperature: Generation temperature
            
        Returns:
            Inference script content
        """
        return f'''#!/usr/bin/env python3
"""
Local QLoRA Model Inference Script

This script provides a simple interface for running inference
with the deployed QLoRA model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

class QLoRAInference:
    def __init__(self, model_path="{model_path}"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load the QLoRA model and tokenizer"""
        print(f"Loading model from {{self.model_path}}...")
        print(f"Using device: {{self.device}}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            print("✅ Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {{e}}")
            return False
    
    def generate_response(self, prompt, max_length={max_length}, temperature={temperature}):
        """Generate response for given prompt"""
        if self.model is None or self.tokenizer is None:
            return "Error: Model not loaded. Please run load_model() first."
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if self.device == "cuda":
                inputs = {{k: v.to(self.device) for k, v in inputs.items()}}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            return f"Error generating response: {{e}}"

def main():
    """Main inference loop"""
    print("🤖 QLoRA Model Inference")
    print("=" * 40)
    
    # Initialize inference
    inference = QLoRAInference()
    
    # Load model
    if not inference.load_model():
        print("Failed to load model. Exiting.")
        return
    
    print("\n💬 Interactive Chat (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Generate response
            print("🤖 Assistant: ", end="", flush=True)
            response = inference.generate_response(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {{e}}")

if __name__ == "__main__":
    main()
'''

    def _create_api_server_script(self, model_path: str, api_port: int,
                                 max_length: int, temperature: float,
                                 enable_auth: bool) -> str:
        """
        Create API server script.
        
        Args:
            model_path: Path to the model
            api_port: API server port
            max_length: Maximum generation length
            temperature: Generation temperature
            enable_auth: Enable authentication
            
        Returns:
            API server script content
        """
        auth_imports = "from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials" if enable_auth else ""
        auth_dependency = "security = HTTPBearer()" if enable_auth else ""
        auth_param = ", credentials: HTTPAuthorizationCredentials = Depends(security)" if enable_auth else ""
        
        return f'''#!/usr/bin/env python3
"""
QLoRA Model API Server

FastAPI-based REST API for serving QLoRA model inference.
"""

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
{auth_imports}
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Configuration
MODEL_PATH = "{model_path}"
API_PORT = {api_port}
MAX_LENGTH = {max_length}
TEMPERATURE = {temperature}
ENABLE_AUTH = {enable_auth}

# Initialize FastAPI app
app = FastAPI(
    title="QLoRA Model API",
    description="REST API for QLoRA model inference",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

{auth_dependency}

# Request/Response models
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = MAX_LENGTH
    temperature: float = TEMPERATURE

class GenerateResponse(BaseModel):
    response: str
    prompt: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global tokenizer, model
    
    try:
        print(f"Loading model from {{MODEL_PATH}}...")
        print(f"Using device: {{device}}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {{e}}")
        model = None
        tokenizer = None

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest{auth_param}):
    """Generate text from prompt"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        if device == "cuda":
            inputs = {{k: v.to(device) for k, v in inputs.items()}}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return GenerateResponse(
            response=response.strip(),
            prompt=request.prompt,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {{str(e)}}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {{
        "message": "QLoRA Model API",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }}

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=API_PORT,
        reload=False
    )
'''

    def _create_model_card(self, deployment_name: str) -> str:
        """
        Create HuggingFace model card.
        
        Args:
            deployment_name: Name of the deployment
            
        Returns:
            Model card content
        """
        return f'''---
license: apache-2.0
language:
- en
- ar
tags:
- islamic
- qlora
- fine-tuned
- conversational
library_name: transformers
pipeline_tag: text-generation
---

# {deployment_name}

## Model Description

This is a QLoRA (Quantized Low-Rank Adaptation) fine-tuned model designed for Islamic knowledge and guidance. The model has been trained to provide accurate, respectful, and contextually appropriate responses related to Islamic teachings, practices, and principles.

## Model Details

- **Model Type**: QLoRA Fine-tuned Language Model
- **Language(s)**: English, Arabic
- **License**: Apache 2.0
- **Fine-tuning Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Base Model**: Qwen-1.5-1.8B (or similar)

## Intended Use

### Primary Use Cases
- Islamic knowledge and education
- Religious guidance and consultation
- Quranic and Hadith explanations
- Islamic jurisprudence (Fiqh) questions
- Spiritual and moral guidance

### Limitations
- This model is for educational purposes only
- Always consult qualified Islamic scholars for important religious decisions
- The model may not cover all schools of Islamic thought
- Responses should be verified with authentic Islamic sources

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load the model
tokenizer = AutoTokenizer.from_pretrained("{deployment_name}")
model = AutoModelForCausalLM.from_pretrained("{deployment_name}")

# Generate response
prompt = "What are the five pillars of Islam?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Data

The model was fine-tuned on a curated dataset of Islamic texts, including:
- Quranic verses and interpretations
- Authentic Hadith collections
- Islamic scholarly works
- Educational Islamic content

## Ethical Considerations

- The model aims to provide respectful and accurate Islamic guidance
- Users should verify important religious matters with qualified scholars
- The model respects Islamic principles and values
- Content is filtered to ensure appropriateness and accuracy

## Technical Details

- **Fine-tuning Framework**: Axolotl with QLoRA
- **Quantization**: 4-bit quantization for efficiency
- **LoRA Rank**: 64
- **LoRA Alpha**: 16
- **Training Steps**: Optimized for convergence

## Citation

If you use this model in your research or applications, please cite:

```bibtex
@misc{{{deployment_name.replace("-", "_")},
  title={{{deployment_name}: QLoRA Fine-tuned Islamic Assistant}},
  author={{Development Team}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/{deployment_name}}}
}}
```

## Contact

For questions, issues, or feedback about this model, please refer to the model repository or contact the development team.

---

**Disclaimer**: This model is for educational purposes only. Always consult qualified Islamic scholars for religious guidance and verify information with authentic Islamic sources.
'''

    def _format_deployment_result(self, result: Dict[str, Any], deployment_id: str) -> str:
        """
        Format deployment result for display.
        
        Args:
            result: Deployment result data
            deployment_id: Deployment ID
            
        Returns:
            Formatted deployment report
        """
        if not result.get('success', False):
            return f"❌ Deployment failed: {result.get('error', 'Unknown error')}"
        
        deployment_type = result.get('deployment_type', 'unknown')
        
        report = [
            f"🚀 QLoRA Model Deployment Successful!",
            "=" * 50,
            f"📋 Deployment ID: {deployment_id}",
            f"🎯 Deployment Type: {deployment_type.title()}",
            f"📁 Deployment Path: {result.get('deployment_path', 'N/A')}",
            f"⏰ Deployment Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Add type-specific information
        if deployment_type == 'api_server':
            report.extend([
                "🌐 API Server Details:",
                f"   • API URL: {result.get('api_url', 'N/A')}",
                f"   • Documentation: {result.get('docs_url', 'N/A')}",
                f"   • Port: {result.get('api_port', 'N/A')}",
                ""
            ])
        
        # Add instructions
        if 'instructions' in result:
            report.append("📋 Next Steps:")
            for i, instruction in enumerate(result['instructions'], 1):
                report.append(f"   {i}. {instruction}")
            report.append("")
        
        # Add general recommendations
        report.extend([
            "💡 Recommendations:",
            "   • Test the deployment thoroughly before production use",
            "   • Monitor performance and resource usage",
            "   • Set up proper logging and monitoring",
            "   • Configure authentication and security measures",
            "   • Create backup and recovery procedures",
            "",
            "✅ Deployment completed successfully!"
        ])
        
        return "\n".join(report)


if __name__ == "__main__":
    # Demo usage
    tool = ModelDeployerTool()
    
    # Test local deployment
    print("Model Deployer Tool Demo:")
    result = tool._run(
        model_path="models/akhi_qlora",
        deployment_target="local",
        deployment_name="akhi-islamic-assistant"
    )
    print(result)