#!/usr/bin/env python3
"""
Sophia LLM Setup Script
Initializes the development environment for training a philosophical/scientific LLM
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import yaml
from rich.console import Console
from rich.progress import track
from rich.panel import Panel

console = Console()

def check_python_version():
    """Ensure Python 3.8+ is being used"""
    if sys.version_info < (3, 8):
        console.print("❌ Python 3.8+ required. Current version:", sys.version, style="bold red")
        sys.exit(1)
    console.print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected", style="bold green")

def check_cuda_availability():
    """Check if CUDA is available for GPU training"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            console.print(f"✅ CUDA available with {gpu_count} GPU(s): {gpu_name}", style="bold green")
            return True
        else:
            console.print("⚠️  CUDA not available. Training will use CPU (much slower)", style="bold yellow")
            return False
    except ImportError:
        console.print("ℹ️  PyTorch not installed yet - will check CUDA after installation", style="bold blue")
        return None

def install_dependencies():
    """Install required Python packages"""
    console.print("📦 Installing dependencies...", style="bold blue")
    
    # Get the directory of this script
    script_dir = Path(__file__).parent.parent
    requirements_file = script_dir / "requirements.txt"
    
    if not requirements_file.exists():
        console.print(f"❌ Requirements file not found: {requirements_file}", style="bold red")
        sys.exit(1)
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        console.print("✅ Dependencies installed successfully", style="bold green")
    except subprocess.CalledProcessError as e:
        console.print(f"❌ Failed to install dependencies: {e}", style="bold red")
        sys.exit(1)

def create_directory_structure():
    """Create necessary directories for the project"""
    console.print("📁 Creating directory structure...", style="bold blue")
    
    base_dir = Path(__file__).parent.parent
    directories = [
        "data/raw",
        "data/processed", 
        "data/datasets",
        "models/checkpoints",
        "models/fine_tuned",
        "logs/training",
        "logs/evaluation",
        "outputs/generated_text",
        "outputs/metrics",
        "cache/huggingface",
        "cache/datasets"
    ]
    
    for directory in track(directories, description="Creating directories..."):
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    console.print("✅ Directory structure created", style="bold green")

def setup_environment_variables():
    """Create .env file with default settings"""
    console.print("🔧 Setting up environment variables...", style="bold blue")
    
    base_dir = Path(__file__).parent.parent
    env_file = base_dir / ".env"
    
    if env_file.exists():
        console.print("ℹ️  .env file already exists, skipping", style="bold yellow")
        return
    
    env_content = """# Sophia LLM Environment Variables

# Hugging Face settings
HF_HOME=./cache/huggingface
TRANSFORMERS_CACHE=./cache/huggingface
HF_DATASETS_CACHE=./cache/datasets

# Training settings
WANDB_PROJECT=sophia-llm
WANDB_ENTITY=mnemia

# Model serving
SOPHIA_MODEL_PATH=./models/fine_tuned/sophia
SOPHIA_API_PORT=8003

# GPU settings
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=INFO
PYTHONPATH=./src:$PYTHONPATH
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    console.print("✅ Environment variables configured", style="bold green")

def download_sample_data():
    """Download some sample philosophical texts for initial testing"""
    console.print("📚 Downloading sample philosophical texts...", style="bold blue")
    
    base_dir = Path(__file__).parent.parent
    sample_dir = base_dir / "data" / "samples"
    sample_dir.mkdir(exist_ok=True)
    
    # Sample texts to download (public domain)
    samples = {
        "plato_republic.txt": "http://www.gutenberg.org/files/1497/1497-0.txt",
        "tao_te_ching.txt": "http://www.gutenberg.org/files/216/216-0.txt", 
        "meditations_marcus_aurelius.txt": "http://www.gutenberg.org/files/2680/2680-0.txt",
        "art_of_war.txt": "http://www.gutenberg.org/files/132/132-0.txt"
    }
    
    try:
        import requests
        for filename, url in track(samples.items(), description="Downloading texts..."):
            file_path = sample_dir / filename
            if not file_path.exists():
                response = requests.get(url)
                response.raise_for_status()
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
        
        console.print("✅ Sample texts downloaded", style="bold green")
    except Exception as e:
        console.print(f"⚠️  Could not download sample texts: {e}", style="bold yellow")
        console.print("You can manually download them later", style="italic")

def create_initial_scripts():
    """Create helpful utility scripts"""
    console.print("🔨 Creating utility scripts...", style="bold blue")
    
    base_dir = Path(__file__).parent.parent
    scripts_dir = base_dir / "scripts"
    
    # Training script
    train_script = scripts_dir / "train.py"
    if not train_script.exists():
        train_content = '''#!/usr/bin/env python3
"""
Sophia LLM Training Script
Run with: python scripts/train.py
"""

import os
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def main():
    print("🧠 Sophia LLM Training")
    print("This script will be implemented as the training pipeline develops.")
    print("For now, please follow the implementation roadmap.")
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "sophia_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"✅ Configuration loaded: {config['project_name']}")
    else:
        print("❌ Configuration file not found")

if __name__ == "__main__":
    main()
'''
        with open(train_script, 'w') as f:
            f.write(train_content)
        train_script.chmod(0o755)
    
    # Data collection script
    collect_script = scripts_dir / "collect_data.py"
    if not collect_script.exists():
        collect_content = '''#!/usr/bin/env python3
"""
Sophia LLM Data Collection Script
Run with: python scripts/collect_data.py
"""

import os
import sys
from pathlib import Path

def main():
    print("📚 Sophia LLM Data Collection")
    print("This script will collect philosophical and scientific texts.")
    print("Implementation coming soon as per the roadmap.")

if __name__ == "__main__":
    main()
'''
        with open(collect_script, 'w') as f:
            f.write(collect_content)
        collect_script.chmod(0o755)
    
    console.print("✅ Utility scripts created", style="bold green")

def print_next_steps():
    """Display next steps to the user"""
    next_steps = """
🎯 Sophia LLM Setup Complete!

Next Steps:
1. Review the configuration in configs/sophia_config.yaml
2. Check out the implementation roadmap in implementation-roadmap.md
3. Start with data collection: python scripts/collect_data.py
4. Begin training preparation: python scripts/train.py

Key Files Created:
├── configs/sophia_config.yaml - Main configuration
├── requirements.txt - Python dependencies  
├── .env - Environment variables
├── scripts/train.py - Training script (starter)
├── scripts/collect_data.py - Data collection (starter)
└── data/samples/ - Sample philosophical texts

Development Workflow:
1. Data Collection (Week 1-4)
2. Model Development (Week 5-12) 
3. Specialized Training (Week 13-16)
4. MNEMIA Integration (Week 17-20)

Happy philosophizing! 🧠✨
"""
    
    console.print(Panel(next_steps, title="Setup Complete", border_style="green"))

def main():
    parser = argparse.ArgumentParser(description="Setup Sophia LLM development environment")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-data", action="store_true", help="Skip sample data download")
    args = parser.parse_args()
    
    console.print(Panel("🧠 Sophia LLM Setup", subtitle="Wisdom through Philosophy & Science", border_style="blue"))
    
    # Setup steps
    check_python_version()
    
    if not args.skip_deps:
        install_dependencies()
        check_cuda_availability()
    
    create_directory_structure()
    setup_environment_variables()
    create_initial_scripts()
    
    if not args.skip_data:
        download_sample_data()
    
    print_next_steps()

if __name__ == "__main__":
    main() 