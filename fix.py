# File: fix_demo_imports.py
"""Fix for demo import errors - run this to patch missing components"""

import sys
import os
from pathlib import Path

def create_missing_modules():
    """Create missing modules and directories"""
    
    print("🔧 Fixing demo import errors...")
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create __init__.py files for proper package structure
    init_files = [
        "data/__init__.py",
        "config/__init__.py",
        "core/__init__.py"
    ]
    
    for init_file in init_files:
        init_path = Path(init_file)
        init_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not init_path.exists():
            init_path.write_text("# Package initialization\n")
            print(f"✅ Created {init_file}")
    
    # Add current directory to Python path for imports
    current_dir = Path.cwd()
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
        print(f"✅ Added {current_dir} to Python path")
    
    print("🎉 Import fixes applied successfully!")

def verify_imports():
    """Verify that the problematic imports now work"""
    
    print("\n🧪 Testing imports...")
    
    try:
        from config.data_config import PreprocessingConfig
        print("✅ PreprocessingConfig import successful")
    except ImportError as e:
        print(f"❌ PreprocessingConfig import failed: {e}")
    
    try:
        from data.integrated_data_pipeline import IntegratedDataPipeline
        print("✅ IntegratedDataPipeline import successful")
    except ImportError as e:
        print(f"❌ IntegratedDataPipeline import failed: {e}")
    
    try:
        # Test creating instances
        config = PreprocessingConfig()
        pipeline = IntegratedDataPipeline()
        print("✅ Object creation successful")
    except Exception as e:
        print(f"❌ Object creation failed: {e}")

def apply_quick_fix():
    """Apply quick fix and verify"""
    create_missing_modules()
    verify_imports()
    
    print("\n📋 Next steps:")
    print("1. Run your demo again")
    print("2. If you still get errors, check the specific import statements")
    print("3. Make sure all files are in the correct directories")

if __name__ == "__main__":
    apply_quick_fix()