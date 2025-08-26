#!/usr/bin/env python3
"""
BondX AI Autonomous Training System Launcher

This script provides an easy way to start the autonomous training system
with different options and configurations.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Optional

def print_banner():
    """Print the BondX AI banner"""
    print("\n" + "="*80)
    print("🚀 BONDX AI AUTONOMOUS TRAINING SYSTEM")
    print("="*80)
    print("🤖 Autonomous Corporate Bond Analytics & ML Training")
    print("📊 Synthetic Dataset Generation & Quality Validation")
    print("🎯 Continuous Training Until Convergence")
    print("="*80)

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - MISSING")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies available")
    return True

def check_bondx_modules():
    """Check if BondX modules are available"""
    print("\n🔍 Checking BondX modules...")
    
    bondx_path = Path(__file__).parent / "bondx"
    if not bondx_path.exists():
        print("❌ BondX modules not found")
        print("   Expected path: " + str(bondx_path))
        return False
    
    print("✅ BondX modules found")
    return True

def start_autonomous_training(config_path: Optional[str] = None):
    """Start the autonomous training system"""
    print("\n🚀 Starting BondX AI Autonomous Training...")
    
    try:
        # Import and start trainer
        from bondx_ai_autonomous_trainer import BondXAIAutonomousTrainer
        
        trainer = BondXAIAutonomousTrainer(config_path)
        print(f"✅ Trainer initialized with config: {config_path or 'default'}")
        
        # Start training loop
        print("🔄 Starting autonomous training loop...")
        asyncio.run(trainer.run_autonomous_training_loop())
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error starting training: {e}")
        print("💡 Check the logs for more details")
        return False
    
    return True

def start_dashboard(output_dir: str = "autonomous_training_output"):
    """Start the monitoring dashboard"""
    print(f"\n📊 Starting BondX AI Dashboard for: {output_dir}")
    
    try:
        # Import and start dashboard
        from bondx_ai_dashboard import BondXAIDashboard
        
        dashboard = BondXAIDashboard(output_dir)
        print("✅ Dashboard initialized")
        
        # Start monitoring
        print("🔄 Starting real-time monitoring...")
        asyncio.run(dashboard.start_monitoring())
        
        print("✅ Dashboard session ended")
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        print("💡 Check the logs for more details")
        return False
    
    return True

def run_tests():
    """Run the comprehensive test suite"""
    print("\n🧪 Running BondX AI Test Suite...")
    
    try:
        # Import and run tests
        from test_autonomous_training import main as run_test_suite
        
        run_test_suite()
        print("✅ Test suite completed")
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False
    
    return True

def generate_sample_dataset():
    """Generate a sample synthetic dataset"""
    print("\n📊 Generating sample synthetic dataset...")
    
    try:
        # Import and run dataset generator
        from data.synthetic.generate_enhanced_synthetic_dataset import (
            generate_enhanced_synthetic_dataset, export_datasets
        )
        
        # Generate dataset
        bonds_data = generate_enhanced_synthetic_dataset(50)  # Smaller for testing
        
        # Export to current directory
        output_dir = Path("sample_dataset_output")
        export_datasets(bonds_data, output_dir)
        
        print(f"✅ Sample dataset generated in: {output_dir}")
        print(f"   - CSV: {output_dir / 'enhanced_bonds_150plus.csv'}")
        print(f"   - JSONL: {output_dir / 'enhanced_bonds_150plus.jsonl'}")
        print(f"   - Metadata: {output_dir / 'dataset_metadata.json'}")
        
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="BondX AI Autonomous Training System Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start autonomous training with default config
  python start_bondx_ai.py --train
  
  # Start training with custom config
  python start_bondx_ai.py --train --config my_config.yaml
  
  # Start monitoring dashboard
  python start_bondx_ai.py --dashboard
  
  # Run test suite
  python start_bondx_ai.py --test
  
  # Generate sample dataset
  python start_bondx_ai.py --generate-dataset
  
  # Start training and dashboard simultaneously
  python start_bondx_ai.py --train --dashboard
        """
    )
    
    parser.add_argument(
        '--train', 
        action='store_true',
        help='Start autonomous training system'
    )
    
    parser.add_argument(
        '--dashboard', 
        action='store_true',
        help='Start monitoring dashboard'
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Run comprehensive test suite'
    )
    
    parser.add_argument(
        '--generate-dataset', 
        action='store_true',
        help='Generate sample synthetic dataset'
    )
    
    parser.add_argument(
        '--config', 
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str,
        default='autonomous_training_output',
        help='Output directory for training results'
    )
    
    parser.add_argument(
        '--check-only', 
        action='store_true',
        help='Only check dependencies and modules, don\'t start anything'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install missing packages.")
        sys.exit(1)
    
    # Check BondX modules
    if not check_bondx_modules():
        print("\n❌ BondX modules not found. Please ensure the project structure is correct.")
        sys.exit(1)
    
    # If only checking, exit here
    if args.check_only:
        print("\n✅ All checks passed! System is ready to run.")
        return
    
    # Validate arguments
    if not any([args.train, args.dashboard, args.test, args.generate_dataset]):
        print("\n❌ No action specified. Use --help to see available options.")
        sys.exit(1)
    
    # Run requested actions
    success = True
    
    if args.generate_dataset:
        success &= generate_sample_dataset()
    
    if args.test:
        success &= run_tests()
    
    if args.train:
        success &= start_autonomous_training(args.config)
    
    if args.dashboard:
        success &= start_dashboard(args.output_dir)
    
    # Final status
    if success:
        print("\n🎉 All requested actions completed successfully!")
    else:
        print("\n⚠️  Some actions encountered issues. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Launcher stopped by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
