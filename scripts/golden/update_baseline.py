#!/usr/bin/env python3
"""
Baseline Update Script for Golden Dataset Vault

Updates baselines for golden datasets with explicit maintainer approval.
Maintains a changelog of all updates for audit purposes.

Usage:
    python update_baseline.py --dataset v1_dirty --approve --reason "Policy update"
    python update_baseline.py --dataset v1_dirty --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.DEBUG
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(file)s:%(lineno)d - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('baseline_update.log')
        ]
    )

def load_changelog(changelog_path: Path) -> List[Dict[str, Any]]:
    """Load existing changelog or create new one."""
    if changelog_path.exists():
        try:
            with open(changelog_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logging.warning("Could not load existing changelog, creating new one")
    
    return []

def save_changelog(changelog: List[Dict[str, Any]], changelog_path: Path):
    """Save changelog to file."""
    changelog_path.parent.mkdir(parents=True, exist_ok=True)
    with open(changelog_path, 'w') as f:
        json.dump(changelog, f, indent=2)

def add_changelog_entry(dataset_name: str, reason: str, policy_version: str, 
                       reviewer: str, dry_run: bool = False) -> Dict[str, Any]:
    """Add a new entry to the changelog."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_name,
        'reason': reason,
        'policy_version': policy_version,
        'reviewer': reviewer,
        'dry_run': dry_run,
        'action': 'baseline_update'
    }
    
    if dry_run:
        entry['status'] = 'dry_run'
    else:
        entry['status'] = 'approved'
    
    return entry

def get_policy_version() -> str:
    """Get current policy version from quality policy file."""
    policy_file = Path("bondx/quality/config/quality_policy.yaml")
    
    if not policy_file.exists():
        return "unknown"
    
    try:
        with open(policy_file, 'r') as f:
            content = f.read()
            # Look for version-like patterns in the file
            # This is a simple heuristic - in practice you might have a proper version field
            if "version:" in content:
                for line in content.split('\n'):
                    if line.strip().startswith('version:'):
                        return line.split(':')[1].strip()
            return "1.0.0"  # Default version
    except Exception as e:
        logging.warning(f"Could not read policy version: {e}")
        return "unknown"

def copy_validation_outputs_to_baseline(dataset_name: str, golden_dir: Path, 
                                       baseline_dir: Path) -> bool:
    """Copy validation outputs to baseline directory."""
    logger = logging.getLogger(__name__)
    
    source_dir = golden_dir / "validation_outputs" / dataset_name
    target_dir = baseline_dir / dataset_name
    
    if not source_dir.exists():
        logger.error(f"Validation outputs not found: {source_dir}")
        logger.error("Run validate_golden.py first to generate outputs")
        return False
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    files_to_copy = ['last_run_report.json', 'metrics.json', 'summary.txt']
    copied_files = []
    
    for filename in files_to_copy:
        source_file = source_dir / filename
        target_file = target_dir / filename
        
        if source_file.exists():
            try:
                import shutil
                shutil.copy2(source_file, target_file)
                copied_files.append(filename)
                logger.info(f"Copied {filename} to baseline")
            except Exception as e:
                logger.error(f"Failed to copy {filename}: {e}")
                return False
        else:
            logger.warning(f"Source file not found: {source_file}")
    
    if not copied_files:
        logger.error("No files were copied to baseline")
        return False
    
    # Create baseline metadata
    metadata = {
        'baseline_created': datetime.now().isoformat(),
        'dataset_name': dataset_name,
        'files': copied_files,
        'source_validation': str(source_dir),
        'policy_version': get_policy_version()
    }
    
    metadata_file = target_dir / "baseline_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Baseline metadata saved to {metadata_file}")
    return True

def validate_baseline_update(dataset_name: str, golden_dir: Path, 
                           baseline_dir: Path) -> bool:
    """Validate that baseline update is safe."""
    logger = logging.getLogger(__name__)
    
    # Check if dataset exists
    dataset_path = golden_dir / dataset_name / f"{dataset_name}.csv"
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return False
    
    # Check if validation outputs exist
    outputs_dir = golden_dir / "validation_outputs" / dataset_name
    if not outputs_dir.exists():
        logger.error(f"Validation outputs not found: {outputs_dir}")
        logger.error("Run validate_golden.py first")
        return False
    
    # Check if baseline already exists
    existing_baseline = baseline_dir / dataset_name
    if existing_baseline.exists():
        logger.info(f"Updating existing baseline for {dataset_name}")
        
        # Check if there are significant changes
        try:
            old_metadata_file = existing_baseline / "baseline_metadata.json"
            if old_metadata_file.exists():
                with open(old_metadata_file, 'r') as f:
                    old_metadata = json.load(f)
                
                old_policy = old_metadata.get('policy_version', 'unknown')
                current_policy = get_policy_version()
                
                if old_policy != current_policy:
                    logger.info(f"Policy version changed: {old_policy} -> {current_policy}")
                else:
                    logger.warning("Policy version unchanged - ensure this update is intentional")
        except Exception as e:
            logger.warning(f"Could not check policy version: {e}")
    else:
        logger.info(f"Creating new baseline for {dataset_name}")
    
    return True

def dry_run_baseline_update(dataset_name: str, golden_dir: Path, 
                           baseline_dir: Path) -> Dict[str, Any]:
    """Perform a dry run of baseline update to show what would change."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"DRY RUN: Simulating baseline update for {dataset_name}")
    
    # Check what would be copied
    source_dir = golden_dir / "validation_outputs" / dataset_name
    target_dir = baseline_dir / dataset_name
    
    if not source_dir.exists():
        return {
            'status': 'error',
            'message': f'Validation outputs not found: {source_dir}'
        }
    
    # List files that would be copied
    files_to_copy = ['last_run_report.json', 'metrics.json', 'summary.txt']
    available_files = []
    missing_files = []
    
    for filename in files_to_copy:
        if (source_dir / filename).exists():
            available_files.append(filename)
        else:
            missing_files.append(filename)
    
    # Check if baseline already exists
    baseline_exists = target_dir.exists()
    
    return {
        'status': 'dry_run',
        'dataset': dataset_name,
        'baseline_exists': baseline_exists,
        'files_to_copy': available_files,
        'missing_files': missing_files,
        'source_dir': str(source_dir),
        'target_dir': str(target_dir),
        'policy_version': get_policy_version()
    }

def main():
    """Main baseline update function."""
    parser = argparse.ArgumentParser(description="Update golden dataset baselines")
    parser.add_argument('--dataset', required=True, help='Dataset name to update (e.g., v1_dirty)')
    parser.add_argument('--approve', action='store_true', help='Explicit approval flag required')
    parser.add_argument('--reason', required=True, help='Reason for baseline update')
    parser.add_argument('--reviewer', required=True, help='Name of reviewer/maintainer')
    parser.add_argument('--dry-run', action='store_true', help='Show what would change without updating')
    parser.add_argument('--golden-dir', default='data/golden', help='Golden datasets directory')
    parser.add_argument('--baseline-dir', default='data/golden/baselines', help='Baselines directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not args.approve and not args.dry_run:
        logger.error("Either --approve or --dry-run must be specified")
        sys.exit(1)
    
    if not args.reason or not args.reason.strip():
        logger.error("--reason is required and cannot be empty")
        sys.exit(1)
    
    if not args.reviewer or not args.reviewer.strip():
        logger.error("--reviewer is required and cannot be empty")
        sys.exit(1)
    
    golden_dir = Path(args.golden_dir)
    baseline_dir = Path(args.baseline_dir)
    
    if not golden_dir.exists():
        logger.error(f"Golden directory not found: {golden_dir}")
        sys.exit(1)
    
    # Check if dataset exists
    dataset_path = golden_dir / args.dataset
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    
    logger.info(f"Processing dataset: {args.dataset}")
    logger.info(f"Reason: {args.reason}")
    logger.info(f"Reviewer: {args.reviewer}")
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'APPROVED UPDATE'}")
    
    if args.dry_run:
        # Perform dry run
        result = dry_run_baseline_update(args.dataset, golden_dir, baseline_dir)
        
        print("\n" + "="*60)
        print("DRY RUN RESULTS")
        print("="*60)
        print(f"Dataset: {result['dataset']}")
        print(f"Baseline exists: {result['baseline_exists']}")
        print(f"Policy version: {result['policy_version']}")
        print(f"Files to copy: {', '.join(result['files_to_copy'])}")
        
        if result['missing_files']:
            print(f"Missing files: {', '.join(result['missing_files'])}")
        
        print(f"\nSource: {result['source_dir']}")
        print(f"Target: {result['target_dir']}")
        
        if result['status'] == 'error':
            print(f"\n❌ ERROR: {result['message']}")
            sys.exit(1)
        else:
            print("\n✅ Dry run completed successfully")
            print("Use --approve to perform actual update")
        
        sys.exit(0)
    
    # Perform actual update
    logger.info("Performing baseline update...")
    
    # Validate update is safe
    if not validate_baseline_update(args.dataset, golden_dir, baseline_dir):
        logger.error("Baseline update validation failed")
        sys.exit(1)
    
    # Copy files to baseline
    if not copy_validation_outputs_to_baseline(args.dataset, golden_dir, baseline_dir):
        logger.error("Failed to copy files to baseline")
        sys.exit(1)
    
    # Update changelog
    changelog_path = golden_dir / "CHANGELOG.json"
    changelog = load_changelog(changelog_path)
    
    policy_version = get_policy_version()
    new_entry = add_changelog_entry(
        dataset_name=args.dataset,
        reason=args.reason,
        policy_version=policy_version,
        reviewer=args.reviewer,
        dry_run=False
    )
    
    changelog.append(new_entry)
    save_changelog(changelog, changelog_path)
    
    logger.info(f"Changelog updated: {changelog_path}")
    
    # Print success message
    print("\n" + "="*60)
    print("BASELINE UPDATE COMPLETED")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Reason: {args.reason}")
    print(f"Reviewer: {args.reviewer}")
    print(f"Policy Version: {policy_version}")
    print(f"Timestamp: {new_entry['timestamp']}")
    print(f"Changelog: {changelog_path}")
    print("\n✅ Baseline updated successfully!")
    
    # Verify update
    baseline_path = baseline_dir / args.dataset
    if baseline_path.exists():
        print(f"Baseline location: {baseline_path}")
        
        # List files in baseline
        baseline_files = list(baseline_path.glob('*'))
        if baseline_files:
            print("Baseline files:")
            for file in baseline_files:
                print(f"  - {file.name}")
    
    logger.info("Baseline update completed successfully")

if __name__ == "__main__":
    main()
