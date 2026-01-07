#!/usr/bin/env python3
"""
Decision Intelligence Studio - End-to-End Pipeline Orchestration

Runs the complete pipeline:
1. Generate sample data
2. Create canonical dataset  
3. Run data quality checks
4. Estimate causal effects (ATE & CATE)
5. Run refutation tests
6. Generate reports

Usage:
    python run_pipeline.py              # Run full pipeline
    python run_pipeline.py --quick      # Skip refutation tests
    python run_pipeline.py --step 4     # Run only step 4
"""
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils.logging_config import get_logger
from src.data.generate_sample_data import main as generate_data
from src.data.create_canonical import main as create_canonical
from src.data.data_quality_checks import main as run_dq_checks
from src.causal.estimation import main as run_estimation
from src.causal.refutation import main as run_refutation

logger = get_logger(__name__)


class PipelineOrchestrator:
    """Orchestrates the full Decision Intelligence pipeline"""
    
    def __init__(self, skip_refutation: bool = False):
        self.skip_refutation = skip_refutation
        self.start_time = None
        self.step_times = {}
        self.results = {
            'success': False,
            'steps_completed': [],
            'steps_failed': [],
        }
    
    def print_banner(self):
        """Print welcome banner"""
        banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║         DECISION INTELLIGENCE STUDIO - PIPELINE RUNNER          ║
║                                                                  ║
║              Causal Inference for Data-Driven Decisions          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""
        print(banner)
        logger.info(f"Pipeline started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_step(self, step_num: int, step_name: str, step_func) -> bool:
        """
        Run a single pipeline step
        
        Args:
            step_num: Step number
            step_name: Human-readable step name
            step_func: Function to execute
            
        Returns:
            bool: Success status
        """
        logger.info("\n")
        logger.info("╔" + "═" * 70 + "╗")
        logger.info(f"║  STEP {step_num}: {step_name:<60}║")
        logger.info("╚" + "═" * 70 + "╝")
        
        step_start = time.time()
        
        try:
            success = step_func()
            
            if success is None:
                success = True
            
            step_time = time.time() - step_start
            self.step_times[step_name] = step_time
            
            if success:
                self.results['steps_completed'].append(step_name)
                logger.info(f"\n✓ Step {step_num} completed in {step_time:.1f}s")
                return True
            else:
                self.results['steps_failed'].append(step_name)
                logger.error(f"\n✗ Step {step_num} failed")
                return False
                
        except Exception as e:
            step_time = time.time() - step_start
            self.step_times[step_name] = step_time
            self.results['steps_failed'].append(step_name)
            logger.error(f"\n✗ Step {step_num} failed with error: {e}")
            logger.exception(e)
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete pipeline"""
        self.start_time = time.time()
        
        # Step 1: Generate sample data
        if not self.run_step(1, "Generate Sample Data", generate_data):
            return False
        
        # Step 2: Create canonical dataset
        if not self.run_step(2, "Create Canonical Dataset", create_canonical):
            return False
        
        # Step 3: Data quality checks
        if not self.run_step(3, "Data Quality Checks", run_dq_checks):
            return False
        
        # Step 4: Causal estimation
        if not self.run_step(4, "Causal Estimation (ATE & CATE)", run_estimation):
            return False
        
        # Step 5: Refutation tests (optional)
        if not self.skip_refutation:
            if not self.run_step(5, "Refutation Tests", run_refutation):
                logger.warning("Refutation tests failed, but continuing...")
        else:
            logger.info("\n⊗ Skipping refutation tests (--quick mode)")
        
        self.results['success'] = True
        return True
    
    def run_single_step(self, step_num: int) -> bool:
        """Run a single step of the pipeline"""
        steps = {
            1: ("Generate Sample Data", generate_data),
            2: ("Create Canonical Dataset", create_canonical),
            3: ("Data Quality Checks", run_dq_checks),
            4: ("Causal Estimation", run_estimation),
            5: ("Refutation Tests", run_refutation),
        }
        
        if step_num not in steps:
            logger.error(f"Invalid step number: {step_num}")
            return False
        
        step_name, step_func = steps[step_num]
        self.start_time = time.time()
        
        success = self.run_step(step_num, step_name, step_func)
        self.results['success'] = success
        
        return success
    
    def print_summary(self):
        """Print pipeline execution summary"""
        total_time = time.time() - self.start_time
        
        print("\n")
        logger.info("╔" + "═" * 70 + "╗")
        logger.info("║" + " " * 20 + "PIPELINE SUMMARY" + " " * 34 + "║")
        logger.info("╚" + "═" * 70 + "╝")
        
        # Overall status
        if self.results['success']:
            logger.info("✓ Pipeline completed successfully!")
        else:
            logger.error("✗ Pipeline failed")
        
        logger.info(f"\nTotal execution time: {total_time:.1f}s")
        
        # Step breakdown
        logger.info("\nStep execution times:")
        for step_name, step_time in self.step_times.items():
            status = "✓" if step_name in self.results['steps_completed'] else "✗"
            logger.info(f"  {status} {step_name:<40} {step_time:>6.1f}s")
        
        # What to do next
        if self.results['success']:
            logger.info("\n" + "═" * 72)
            logger.info("NEXT STEPS:")
            logger.info("═" * 72)
            logger.info("1. Start the API server:")
            logger.info("   $ python -m src.api.main")
            logger.info("")
            logger.info("2. View results:")
            logger.info("   $ python -c \"import pandas as pd; print(pd.read_parquet('data/outputs/uplift_scores.parquet').head())\"")
            logger.info("")
            logger.info("3. Test the API:")
            logger.info("   $ curl http://localhost:8000/health")
            logger.info("   $ curl http://localhost:8000/stats")
            logger.info("═" * 72)
        
        print("\n")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run Decision Intelligence Studio Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py              # Run full pipeline
  python run_pipeline.py --quick      # Skip refutation tests
  python run_pipeline.py --step 4     # Run only step 4
        """
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Skip refutation tests for faster execution'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run only a specific step (1-5)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(skip_refutation=args.quick)
    
    # Print banner
    orchestrator.print_banner()
    
    # Run pipeline
    if args.step:
        success = orchestrator.run_single_step(args.step)
    else:
        success = orchestrator.run_full_pipeline()
    
    # Print summary
    orchestrator.print_summary()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()