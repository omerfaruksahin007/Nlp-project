"""
PROMPT 12 - Ablation Study Framework
Turkish Legal RAG - Experiment Management System

This package provides tools for running and comparing ablation studies:
- experiment_configs.py: Configuration definitions for 5 experiments
- metrics_collector.py: Collection and storage of evaluation metrics
- results_visualizer.py: Comparison tables and report generation
- ablation_runner.py: Main CLI orchestrator

Usage:
    python ablation_runner.py --all              # Run all experiments
    python ablation_runner.py --exp baseline     # Run specific experiment
    python ablation_runner.py --compare          # Generate reports
    python ablation_runner.py --list             # List experiments
"""

__version__ = "1.0.0"
__author__ = "Turkish Legal RAG Team"
