# Transformer Builder - Colab Templates Project Context

## Overview

Production-grade testing and training infrastructure for transformer models exported from Transformer Builder (transformer-builder.com). Provides two-notebook architecture for model validation (template.ipynb) and training utilities (training.ipynb) optimized for Google Colab's constrained environment.

## Vision & Goals

**Primary Mission**: Transform basic training utilities into production-ready MLOps infrastructure while maintaining beginner-friendly experience and Colab compatibility.

**Current State (v3.4.0)**: Functional but basic training framework with separated validation/training notebooks to prevent NumPy corruption issues.

**Target State**: Enterprise-grade ML training platform with experiment tracking, model registry, reproducibility framework, and advanced training features—all working within Colab's 12GB GPU and 12-hour session constraints.

## Target Users

1. **ML Practitioners**: Building custom transformers via Transformer Builder UI, need production training workflows
2. **ML Engineers**: Require experiment tracking, checkpoint management, model export for deployment
3. **Researchers**: Need reproducibility, hyperparameter optimization, benchmarking capabilities
4. **Beginners**: Visual transformer builders who need guided training setup with progressive disclosure

## Success Criteria

- W&B integration tracks all experiments with <5 lines of user code
- HuggingFace Hub auto-publishes trained models with proper metadata
- Training survives Colab session timeouts via Google Drive checkpoints
- Real dataset integration (HF datasets) replaces synthetic data
- Hyperparameter search finds optimal configs 50% faster via pruning
- Model export produces ONNX/TorchScript/PyTorch formats automatically
- All features work in Colab free tier (12GB GPU, 12-hour sessions)

## Key Constraints

- **Zero-installation in template.ipynb**: Cannot install packages to avoid NumPy corruption
- **Colab resource limits**: 12GB GPU memory, 12-hour max runtime, ephemeral storage
- **Session timeout resilience**: Must checkpoint to Google Drive every epoch
- **Beginner-friendly**: Progressive disclosure, optional advanced features, clear error messages
- **Architecture-agnostic**: Support decoder-only (GPT), encoder-only (BERT), encoder-decoder (T5)

## Timeline

- **Phase 1 (MLOps Foundation)**: 16-18 hours - W&B, HF Hub, reproducibility
- **Phase 2 (ML Training)**: 24-26 hours - Datasets, metrics, training loop improvements
- **Phase 3 (Production Features)**: 16-18 hours - Checkpoints, export, pipeline
- **Phase 4 (Code Quality)**: 8-10 hours - Deduplication, error handling, optimization

**Total Estimated Effort**: 64-72 hours (~9-10 working days)
