"""Shared, framework-independent IR utilities.

Submodules are imported directly so each carries only its own dependencies and
stays decoupled (e.g. ``dataset_loader`` needs ``ir_datasets``; ``errors`` needs
``fastapi``). Import what you need:

    from shared.ir_common.dataset_loader import DatasetLoader
    from shared.ir_common.errors import install_error_handlers
    from shared.ir_common.config import find_repo_env
"""
