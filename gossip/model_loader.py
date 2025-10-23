"""
Model loader for downloading and loading PyTorch models from Google Cloud Storage.

Supports:
- Downloading models from GCS to local cache
- Loading models into memory with PyTorch
- Unloading models to free GPU/CPU memory
- Local caching to avoid re-downloading
"""

import os
import shutil
import logging
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from google.cloud import storage


logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Manages downloading and loading PyTorch models from Google Cloud Storage.
    """

    def __init__(
        self,
        gcs_bucket: str = "remote_model",
        cache_dir: str = "/tmp/model_cache",
        device: str = "cpu"
    ):
        """
        Initialize the model loader.

        Args:
            gcs_bucket: GCS bucket name (without gs:// prefix)
            cache_dir: Local directory for caching downloaded models
            device: PyTorch device ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.gcs_bucket = gcs_bucket
        self.cache_dir = Path(cache_dir)
        self.device = device

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Storage client (will use default credentials)
        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(gcs_bucket)
            logger.info(f"Connected to GCS bucket: {gcs_bucket}")
        except Exception as e:
            logger.warning(f"Failed to connect to GCS: {e}. Will use local models only.")
            self.storage_client = None
            self.bucket = None

        # Cache of loaded models {model_id: model_object}
        self.loaded_models: Dict[str, Any] = {}

    def _download_from_gcs(self, model_id: str) -> Path:
        """
        Download a model from GCS to local cache.

        Args:
            model_id: Model identifier (e.g., 'opt-1.3b')

        Returns:
            Path to local model directory
        """
        local_model_path = self.cache_dir / model_id

        # Check if already cached
        if local_model_path.exists():
            logger.info(f"Model {model_id} already cached at {local_model_path}")
            return local_model_path

        if not self.bucket:
            raise RuntimeError(f"Cannot download {model_id}: GCS client not initialized")

        logger.info(f"Downloading model {model_id} from gs://{self.gcs_bucket}/{model_id}")

        # Create temporary directory for download
        temp_dir = local_model_path.with_suffix('.tmp')
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # List all blobs with the model prefix
            blobs = list(self.bucket.list_blobs(prefix=f"{model_id}/"))

            if not blobs:
                raise FileNotFoundError(f"No files found for model {model_id} in GCS bucket")

            logger.info(f"Downloading {len(blobs)} files for {model_id}")

            # Download all files
            for blob in blobs:
                # Get relative path within model directory
                relative_path = blob.name[len(f"{model_id}/"):]
                if not relative_path:  # Skip if it's just the directory
                    continue

                local_file = temp_dir / relative_path
                local_file.parent.mkdir(parents=True, exist_ok=True)

                logger.debug(f"Downloading {blob.name} -> {local_file}")
                blob.download_to_filename(str(local_file))

            # Move from temp to final location atomically
            temp_dir.rename(local_model_path)
            logger.info(f"Successfully downloaded {model_id} to {local_model_path}")

            return local_model_path

        except Exception as e:
            # Clean up temp directory on error
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise RuntimeError(f"Failed to download model {model_id}: {e}")

    def load_model(self, model_id: str) -> bool:
        """
        Load a model into memory.

        Args:
            model_id: Model identifier (e.g., 'opt-1.3b')

        Returns:
            True if successful, False otherwise
        """
        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} already loaded")
            return True

        try:
            # Download from GCS if needed
            local_model_path = self._download_from_gcs(model_id)

            logger.info(f"Loading model {model_id} from {local_model_path}")

            # Load the model with transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
            model = AutoModelForCausalLM.from_pretrained(
                str(local_model_path),
                torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
                device_map=self.device if self.device.startswith('cuda') else None,
                low_cpu_mem_usage=True
            )

            # Move to device if CPU
            if not self.device.startswith('cuda'):
                model = model.to(self.device)

            # Store in cache
            self.loaded_models[model_id] = {
                'model': model,
                'tokenizer': tokenizer,
                'path': local_model_path
            }

            logger.info(f"Successfully loaded model {model_id} on device {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}", exc_info=True)
            return False

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_id: Model identifier

        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.loaded_models:
            logger.warning(f"Model {model_id} not loaded, cannot unload")
            return False

        try:
            logger.info(f"Unloading model {model_id}")

            # Get model info
            model_info = self.loaded_models[model_id]

            # Delete model and tokenizer
            del model_info['model']
            del model_info['tokenizer']

            # Remove from cache
            del self.loaded_models[model_id]

            # Clear CUDA cache if using GPU
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()

            logger.info(f"Successfully unloaded model {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}", exc_info=True)
            return False

    def get_loaded_models(self) -> list[str]:
        """Get list of currently loaded model IDs."""
        return list(self.loaded_models.keys())

    def is_model_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded in memory."""
        return model_id in self.loaded_models

    def get_model_memory_usage(self, model_id: str) -> Optional[int]:
        """
        Get approximate memory usage of a loaded model in bytes.

        Args:
            model_id: Model identifier

        Returns:
            Memory usage in bytes, or None if model not loaded
        """
        if model_id not in self.loaded_models:
            return None

        try:
            model = self.loaded_models[model_id]['model']

            # Calculate total parameters size
            total_bytes = 0
            for param in model.parameters():
                total_bytes += param.nelement() * param.element_size()

            # Add buffer size
            for buffer in model.buffers():
                total_bytes += buffer.nelement() * buffer.element_size()

            return total_bytes

        except Exception as e:
            logger.error(f"Failed to calculate memory usage for {model_id}: {e}")
            return None

    def clear_cache(self, keep_loaded: bool = True):
        """
        Clear the local model cache.

        Args:
            keep_loaded: If True, keep models that are currently loaded in memory
        """
        if keep_loaded:
            # Only delete cached files for models not currently loaded
            for item in self.cache_dir.iterdir():
                if item.is_dir() and item.name not in self.loaded_models:
                    logger.info(f"Removing cached model: {item.name}")
                    shutil.rmtree(item)
        else:
            # Clear entire cache
            logger.info(f"Clearing entire model cache: {self.cache_dir}")
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_available_models(self) -> list[str]:
        """
        Get list of models available in GCS bucket.

        Returns:
            List of model IDs
        """
        if not self.bucket:
            logger.warning("GCS client not initialized, cannot list models")
            return []

        try:
            # List top-level directories in bucket (these are model IDs)
            blobs = self.bucket.list_blobs(delimiter='/')
            # Consume the iterator to get prefixes
            list(blobs)
            prefixes = blobs.prefixes if hasattr(blobs, 'prefixes') else []

            # Remove trailing slashes
            models = [p.rstrip('/') for p in prefixes]
            logger.info(f"Found {len(models)} models in GCS: {models}")
            return models

        except Exception as e:
            logger.error(f"Failed to list models from GCS: {e}")
            return []
