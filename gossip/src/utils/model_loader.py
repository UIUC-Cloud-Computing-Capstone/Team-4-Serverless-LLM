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
import threading
import time
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
        
        # Download progress tracking
        self._download_progress_lock = threading.Lock()

    def _download_with_timeout(self, blob, local_file: Path, timeout_seconds: int = 300) -> bool:
        """
        Download a file with timeout and progress reporting.
        
        Args:
            blob: GCS blob object
            local_file: Local file path
            timeout_seconds: Timeout in seconds (default: 5 minutes)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_size = blob.size
            file_size_mb = file_size / (1024 ** 2)
            
            logger.info(f"Downloading {blob.name} ({file_size_mb:.1f} MB) with {timeout_seconds}s timeout")
            
            # Use a simple timeout approach
            import signal
            import threading
            
            download_success = [False]
            download_error = [None]
            
            def download_worker():
                try:
                    blob.download_to_filename(str(local_file))
                    download_success[0] = True
                except Exception as e:
                    download_error[0] = e
            
            # Start download in a separate thread
            download_thread = threading.Thread(target=download_worker)
            download_thread.daemon = True
            download_thread.start()
            
            # Wait for completion or timeout
            download_thread.join(timeout=timeout_seconds)
            
            if download_thread.is_alive():
                logger.error(f"Download timeout after {timeout_seconds}s for {blob.name}")
                return False
            
            if not download_success[0]:
                logger.error(f"Download failed for {blob.name}: {download_error[0]}")
                return False
            
            # Verify file size
            if local_file.stat().st_size != file_size:
                logger.error(f"File size mismatch: expected {file_size}, got {local_file.stat().st_size}")
                local_file.unlink()
                return False
            
            logger.info(f"Successfully downloaded {blob.name} ({file_size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            logger.error(f"Download with timeout failed for {blob.name}: {e}")
            return False


    def _download_from_gcs(self, model_id: str) -> Path:
        """
        Download a model from GCS to local cache.

        Args:
            model_id: Model identifier (e.g., 'opt-1.3b', 'opt-2.7b')

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

        # GCS path format: models/facebook/opt-1.3b/
        gcs_prefix = f"models/facebook/{model_id}/"
        logger.info(f"Downloading model from gs://{self.gcs_bucket}/{gcs_prefix}")

        # Create temporary directory for download
        temp_dir = local_model_path.with_suffix('.tmp')
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # List all blobs with the model prefix
            blobs = list(self.bucket.list_blobs(prefix=gcs_prefix))

            if not blobs:
                raise FileNotFoundError(f"No files found for model {model_id} in GCS bucket")

            logger.info(f"Downloading {len(blobs)} files for {model_id}")

            # Calculate total download size
            total_bytes = 0
            for blob in blobs:
                if blob.name != gcs_prefix:  # Skip directory entries
                    total_bytes += blob.size

            total_mb = total_bytes / (1024 ** 2)
            logger.info(f"Total download size: {total_mb:.2f} MB")

            # Download all files with progress tracking
            downloaded_bytes = 0
            for i, blob in enumerate(blobs):
                # Get relative path within model directory
                relative_path = blob.name[len(gcs_prefix):]
                if not relative_path:  # Skip if it's just the directory
                    continue

                local_file = temp_dir / relative_path
                local_file.parent.mkdir(parents=True, exist_ok=True)

                logger.debug(f"Downloading {blob.name} -> {local_file}")
                
                # Use timeout-based download for all files
                if blob.size > 100 * 1024 * 1024:  # Large file > 100MB
                    # Use timeout for large files (5 minutes)
                    success = self._download_with_timeout(blob, local_file, timeout_seconds=300)
                    if not success:
                        raise RuntimeError(f"Download with timeout failed for {blob.name}")
                else:
                    # Use shorter timeout for small files (30 seconds)
                    success = self._download_with_timeout(blob, local_file, timeout_seconds=30)
                    if not success:
                        raise RuntimeError(f"Download with timeout failed for {blob.name}")
                
                # Update progress tracking
                downloaded_bytes += blob.size
                downloaded_mb = downloaded_bytes / (1024 ** 2)
                progress_pct = (downloaded_bytes / total_bytes * 100) if total_bytes > 0 else 0
                
                # Log progress every 10 files or for large files (>100MB)
                if (i + 1) % 10 == 0 or blob.size > 100 * 1024 * 1024:
                    logger.info(f"Download progress: {downloaded_mb:.1f}/{total_mb:.1f} MB "
                               f"({progress_pct:.1f}%) - File {i+1}/{len(blobs)}")

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
            model_id: Model identifier (e.g., 'opt-1.3b' or 'google/gemma-2-2b')

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"========== STARTING MODEL LOAD: {model_id} ==========")

        if model_id in self.loaded_models:
            logger.info(f"Model {model_id} already loaded in memory")
            return True

        try:
            # Try to download from GCS first
            local_model_path = None
            try:
                logger.info(f"Attempting to download {model_id} from GCS bucket: {self.gcs_bucket}")
                local_model_path = self._download_from_gcs(model_id)
                logger.info(f"Successfully downloaded from GCS to {local_model_path}")
            except Exception as gcs_error:
                logger.warning(f"GCS download failed for {model_id}: {gcs_error}")
                logger.info(f"Falling back to HuggingFace for {model_id}")

                # Fallback: Download directly from HuggingFace
                local_model_path = self.cache_dir / model_id.replace('/', '_')
                if not local_model_path.exists():
                    logger.info(f"Model not cached locally. Will download {model_id} from HuggingFace...")
                    # HuggingFace will download and cache automatically
                    # Use the original model_id (e.g., 'google/gemma-2-2b')
                    local_model_path = model_id
                    logger.info(f"HuggingFace will download to: {local_model_path}")
                else:
                    logger.info(f"Using cached model at {local_model_path}")

            logger.info(f"Loading model {model_id} from {local_model_path}")
            logger.info(f"Device: {self.device}")

            # Load the model with transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load tokenizer
            logger.info(f"Loading tokenizer for {model_id}...")
            tokenizer_start = time.time()
            tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))
            tokenizer_time = time.time() - tokenizer_start
            logger.info(f"Tokenizer loaded in {tokenizer_time:.2f}s")

            # Load model
            logger.info(f"Loading model for {model_id}...")
            logger.info(f"Using dtype: {'float16' if self.device.startswith('cuda') else 'float32'}")
            model_start = time.time()

            model = AutoModelForCausalLM.from_pretrained(
                str(local_model_path),
                torch_dtype=torch.float16 if self.device.startswith('cuda') else torch.float32,
                device_map=self.device if self.device.startswith('cuda') else None,
                low_cpu_mem_usage=True
            )

            model_time = time.time() - model_start
            logger.info(f"Model loaded in {model_time:.2f}s")

            # Move to device if CPU
            if not self.device.startswith('cuda'):
                logger.info(f"Moving model to device: {self.device}")
                device_move_start = time.time()
                model = model.to(self.device)
                device_move_time = time.time() - device_move_start
                logger.info(f"Model moved to device in {device_move_time:.2f}s")

            # Store in cache
            self.loaded_models[model_id] = {
                'model': model,
                'tokenizer': tokenizer,
                'path': local_model_path
            }

            # Log model info
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

            mem_usage = self.get_model_memory_usage(model_id)
            if mem_usage:
                logger.info(f"Model memory usage: {mem_usage / (1024**2):.1f} MB")

            logger.info(f"========== MODEL LOAD COMPLETE: {model_id} ==========")
            return True

        except Exception as e:
            logger.error(f"========== MODEL LOAD FAILED: {model_id} ==========")
            logger.error(f"Error: {e}", exc_info=True)
            return False

    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_id: Model identifier

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"========== STARTING MODEL UNLOAD: {model_id} ==========")

        if model_id not in self.loaded_models:
            logger.warning(f"Model {model_id} not loaded, cannot unload")
            return False

        try:
            # Get memory usage before unloading
            mem_before = self.get_model_memory_usage(model_id)
            if mem_before:
                logger.info(f"Memory usage before unload: {mem_before / (1024**2):.1f} MB")

            # Get model info
            model_info = self.loaded_models[model_id]

            # Delete model and tokenizer
            logger.info(f"Deleting model and tokenizer from memory...")
            del model_info['model']
            del model_info['tokenizer']

            # Remove from cache
            del self.loaded_models[model_id]

            # Clear CUDA cache if using GPU
            if self.device.startswith('cuda'):
                logger.info("Clearing CUDA cache...")
                torch.cuda.empty_cache()

            logger.info(f"========== MODEL UNLOAD COMPLETE: {model_id} ==========")
            return True

        except Exception as e:
            logger.error(f"========== MODEL UNLOAD FAILED: {model_id} ==========")
            logger.error(f"Error: {e}", exc_info=True)
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
