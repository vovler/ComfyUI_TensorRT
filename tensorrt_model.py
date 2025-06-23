"""
TensorRT Model Management

Centralized TensorRT runtime and builder management following best practices:
- One Runtime for the entire application
- One Builder for the entire script (reused for builds)
- Individual engines and contexts per model
- Individual NetworkDefinition, OnnxParser, and BuilderConfig per model being built
"""

import os
import threading
import tensorrt as trt
from typing import Optional, Dict, Any
from .utils.tensorrt_error_recorder import TrTErrorRecorder


class TensorRTManager:
    """
    Singleton manager for TensorRT runtime and builder instances.
    Ensures proper resource management and follows TensorRT best practices.
    """
    
    _instance: Optional['TensorRTManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'TensorRTManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not getattr(self, '_initialized', False):
            self._initialize()
            self._initialized = True
    
    def _initialize(self):
        """Initialize TensorRT components"""
        print("Initializing TensorRT Manager...")
        
        # Initialize TensorRT plugins
        trt.init_libnvinfer_plugins(None, "")
        
        # Create global logger with INFO level
        self._logger = trt.Logger(trt.Logger.INFO)
        
        # Create global runtime with error recorder
        self._runtime = trt.Runtime(self._logger)
        self._runtime.error_recorder = TrTErrorRecorder()
        
        # Create global builder (reused for all conversions)
        self._builder = trt.Builder(self._logger)
        self._builder.error_recorder = TrTErrorRecorder()
        
        # Track active engines and contexts for cleanup
        self._active_engines: Dict[str, trt.ICudaEngine] = {}
        self._active_contexts: Dict[str, trt.IExecutionContext] = {}
        
        print("TensorRT Manager initialized successfully")
    
    @property
    def logger(self) -> trt.Logger:
        """Get the global TensorRT logger"""
        return self._logger
    
    @property
    def runtime(self) -> trt.Runtime:
        """Get the global TensorRT runtime"""
        return self._runtime
    
    @property
    def builder(self) -> trt.Builder:
        """Get the global TensorRT builder (reused for all builds)"""
        return self._builder
    
    def create_network(self, explicit_batch: bool = True) -> trt.INetworkDefinition:
        """
        Create a new network definition for model building.
        Should be used once per model being built, then discarded.
        """
        flags = 0
        if explicit_batch:
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        
        network = self._builder.create_network(flags)
        print(f"Created new NetworkDefinition with flags: {flags}")
        return network
    
    def create_onnx_parser(self, network: trt.INetworkDefinition) -> trt.OnnxParser:
        """
        Create a new ONNX parser for model building.
        Should be used once per model being built, then discarded.
        """
        parser = trt.OnnxParser(network, self._logger)
        print("Created new OnnxParser")
        return parser
    
    def create_builder_config(self) -> trt.IBuilderConfig:
        """
        Create a new builder config for model building.
        Should be used once per model being built, then discarded.
        """
        config = self._builder.create_builder_config()
        print("Created new BuilderConfig")
        return config
    
    def deserialize_engine(self, engine_path: str, engine_id: Optional[str] = None) -> trt.ICudaEngine:
        """
        Deserialize a TensorRT engine from file.
        Engines are cached and tracked for proper cleanup.
        
        Args:
            engine_path: Path to the .engine file
            engine_id: Optional identifier for tracking (defaults to engine_path)
            
        Returns:
            Deserialized TensorRT engine
        """
        if engine_id is None:
            engine_id = engine_path
            
        # Check if engine is already loaded
        if engine_id in self._active_engines:
            print(f"Reusing existing engine: {engine_id}")
            return self._active_engines[engine_id]
        
        print(f"Deserializing TensorRT engine from: {engine_path}")
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")
        
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        engine = self._runtime.deserialize_cuda_engine(engine_data)
        
        # Check for deserialization errors
        if self._runtime.error_recorder.num_errors() > 0:
            error_msgs = []
            for i in range(self._runtime.error_recorder.num_errors()):
                error_msgs.append(self._runtime.error_recorder.get_error_desc(i))
            self._runtime.error_recorder.clear()
            raise RuntimeError(f"Engine deserialization errors: {'; '.join(error_msgs)}")
        
        if engine is None:
            raise RuntimeError(f"Failed to deserialize engine from: {engine_path}")
        
        # Cache the engine
        self._active_engines[engine_id] = engine
        print(f"Successfully deserialized and cached engine: {engine_id}")
        
        return engine
    
    def create_execution_context(self, engine: trt.ICudaEngine, context_id: Optional[str] = None) -> trt.IExecutionContext:
        """
        Create an execution context for an engine.
        Contexts are tracked for proper cleanup.
        
        Args:
            engine: The TensorRT engine
            context_id: Optional identifier for tracking
            
        Returns:
            TensorRT execution context
        """
        if context_id is None:
            context_id = f"context_{id(engine)}"
        
        # Check if context already exists
        if context_id in self._active_contexts:
            print(f"Reusing existing context: {context_id}")
            return self._active_contexts[context_id]
        
        print(f"Creating execution context: {context_id}")
        context = engine.create_execution_context()
        
        # Check for context creation errors
        if self._runtime.error_recorder.num_errors() > 0:
            error_msgs = []
            for i in range(self._runtime.error_recorder.num_errors()):
                error_msgs.append(self._runtime.error_recorder.get_error_desc(i))
            self._runtime.error_recorder.clear()
            print(f"Warning: Context creation errors: {'; '.join(error_msgs)}")
        
        if context is None:
            raise RuntimeError("Failed to create execution context")
        
        # Cache the context
        self._active_contexts[context_id] = context
        print(f"Successfully created and cached context: {context_id}")
        
        return context
    
    def build_serialized_network(self, network: trt.INetworkDefinition, config: trt.IBuilderConfig) -> bytes:
        """
        Build a serialized TensorRT engine from network and config.
        Uses the global builder instance.
        
        Args:
            network: The network definition (will be consumed)
            config: The builder config (will be consumed)
            
        Returns:
            Serialized engine bytes
        """
        print("Building serialized TensorRT network...")
        serialized_engine = self._builder.build_serialized_network(network, config)
        
        # Check for build errors
        if self._builder.error_recorder.num_errors() > 0:
            error_msgs = []
            for i in range(self._builder.error_recorder.num_errors()):
                error_msgs.append(self._builder.error_recorder.get_error_desc(i))
            self._builder.error_recorder.clear()
            raise RuntimeError(f"Engine build errors: {'; '.join(error_msgs)}")
        
        if serialized_engine is None:
            raise RuntimeError("Failed to build serialized network")
        
        print(f"Successfully built serialized network ({len(serialized_engine)} bytes)")
        return serialized_engine
    
    def unload_engine(self, engine_id: str):
        """
        Unload and cleanup a specific engine and its contexts.
        
        Args:
            engine_id: The engine identifier to unload
        """
        # Remove associated contexts first
        contexts_to_remove = [cid for cid in self._active_contexts.keys() if cid.startswith(f"context_{id(self._active_engines.get(engine_id))}")]
        for context_id in contexts_to_remove:
            context = self._active_contexts.pop(context_id, None)
            if context is not None:
                del context
                print(f"Cleaned up context: {context_id}")
        
        # Remove the engine
        engine = self._active_engines.pop(engine_id, None)
        if engine is not None:
            del engine
            print(f"Cleaned up engine: {engine_id}")
    
    def unload_context(self, context_id: str):
        """
        Unload and cleanup a specific execution context.
        
        Args:
            context_id: The context identifier to unload
        """
        context = self._active_contexts.pop(context_id, None)
        if context is not None:
            del context
            print(f"Cleaned up context: {context_id}")
    
    def get_engine_info(self, engine: trt.ICudaEngine) -> Dict[str, Any]:
        """
        Get detailed information about a TensorRT engine.
        
        Args:
            engine: The TensorRT engine
            
        Returns:
            Dictionary containing engine information
        """
        info = {
            "num_io_tensors": engine.num_io_tensors,
            "num_optimization_profiles": engine.num_optimization_profiles,
            "tensors": []
        }
        
        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            tensor_shape = engine.get_tensor_shape(tensor_name)
            tensor_dtype = engine.get_tensor_dtype(tensor_name)
            is_input = engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
            
            tensor_info = {
                "name": tensor_name,
                "shape": list(tensor_shape),
                "dtype": str(tensor_dtype),
                "is_input": is_input
            }
            
            # Add profile information for input tensors
            if is_input:
                profiles = []
                for profile_idx in range(engine.num_optimization_profiles):
                    try:
                        min_shape, opt_shape, max_shape = engine.get_tensor_profile_shape(tensor_name, profile_idx)
                        profiles.append({
                            "min": list(min_shape),
                            "opt": list(opt_shape),
                            "max": list(max_shape)
                        })
                    except Exception as e:
                        profiles.append({"error": str(e)})
                tensor_info["profiles"] = profiles
            
            info["tensors"].append(tensor_info)
        
        return info
    
    def cleanup_all(self):
        """
        Cleanup all active engines and contexts.
        Should be called when shutting down the application.
        """
        print("Cleaning up all TensorRT resources...")
        
        # Cleanup all contexts
        for context_id, context in list(self._active_contexts.items()):
            del context
            print(f"Cleaned up context: {context_id}")
        self._active_contexts.clear()
        
        # Cleanup all engines
        for engine_id, engine in list(self._active_engines.items()):
            del engine
            print(f"Cleaned up engine: {engine_id}")
        self._active_engines.clear()
        
        print("TensorRT cleanup complete")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        if hasattr(self, '_initialized') and self._initialized:
            self.cleanup_all()


# Global instance accessor
def get_tensorrt_manager() -> TensorRTManager:
    """Get the global TensorRT manager instance"""
    return TensorRTManager()


# Convenience functions for common operations
def get_runtime() -> trt.Runtime:
    """Get the global TensorRT runtime"""
    return get_tensorrt_manager().runtime


def get_builder() -> trt.Builder:
    """Get the global TensorRT builder"""
    return get_tensorrt_manager().builder


def get_logger() -> trt.Logger:
    """Get the global TensorRT logger"""
    return get_tensorrt_manager().logger


def deserialize_engine(engine_path: str, engine_id: Optional[str] = None) -> trt.ICudaEngine:
    """Convenience function to deserialize an engine"""
    return get_tensorrt_manager().deserialize_engine(engine_path, engine_id)


def create_execution_context(engine: trt.ICudaEngine, context_id: Optional[str] = None) -> trt.IExecutionContext:
    """Convenience function to create an execution context"""
    return get_tensorrt_manager().create_execution_context(engine, context_id) 