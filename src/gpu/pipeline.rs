//! GPU compute pipeline management and caching
//!
//! Provides utilities for creating and caching compute pipelines.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::ComputePipeline;

/// Cache key for pipelines
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PipelineKey {
    /// Shader entry point name
    pub entry_point: String,
    /// Shader source hash (for invalidation)
    pub shader_hash: u64,
}

/// Pipeline cache for reusing compiled compute pipelines
pub struct PipelineCache {
    cache: Arc<Mutex<HashMap<PipelineKey, Arc<ComputePipeline>>>>,
}

impl PipelineCache {
    /// Create a new pipeline cache
    pub fn new() -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get or create a pipeline
    pub fn get_or_create<F>(
        &self,
        key: PipelineKey,
        create_fn: F,
    ) -> Arc<ComputePipeline>
    where
        F: FnOnce() -> ComputePipeline,
    {
        let mut cache = self.cache.lock().unwrap();

        if let Some(pipeline) = cache.get(&key) {
            return pipeline.clone();
        }

        let pipeline = Arc::new(create_fn());
        cache.insert(key, pipeline.clone());
        pipeline
    }

    /// Clear the cache
    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
    }
}

impl Default for PipelineCache {
    fn default() -> Self {
        Self::new()
    }
}
