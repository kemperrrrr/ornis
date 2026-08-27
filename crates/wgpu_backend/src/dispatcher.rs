/// Where a piece of work should run: forced CPU, forced GPU, or decided by
/// workload size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTarget {
    /// Always execute on the CPU, regardless of workload size.
    Cpu,
    /// Always execute on the GPU, regardless of workload size.
    Gpu,
    /// Execute on the GPU once the element count reaches the contained
    /// threshold; below it the CPU wins (dispatch overhead dominates).
    Auto(usize),
}

impl Default for ExecutionTarget {
    fn default() -> Self {
        Self::Auto(10_000)
    }
}

impl ExecutionTarget {
    /// Picks the concrete [`Platform`] for a workload of `element_count`
    /// elements.
    pub fn resolve(self, element_count: usize) -> Platform {
        match self {
            Self::Cpu => Platform::Cpu,
            Self::Gpu => Platform::Gpu,
            Self::Auto(threshold) => {
                if element_count >= threshold {
                    Platform::Gpu
                } else {
                    Platform::Cpu
                }
            }
        }
    }
}

/// The concrete side a workload ends up on after [`ExecutionTarget`] is
/// resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Work executes on the CPU.
    Cpu,
    /// Work executes on the GPU via a compute dispatch.
    Gpu,
}

/// Per-dispatch knobs: which target policy to apply, the compute workgroup
/// size, and a debug label for GPU captures.
pub struct DispatchConfig {
    /// Policy deciding CPU vs GPU for this dispatch.
    pub target: ExecutionTarget,
    /// Workgroup size used to compute the dispatch grid on the GPU path.
    pub workgroup_size: u32,
    /// Label attached to the encoder and compute pass for debugging.
    pub label: String,
}

impl Default for DispatchConfig {
    fn default() -> Self {
        Self {
            target: ExecutionTarget::default(),
            workgroup_size: 64,
            label: String::new(),
        }
    }
}

/// Resolves `config.target` against `element_count` into a concrete
/// [`Platform`].
pub fn choose_platform(config: &DispatchConfig, element_count: usize) -> Platform {
    config.target.resolve(element_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_below_threshold() {
        let cfg = DispatchConfig {
            target: ExecutionTarget::Auto(10_000),
            ..Default::default()
        };
        assert_eq!(choose_platform(&cfg, 100), Platform::Cpu);
        assert_eq!(choose_platform(&cfg, 9_999), Platform::Cpu);
    }

    #[test]
    fn gpu_at_or_above_threshold() {
        let cfg = DispatchConfig {
            target: ExecutionTarget::Auto(10_000),
            ..Default::default()
        };
        assert_eq!(choose_platform(&cfg, 10_000), Platform::Gpu);
        assert_eq!(choose_platform(&cfg, 100_000), Platform::Gpu);
    }

    #[test]
    fn explicit_targets() {
        assert_eq!(ExecutionTarget::Cpu.resolve(1_000_000), Platform::Cpu);
        assert_eq!(ExecutionTarget::Gpu.resolve(1), Platform::Gpu);
    }
}
