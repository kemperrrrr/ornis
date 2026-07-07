#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTarget {
    Cpu,
    Gpu,
    Auto(usize),
}

impl Default for ExecutionTarget {
    fn default() -> Self {
        Self::Auto(10_000)
    }
}

impl ExecutionTarget {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Cpu,
    Gpu,
}

pub struct DispatchConfig {
    pub target: ExecutionTarget,
    pub workgroup_size: u32,
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
