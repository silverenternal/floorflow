//! Runtime contracts shared by generated natural-language flow components.

/// Immutable description of one natural-language hierarchy level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HierarchySpec {
    pub id: &'static str,
    pub name: &'static str,
    pub role: &'static str,
    pub max_state_tokens: usize,
}

impl HierarchySpec {
    pub const fn new(
        id: &'static str,
        name: &'static str,
        role: &'static str,
        max_state_tokens: usize,
    ) -> Self {
        Self {
            id,
            name,
            role,
            max_state_tokens,
        }
    }
}

/// Compile-time declaration of a typed flow channel.
pub trait TypedChannel: Sized + Copy + 'static {
    const ALL: &'static [Self];

    fn as_str(self) -> &'static str;
    fn index(self) -> usize;

    fn bit(self) -> u64 {
        1_u64 << self.index()
    }

    fn from_name(name: &str) -> Option<Self>;
}

/// Semantic family of a benchmark proxy target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkTargetKind {
    Scalar,
    Token,
    Regression,
}

impl BenchmarkTargetKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Token => "token",
            Self::Regression => "regression",
        }
    }
}

/// Compile-time benchmark target with an explicit output family.
pub trait BenchmarkTarget: TypedChannel {
    fn kind(self) -> BenchmarkTargetKind;
}

/// Required validity facets for one structured output family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputContractSpec {
    pub name: &'static str,
    pub structural: bool,
    pub schema: bool,
    pub executable: bool,
}

impl OutputContractSpec {
    pub const fn new(name: &'static str, structural: bool, schema: bool, executable: bool) -> Self {
        Self {
            name,
            structural,
            schema,
            executable,
        }
    }
}

pub trait OutputContracts: Sized + Copy + 'static {
    const CONTRACTS: &'static [OutputContractSpec];

    fn contract(self) -> OutputContractSpec;
}

/// Compile-time declaration of an ordered hierarchy.
pub trait HierarchyLevels: Sized + Copy + 'static {
    const SPECS: &'static [HierarchySpec];

    fn spec(self) -> HierarchySpec;
    fn from_id(id: &str) -> Option<Self>;

    fn level_id(self) -> &'static str {
        self.spec().id
    }

    fn channel_mask(channels: &[u64]) -> u64 {
        channels.iter().fold(0, |mask, channel| mask | channel)
    }
}

#[cfg(test)]
mod tests {
    use super::{HierarchyLevels, HierarchySpec};

    #[derive(Clone, Copy)]
    struct Specs;

    impl HierarchyLevels for Specs {
        const SPECS: &'static [HierarchySpec] = &[];

        fn spec(self) -> HierarchySpec {
            HierarchySpec::new("", "", "", 0)
        }

        fn from_id(_id: &str) -> Option<Self> {
            None
        }
    }

    #[test]
    fn channel_masks_compose_without_losing_bits() {
        assert_eq!(Specs::channel_mask(&[1, 4, 8]), 13);
    }
}
