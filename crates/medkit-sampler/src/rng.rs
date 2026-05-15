use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy)]
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    pub(crate) fn next_usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_u64() as usize) % upper
        }
    }
}

pub(crate) fn seed_for(base: u64, case_id: &str, epoch: u64, worker: u64, index: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(base.to_le_bytes());
    hasher.update(case_id.as_bytes());
    hasher.update(epoch.to_le_bytes());
    hasher.update(worker.to_le_bytes());
    hasher.update(index.to_le_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[0..8].try_into().expect("digest slice length"))
}
