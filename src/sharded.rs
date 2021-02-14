use std::{
    collections::hash_map::RandomState,
    fmt,
    hash::{self, BuildHasher, Hash},
};

use hash::Hasher;

use crate::unsharded;

pub struct CacheHandle<'a, K, V>(unsharded::CacheHandle<'a, K, V>)
where
    K: Hash + Eq + Clone;

impl<'a, K, V> CacheHandle<'a, K, V>
where
    K: Hash + Eq + Clone,
{
    pub fn value(&self) -> &V {
        self.0.value()
    }
}

pub struct LruCache<K, V, S = RandomState> {
    shards: Vec<unsharded::LruCache<K, V>>,
    hasher: S,
}

impl<K, V> fmt::Debug for LruCache<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.shards.iter()).finish()
    }
}

unsafe impl<K, V> Send for LruCache<K, V> {}
unsafe impl<K, V> Sync for LruCache<K, V> {}

fn default_shards() -> usize {
    16
}

impl<K, V, S> LruCache<K, V, S>
where
    K: Send + Sync + Hash + Eq + Clone,
    V: Send + Sync,
    S: BuildHasher,
{
    pub fn with_shards_hasher(capacity: u64, shards: usize, hasher: S) -> Self {
        let shards = shards as u64;
        let cap_per_shard = (capacity + shards - 1) / shards; // Round up.
        Self {
            hasher,
            shards: (0..shards)
                .map(|_| unsharded::LruCache::new(cap_per_shard))
                .collect(),
        }
    }
}

impl<K, V> LruCache<K, V, RandomState>
where
    K: Send + Sync + Hash + Eq + Clone,
    V: Send + Sync,
{
    pub fn new(capacity: u64) -> Self {
        Self::with_shards_hasher(capacity, default_shards(), RandomState::default())
    }
}

impl<K, V, S> LruCache<K, V, S>
where
    K: Hash + Eq + Clone,
    S: BuildHasher,
{
    fn shard(&self, key: &K) -> &unsharded::LruCache<K, V> {
        let mut hasher = self.hasher.build_hasher();
        key.hash(&mut hasher);
        let h = hasher.finish() as usize;
        let shard_idx = h % self.shards.len();
        &self.shards[shard_idx]
    }

    /// Evict a value if it is present and unpinned.
    pub fn advice_evict(&self, key: K) {
        self.shard(&key).advice_evict(key)
    }

    /// Prune all unpinned values.
    pub fn prune(&self) {
        for s in &self.shards {
            s.prune();
        }
    }

    // Get the total charge of all values.
    pub fn total_charge(&self) -> u64 {
        self.shards.iter().map(|s| s.total_charge()).sum()
    }

    /// Get the cache handle for the key, return `None` if not present. The
    /// value is pinned in cache while the cache handle is live.
    pub fn get(&self, key: K) -> Option<CacheHandle<'_, K, V>> {
        self.shard(&key).get(key).map(|h| CacheHandle(h))
    }

    /// Error handling version of `get_or_init`. Value is not inserted if error
    /// occurs.
    pub fn get_or_try_init<E>(
        &self,
        key: K,
        charge: u64,
        init: impl FnOnce(&K) -> Result<V, E>,
    ) -> Result<CacheHandle<'_, K, V>, E> {
        self.shard(&key)
            .get_or_try_init(key, charge, init)
            .map(|h| CacheHandle(h))
    }

    /// Get the cache handle for the key, initialize the value if not present.
    /// The value is pinned in cache while the cache handle is live.
    ///
    /// Multiple threads calling `get_or_init` on the same key is fine. The
    /// value would be constructed exactly once.
    pub fn get_or_init(
        &self,
        key: K,
        charge: u64,
        init: impl FnOnce(&K) -> V,
    ) -> CacheHandle<'_, K, V> {
        CacheHandle(self.shard(&key).get_or_init(key, charge, init))
    }
}

mod compile_time_assertions {
    use super::*;

    #[allow(unreachable_code)]
    fn _assert_public_types_send_sync() {
        _assert_send_sync::<LruCache<u32, u32>>(unreachable!());
        _assert_send_sync::<CacheHandle<u32, u32>>(unreachable!());
    }

    fn _assert_send<S: Send>(_: &S) {}

    fn _assert_send_sync<S: Send + Sync>(_: &S) {}
}
