use std::{
    borrow::Borrow,
    cell::UnsafeCell,
    collections::{hash_map::Entry, HashMap},
    hash::Hash,
    ops::{Deref, DerefMut},
    ptr::null,
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex, MutexGuard,
    },
    todo,
};

use once_cell::sync::OnceCell;

enum Never {}

struct NodeState<K, V> {
    /// Count of corresponding `CacheHandle`.
    rc: u32,
    charge: u64,
    key: Option<K>,
    prev: *const Node<K, V>,
    next: *const Node<K, V>,
}

/// Represents an element in cache. The node must be in one of following states:
/// 1. In-use: The node is pinned (referenced) by at least one `CacheHandle`,
///    thus cannot be evicted. In such state, `rc` > 0, `prev` and `next` are
///    not valid.
/// 2. In-LRU-list: The node is not pinned (referenced) by any `CacheHandle`,
///    facing potential eviction. In such state, `rc` == 0, `prev` and `next`
///    fields form a circular linked list, in LRU order.
///
/// NOTE: NEVER obtain a mutable reference to nodes (except during
/// initialization). Otherwise, concurrent reader from `CacheHandle` would
/// violate the aliasing rules.
struct Node<K, V> {
    /// NOTE: Reading or writing state requires you holding the big lock.
    state: UnsafeCell<NodeState<K, V>>,

    /// NOTE: Reading or writing value should be done without the big lock held.
    /// `OnceCell` is used to coordinate possible concurrent initialization,
    /// i.e. `Lru::get_or_init` issued by multiple thread to the same key at the
    /// same time.
    value: OnceCell<V>,

    /// Is the value initialized?
    ///
    /// NOTE: An uninitialized (due to failure in `Lru::get_or_try_init`) value
    /// is NOT allowed to have a corresponding `CacheHandle`.
    value_init: AtomicBool,
}

pub struct CacheHandle<'a, K, V>
where
    K: Hash + Eq + Clone,
{
    lru: &'a Lru<K, V>,
    node: *const Node<K, V>,
}

unsafe impl<'a, K, V> Send for CacheHandle<'a, K, V> where K: Hash + Eq + Clone {}
unsafe impl<'a, K, V> Sync for CacheHandle<'a, K, V> where K: Hash + Eq + Clone {}

impl<'a, K, V> CacheHandle<'a, K, V>
where
    K: Hash + Eq + Clone,
{
    pub fn value(&self) -> &V {
        unsafe { (*self.node).value.get().unwrap() }
    }
}

impl<'a, K, V> Drop for CacheHandle<'a, K, V>
where
    K: Hash + Eq + Clone,
{
    fn drop(&mut self) {
        if let Ok(mut guard) = self.lru.state.lock() {
            Lru::node_unpin(guard.deref_mut(), self.node);
            Lru::maybe_evict_old(guard, false);
        }
    }
}

struct LruState<K, V> {
    map: HashMap<K, NodeBox<K, V>>,

    capacity: u64,

    /// Dummy head node of the circular linked list. Nodes are in LRU order.
    list_dummy: NodeBox<K, V>,

    /// Size of the circular linked list. Dummy head node is excluded.
    list_size: usize,

    /// Total charge of all values. Compared against `capacity`.
    total_charge: u64,
}

/// A concurrent LRU cache.
///
/// A value is pinned while its `CacheHandle` is live, and thus cannot be
/// evicted. Capacity specifies how many values could be cached, including
/// pinned and unpinned. Eviction happens when #value > capacity, or
/// `Lru::prune` is called.
///
/// Keys should be lightweight, while values could be heavyweight. That is,
/// creating (`Lru::get_or_init`) and dropping value could be time-consuming,
/// and the cache itself would not be blocked.
pub struct Lru<K, V> {
    state: Mutex<LruState<K, V>>,
}

unsafe impl<K, V> Send for Lru<K, V> {}
unsafe impl<K, V> Sync for Lru<K, V> {}

impl<K, V> Lru<K, V>
where
    K: Send + Sync + Hash + Eq + Clone,
    V: Send + Sync,
{
    pub fn new(capacity: u64) -> Self {
        Self::new_impl(capacity)
    }
}

struct NodeBox<K, V>(*const Node<K, V>);

impl<K, V> NodeBox<K, V> {
    fn new(node: Node<K, V>) -> Self {
        Self(Box::into_raw(Box::new(node)))
    }

    fn as_ptr(&self) -> *const Node<K, V> {
        self.0
    }
}

impl<K, V> Deref for NodeBox<K, V> {
    type Target = Node<K, V>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

impl<K, V> Drop for NodeBox<K, V> {
    fn drop(&mut self) {
        unsafe {
            Box::from_raw(self.0 as *mut Node<K, V>);
        }
    }
}

impl<K, V> Lru<K, V>
where
    K: Hash + Eq + Clone,
{
    unsafe fn link(cur: *const Node<K, V>, next: *const Node<K, V>) {
        (*(*cur).state.get()).next = next;
        (*(*next).state.get()).prev = cur;
    }

    /// Append node to list tail (newest).
    unsafe fn list_append(this: &mut LruState<K, V>, node: *const Node<K, V>) {
        this.list_size += 1;
        let dummy = this.list_dummy.deref();
        let prev = (*dummy.state.get()).prev;
        Self::link(node, dummy);
        Self::link(prev, node);
    }

    unsafe fn list_remove(this: &mut LruState<K, V>, node: *const Node<K, V>) {
        this.list_size -= 1;
        let node = &mut *(*node).state.get();
        Self::link(node.prev, node.next);
        node.prev = null();
        node.next = null();
    }

    fn maybe_evict_old(mut guard: MutexGuard<LruState<K, V>>, evict_all: bool) {
        let mut evict_nodes = vec![];
        let this = guard.deref_mut();

        unsafe {
            while (evict_all && this.list_size > 0) || (this.total_charge > this.capacity) {
                // Only obtain a shared reference to the dummy node.
                let oldest_ptr = (*this.list_dummy.state.get()).next;
                assert!(oldest_ptr != this.list_dummy.deref());

                let oldest = &mut *(*oldest_ptr).state.get();
                assert!(oldest.rc == 0);

                this.total_charge -= oldest.charge;

                // Remove node from hash map.
                let node = this.map.remove(oldest.key.as_ref().unwrap()).unwrap();
                evict_nodes.push(node);

                // Remove node from LRU list.
                Self::list_remove(this, oldest_ptr);
            }
        }

        // IMPORTANT: Drop user value without lock held.
        drop(guard);
        drop(evict_nodes);
    }

    fn node_unpin(this: &mut LruState<K, V>, node_ptr: *const Node<K, V>) {
        unsafe {
            let node = &mut *(*node_ptr).state.get();
            node.rc = node.rc.checked_sub(1).unwrap();
            if node.rc == 0 {
                Self::list_append(this, node_ptr);
            }
        }
    }

    fn node_pin(this: &mut LruState<K, V>, node_ptr: *const Node<K, V>) {
        unsafe {
            let node = &mut *(*node_ptr).state.get();
            node.rc += 1;
            if node.rc == 1 {
                Self::list_remove(this, node_ptr);
            }
        }
    }

    fn new_impl(capacity: u64) -> Self {
        let dummy = NodeBox::new(Node {
            state: UnsafeCell::new(NodeState {
                prev: null(),
                next: null(),
                key: None,
                rc: 0,
                charge: 0,
            }),
            value: OnceCell::new(),
            value_init: AtomicBool::new(false),
        });

        let ptr = dummy.as_ptr();
        unsafe {
            let dummy = &mut *dummy.deref().state.get();
            dummy.prev = ptr;
            dummy.next = ptr;
        }

        Self {
            state: Mutex::new(LruState {
                map: Default::default(),
                capacity,
                total_charge: 0,
                list_size: 0,
                list_dummy: dummy,
            }),
        }
    }

    /// Evict a value if it is present and unpinned.
    pub fn advice_evict(&self, key: K) {
        let mut guard = self.state.lock().unwrap();
        let this = guard.deref_mut();
        match this.map.entry(key) {
            Entry::Occupied(ent) => unsafe {
                let node_ptr = ent.get().as_ptr();
                let node = &mut *(*node_ptr).state.get();
                // In LRU list?
                if node.rc == 0 {
                    this.total_charge -= node.charge;
                    let evicted = ent.remove();
                    Self::list_remove(this, node_ptr);

                    // IMPORTANT: Drop user value without lock held.
                    drop(guard);
                    drop(evicted);
                }
            },
            _ => {}
        };
    }

    /// Prune all unpinned values.
    pub fn prune(&self) {
        let guard = self.state.lock().unwrap();
        Self::maybe_evict_old(guard, true);
    }

    // Get the total charge of all values.
    pub fn total_charge(&self) -> u64 {
        let mut guard = self.state.lock().unwrap();
        let this = guard.deref_mut();
        this.total_charge
    }

    /// Get the cache handle for the key, return `None` if not present. The
    /// value is pinned in cache while the cache handle is live.
    pub fn get(&self, key: K) -> Option<CacheHandle<'_, K, V>> {
        let mut guard = self.state.lock().unwrap();
        let this = guard.deref_mut();
        let node_ptr = this.map.get(&key)?.as_ptr();

        unsafe {
            let node = &*node_ptr;
            if !node.value_init.load(Ordering::Acquire) {
                return None;
            }
        }

        Self::node_pin(this, node_ptr);

        Some(CacheHandle {
            lru: self,
            node: node_ptr,
        })
    }

    /// Error handling version of `get_or_init`. Value is not inserted if error
    /// occurs.
    pub fn get_or_try_init<E>(
        &self,
        key: K,
        charge: u64,
        init: impl FnOnce(&K) -> Result<V, E>,
    ) -> Result<CacheHandle<'_, K, V>, E> {
        assert!(charge > 0, "charge must > 0");
        let mut guard = self.state.lock().unwrap();
        let this = guard.deref_mut();
        let node_ptr = match this.map.entry(key.clone()) {
            Entry::Occupied(ent) => {
                let node_ptr = ent.get().as_ptr();
                Self::node_pin(this, node_ptr);
                node_ptr
            }
            Entry::Vacant(ent) => {
                this.total_charge += charge;
                let node = NodeBox::new(Node {
                    value: OnceCell::new(),
                    value_init: AtomicBool::new(false),
                    state: UnsafeCell::new(NodeState {
                        charge,
                        prev: null(),
                        key: Some(key.clone()),
                        next: null(),
                        rc: 1,
                    }),
                });

                let node_ptr = node.as_ptr();
                ent.insert(node);
                node_ptr
            }
        };

        Self::maybe_evict_old(guard, false);

        // IMPORTANT: Call user-provided init function without lock held.
        let node = unsafe { &*node_ptr };
        match node.value.get_or_try_init(|| init(&key)) {
            Ok(_) => {
                node.value_init.store(true, Ordering::Release);
                Ok(CacheHandle {
                    lru: self,
                    node: node_ptr,
                })
            }
            Err(e) => {
                let mut guard = self.state.lock().unwrap();
                Self::node_unpin(guard.deref_mut(), node_ptr);

                unsafe {
                    let node = &mut *node.state.get();
                    node.rc = node.rc.checked_sub(1).unwrap();
                    if node.rc == 0 {
                        this.total_charge -= node.charge;
                        let evicted = ent.remove();
                    }
                }

                Err(e)
            }
        }
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
        match self.get_or_try_init(key, charge, |k| Ok::<_, Never>(init(k))) {
            Ok(x) => x,
            _ => unreachable!(),
        }
    }
}

mod compile_time_assertions {
    use super::*;

    #[allow(unreachable_code)]
    fn _assert_public_types_send_sync() {
        _assert_send_sync::<Lru<u32, u32>>(unreachable!());
        _assert_send_sync::<CacheHandle<u32, u32>>(unreachable!());
    }

    fn _assert_send<S: Send>(_: &S) {}

    fn _assert_send_sync<S: Send + Sync>(_: &S) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{distributions::Uniform, prelude::*};
    use std::{
        mem,
        sync::{
            atomic::{AtomicU64, Ordering},
            Arc,
        },
        thread,
    };

    #[test]
    fn simple_get_or_init() {
        let cache = Lru::<u32, u32>::new(10);
        assert_eq!(cache.get_or_init(1, 1, |_| 1).value(), &1);
        assert_eq!(cache.get_or_init(2, 1, |_| 2).value(), &2);
        assert_eq!(cache.get_or_init(1, 1, |_| 3).value(), &1);
    }

    #[test]
    fn reinsert_after_eviction() {
        let cache = Lru::<u32, u32>::new(5);
        assert_eq!(cache.get_or_init(1, 3, |_| 1).value(), &1);
        assert_eq!(cache.get_or_init(2, 3, |_| 2).value(), &2);
        assert_eq!(cache.get_or_init(1, 3, |_| 3).value(), &3);
    }

    #[test]
    fn advice_evict() {
        let cache = Lru::<u32, u32>::new(5);
        let handle = cache.get_or_init(1, 3, |_| 1);
        cache.advice_evict(1); // Shoule be rejected.
        assert!(cache.get(1).is_some());
        drop(handle);
        cache.advice_evict(1); // Shoule be accepted.
        assert!(cache.get(1).is_none());
    }

    struct DropRecorded {
        evicted: Arc<Mutex<Vec<u32>>>,
        id: u32,
    }

    impl Drop for DropRecorded {
        fn drop(&mut self) {
            if !std::thread::panicking() {
                self.evicted.lock().unwrap().push(self.id);
            }
        }
    }

    #[test]
    fn evicted_dropped() {
        let evicted = Arc::new(Mutex::new(Vec::new()));

        let cache = Lru::<u32, DropRecorded>::new(2);
        let insert_new = |id| {
            cache.get_or_init(id, 1, |&id| DropRecorded {
                id,
                evicted: evicted.clone(),
            })
        };

        insert_new(1);
        insert_new(2);
        insert_new(3);

        assert_eq!(*evicted.lock().unwrap(), vec![1]);
    }

    #[test]
    fn pin_unpin_eviction() {
        let evicted = Arc::new(Mutex::new(Vec::new()));

        let cache = Lru::<u32, DropRecorded>::new(2);
        let insert_new = |id| {
            cache.get_or_init(id, 1, |&id| DropRecorded {
                id,
                evicted: evicted.clone(),
            })
        };

        insert_new(1);
        insert_new(2);
        insert_new(3);
        assert_eq!(*evicted.lock().unwrap(), vec![1]);

        // Pin the value.
        let h4 = insert_new(4);
        assert_eq!(*evicted.lock().unwrap(), vec![1, 2]);

        insert_new(5);
        insert_new(6);
        assert_eq!(*evicted.lock().unwrap(), vec![1, 2, 3, 5]);

        // Unpin.
        drop(h4);
        assert_eq!(*evicted.lock().unwrap(), vec![1, 2, 3, 5]);

        insert_new(7);
        insert_new(8);
        assert_eq!(*evicted.lock().unwrap(), vec![1, 2, 3, 5, 6, 4]);
    }

    unsafe fn override_lifetime<'a, 'b, T>(t: &'a T) -> &'b T {
        mem::transmute(t)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn stress() {
        struct IncCounterOnDrop<'a> {
            charge: u64,
            counter: &'a AtomicU64,
        }

        impl<'a> Drop for IncCounterOnDrop<'a> {
            fn drop(&mut self) {
                self.counter.fetch_add(self.charge, Ordering::Relaxed);
            }
        }

        let capacity = 100;
        let threads = 1;
        let per_thread_count = 10000;
        let yield_interval = 1000;

        let init_charge = AtomicU64::new(0);
        let drop_charge = AtomicU64::new(0);
        let lru = Lru::new(capacity);

        let mut handles = vec![];
        for _ in 0..threads {
            handles.push(thread::spawn({
                let lru = unsafe { override_lifetime(&lru) };
                let init_counter = unsafe { override_lifetime(&init_charge) };
                let drop_counter = unsafe { override_lifetime(&drop_charge) };
                move || {
                    let mut rng = StdRng::from_entropy();
                    for i in 0..per_thread_count {
                        let charge = rng.sample(Uniform::new(1, 5));
                        lru.get_or_init(i, charge, |_| {
                            init_counter.fetch_add(charge, Ordering::Relaxed);
                            IncCounterOnDrop {
                                charge,
                                counter: &drop_counter,
                            }
                        });

                        if i % yield_interval == 0 {
                            thread::yield_now();
                        }
                    }
                }
            }));
        }

        // Join threads.
        for h in handles {
            h.join().unwrap();
        }

        assert!(init_charge.load(Ordering::Relaxed) >= per_thread_count);
        assert!(lru.total_charge() <= capacity);
        assert_eq!(
            init_charge.load(Ordering::Relaxed),
            lru.total_charge() + drop_charge.load(Ordering::Relaxed)
        );

        lru.prune();
        assert_eq!(
            init_charge.load(Ordering::Relaxed),
            drop_charge.load(Ordering::Relaxed)
        );
    }
}
