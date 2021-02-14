// MIT License

// Copyright (c) 2020 Jianong Zhong

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! An implementation of a concurrent LRU cache. It is designed to hold heavyweight
//! resources, e.g. file descriptors, disk pages. The implementation is heavily
//! influenced by the [LRU cache in LevelDB].
//!
//! Currently there are two implementations, `unsharded` and `sharded`.
//!
//! - `unsharded` is a linked hashmap protected by a big lock.
//! - `sharded` shards `unsharded` by key, providing better performance under
//!   contention.
//!
//! ## Example
//!
//! ```rust,no_run
//! use concurrent_lru::sharded::LruCache;
//! use std::{fs, io};
//!
//! fn read(_f: &fs::File) -> io::Result<()> {
//!     // Maybe some positioned read...
//!     Ok(())
//! }
//!
//! fn main() -> io::Result<()> {
//!     let cache = LruCache::<String, fs::File>::new(10);
//!
//!     let foo_handle = cache.get_or_try_init("foo".to_string(), 1, |name| {
//!         fs::OpenOptions::new().read(true).open(name)
//!     })?;
//!     read(foo_handle.value())?;
//!     drop(foo_handle); // Unpin foo file.
//!
//!     // Foo is in the cache.
//!     assert!(cache.get("foo".to_string()).is_some());
//!
//!     // Evict foo manually.
//!     cache.prune();
//!     assert!(cache.get("foo".to_string()).is_none());
//!
//!     Ok(())
//! }
//! ```

pub mod sharded;
pub mod unsharded;

#[cfg(test)]
unsafe fn override_lifetime<'a, 'b, T>(t: &'a T) -> &'b T {
    std::mem::transmute(t)
}
