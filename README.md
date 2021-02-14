# Concurrent LRU

[![crates.io Badge]][crates.io package]
[![docs.rs Badge]][docs.rs documentation]
[![License Badge]][license]

An implementation of a concurrent LRU cache. It is designed to hold heavyweight
resources, e.g. file descriptors, disk pages. The implementation is heavily
influenced by the [LRU cache in LevelDB].

Currently there are two implementations, `unsharded` and `sharded`.

- `unsharded` is a linked hashmap protected by a big lock.
- `sharded` shards `unsharded` by key, providing better performance under
  contention.

## Example

```rust,no_run
use concurrent_lru::sharded::LruCache;
use std::{fs, io};

fn read(_f: &fs::File) -> io::Result<()> {
    // Maybe some positioned read...
    Ok(())
}

fn main() -> io::Result<()> {
    let cache = LruCache::<String, fs::File>::new(10);

    let foo_handle = cache.get_or_try_init("foo".to_string(), 1, |name| {
        fs::OpenOptions::new().read(true).open(name)
    })?;
    read(foo_handle.value())?;
    drop(foo_handle); // Unpin foo file.

    // Foo is in the cache.
    assert!(cache.get("foo".to_string()).is_some());

    // Evict foo manually.
    cache.prune();
    assert!(cache.get("foo".to_string()).is_none());

    Ok(())
}
```

## Contribution

Contributions are welcome! Please [fork the
library](https://github.com/ngkv/concurrent_lru/fork), push changes to your
fork, and send a [pull
request](https://help.github.com/articles/creating-a-pull-request-from-a-fork/).
All contributions are shared under an MIT license unless explicitly stated
otherwise in the pull request.

## Performance

TODO

[license badge]: https://img.shields.io/badge/license-MIT-blue.svg
[crates.io badge]: https://img.shields.io/crates/v/concurrent_lru.svg
[crates.io package]: https://crates.io/crates/concurrent_lru
[documentation]: https://docs.rs/concurrent_lru
[docs.rs badge]: https://docs.rs/concurrent_lru/badge.svg
[docs.rs documentation]: https://docs.rs/concurrent_lru/
[license badge]: https://img.shields.io/badge/license-MIT-blue.svg
[license]: https://raw.githubusercontent.com/ngkv/concurrent_lru/master/LICENSE
[LRU cache in LevelDB]: https://github.com/google/leveldb/blob/master/util/cache.cc
