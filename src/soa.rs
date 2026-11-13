//! `JumpArraySoA`：与 [`crate::JumpArray`] 同算法，但键与值分数组存放（SoA）。
//!
//! 动机：块内短扫与块内二分只比较键。AoS 的 `(K, E)` 每条 16 B（bench 里
//! `(i64,u32)`），一次 cache line 只放 4 条；SoA 的键数组 8 B/条，一次放 8 条，
//! 扫描/二分的缓存行数减半。聚簇数据（块内元素多、频繁走块内二分）最可能受益。
//!
//! 代价：插入/删除要同时搬两个数组（总字节数不变）；区间遍历要在两个切片上 zip。
//! 这里只实现读路径所需的最小面（无游标），用于与 AoS 版本做 A/B。

use crate::Key;
use std::cmp::Ordering;

pub struct JumpArraySoA<K: Key, E> {
    keys: Vec<K>,
    vals: Vec<E>,
    /// 块起点 → 第一个 `key >= block_start` 的下标。
    blocks: Vec<u32>,
    base: f64,
    width: f64,
    dirty: bool,
    linear_max: usize,
    binary_below: usize,
}

const DEFAULT_LINEAR_MAX: usize = 8;
const DEFAULT_BINARY_BELOW: usize = 2048;

impl<K: Key, E> Default for JumpArraySoA<K, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, E> JumpArraySoA<K, E> {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            vals: Vec::new(),
            blocks: Vec::new(),
            base: 0.0,
            width: 1.0,
            dirty: true,
            linear_max: DEFAULT_LINEAR_MAX,
            binary_below: DEFAULT_BINARY_BELOW,
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn stats(&self) -> (usize, usize, f64) {
        (self.blocks.len(), self.keys.len(), self.width)
    }

    /// 重建块表（批量写完之后调一次）。
    pub fn rebuild(&mut self) {
        self.blocks.clear();
        if self.keys.is_empty() {
            self.base = 0.0;
            self.width = 1.0;
            self.dirty = false;
            return;
        }
        let first = self.keys[0].to_f64();
        let last = self.keys[self.keys.len() - 1].to_f64();
        let span = (last - first).max(0.0);
        let width = pow2_ceil(span / self.keys.len().max(1) as f64);
        self.base = first;
        self.width = width;
        let nblocks = ((span / width).ceil() as usize).saturating_add(1).max(1);
        self.blocks.reserve(nblocks);
        let mut i = 0usize;
        for b in 0..nblocks {
            let start = first + b as f64 * width;
            while i < self.keys.len() && self.keys[i].to_f64() < start {
                i += 1;
            }
            self.blocks.push(i as u32);
        }
        self.dirty = false;
    }

    #[inline]
    fn use_index(&self) -> bool {
        !self.dirty && !self.blocks.is_empty() && self.keys.len() > self.binary_below
    }

    #[inline]
    fn block_of(&self, k: &K) -> usize {
        if self.blocks.is_empty() {
            return 0;
        }
        let x = (k.to_f64() - self.base) / self.width;
        if !x.is_finite() || x <= 0.0 {
            0
        } else {
            let b = x.floor();
            if b >= self.blocks.len() as f64 {
                self.blocks.len() - 1
            } else {
                b as usize
            }
        }
    }

    #[inline]
    fn block_span(&self, b: usize) -> (usize, usize) {
        let start = self.blocks[b] as usize;
        let end = if b + 1 < self.blocks.len() {
            self.blocks[b + 1] as usize
        } else {
            self.keys.len()
        };
        (start, end)
    }

    fn lower_bound(&self, k: &K) -> usize {
        self.keys
            .partition_point(|x| x.cmp_key(k) == Ordering::Less)
    }

    fn upper_bound(&self, k: &K) -> usize {
        self.keys
            .partition_point(|x| x.cmp_key(k) != Ordering::Greater)
    }

    fn lower_bound_fast(&self, k: &K) -> usize {
        if !self.use_index() {
            return self.lower_bound(k);
        }
        let (start, end) = self.block_span(self.block_of(k));
        if end - start <= self.linear_max {
            let mut i = start;
            while i < end && self.keys[i].cmp_key(k) == Ordering::Less {
                i += 1;
            }
            i
        } else {
            start
                + self.keys[start..end].partition_point(|x| x.cmp_key(k) == Ordering::Less)
        }
    }

    fn upper_bound_fast(&self, k: &K) -> usize {
        if !self.use_index() {
            return self.upper_bound(k);
        }
        let (start, end) = self.block_span(self.block_of(k));
        if end - start <= self.linear_max {
            let mut i = start;
            while i < end && self.keys[i].cmp_key(k) != Ordering::Greater {
                i += 1;
            }
            i
        } else {
            start
                + self.keys[start..end].partition_point(|x| x.cmp_key(k) != Ordering::Greater)
        }
    }

    pub fn insert(&mut self, key: K, value: E) -> Result<(), E> {
        if let Some(last) = self.keys.last()
            && key.cmp_key(last) == Ordering::Greater
        {
            self.keys.push(key);
            self.vals.push(value);
            self.dirty = true;
            return Ok(());
        }
        let pos = self.lower_bound(&key);
        if pos < self.keys.len() && self.keys[pos].cmp_key(&key) == Ordering::Equal {
            return Err(value);
        }
        self.keys.insert(pos, key);
        self.vals.insert(pos, value);
        self.dirty = true;
        Ok(())
    }

    pub fn remove(&mut self, key: &K) -> Option<E> {
        if let Some(last) = self.keys.last()
            && last.cmp_key(key) == Ordering::Equal
        {
            self.keys.pop();
            self.dirty = true;
            return self.vals.pop();
        }
        let pos = self.lower_bound(key);
        if pos >= self.keys.len() || self.keys[pos].cmp_key(key) != Ordering::Equal {
            return None;
        }
        self.keys.remove(pos);
        self.dirty = true;
        Some(self.vals.remove(pos))
    }

    pub fn get(&self, key: &K) -> Option<&E> {
        self.find_le(key)
            .filter(|(k, _)| k.cmp_key(key) == Ordering::Equal)
            .map(|(_, v)| v)
    }

    /// 最后一个 `key <= k`。
    pub fn find_le(&self, k: &K) -> Option<(&K, &E)> {
        let i = self.upper_bound_fast(k);
        if i == 0 {
            None
        } else {
            Some((&self.keys[i - 1], &self.vals[i - 1]))
        }
    }

    /// 第一个 `key >= k`。
    pub fn find_ge(&self, k: &K) -> Option<(&K, &E)> {
        let i = self.lower_bound_fast(k);
        self.keys.get(i).map(|key| (key, &self.vals[i]))
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &E)> {
        self.keys.iter().zip(self.vals.iter())
    }

    /// `[lo, hi]` 区间（含两端、升序、不克隆）。
    pub fn range(&self, lo: K, hi: K) -> SoaRange<'_, K, E> {
        let none = || SoaRange {
            keys: &[],
            vals: &[],
        };
        if self.is_empty() || lo.cmp_key(&hi) == Ordering::Greater {
            return none();
        }
        let from = self.lower_bound_fast(&lo);
        let to = self.upper_bound_fast(&hi).max(from);
        SoaRange {
            keys: &self.keys[from..to],
            vals: &self.vals[from..to],
        }
    }
}

fn pow2_ceil(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return 1.0;
    }
    let e = x.log2().ceil().clamp(-60.0, 40.0);
    2f64.powf(e)
}

/// [`JumpArraySoA::range`] 的迭代器：两个切片 zip。
pub struct SoaRange<'a, K: Key, E> {
    keys: &'a [K],
    vals: &'a [E],
}

impl<'a, K: Key, E> Iterator for SoaRange<'a, K, E> {
    type Item = (&'a K, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        let (key, keys) = self.keys.split_first()?;
        let (val, vals) = self.vals.split_first()?;
        self.keys = keys;
        self.vals = vals;
        Some((key, val))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.keys.len(), Some(self.keys.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 与"排序 Vec + 二分"逐操作对照（含 rebuild 时机）。
    #[test]
    fn soa_matches_sorted_vec_semantics() {
        let mut rng: u64 = 0x5eed_face_1234_5678;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut a = JumpArraySoA::<i64, i64>::new();
        let mut model: Vec<(i64, i64)> = Vec::new();
        for step in 0..4000 {
            let k = (next() % 30_000) as i64;
            if next() % 3 != 0 {
                let r = a.insert(k, k * 2);
                let dup = model.binary_search_by_key(&k, |x| x.0).is_ok();
                assert_eq!(r.is_err(), dup, "step {step} insert {k}");
                if !dup {
                    let pos = model.partition_point(|x| x.0 < k);
                    model.insert(pos, (k, k * 2));
                }
            } else {
                let r = a.remove(&k);
                match model.binary_search_by_key(&k, |x| x.0) {
                    Ok(pos) => {
                        model.remove(pos);
                        assert_eq!(r, Some(k * 2), "step {step} remove {k}");
                    }
                    Err(_) => assert!(r.is_none(), "step {step} remove {k}"),
                }
            }
            if step % 89 == 0 {
                a.rebuild();
            }
            let q = (next() % 30_000) as i64;
            let le = model.partition_point(|x| x.0 <= q);
            assert_eq!(
                a.find_le(&q).map(|(k, _)| *k),
                (le > 0).then(|| model[le - 1].0),
                "step {step} find_le {q}"
            );
            let ge = model.partition_point(|x| x.0 < q);
            assert_eq!(
                a.find_ge(&q).map(|(k, _)| *k),
                model.get(ge).map(|x| x.0),
                "step {step} find_ge {q}"
            );
            let (lo, hi) = if q < 15_000 { (q, q + 200) } else { (q - 200, q) };
            let want: Vec<i64> = model
                .iter()
                .filter(|(k, _)| *k >= lo && *k <= hi)
                .map(|(k, _)| *k)
                .collect();
            let got: Vec<i64> = a.range(lo, hi).map(|(k, _)| *k).collect();
            assert_eq!(got, want, "step {step} range [{lo},{hi}]");
        }
        a.rebuild();
        assert_eq!(
            a.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            model.iter().map(|x| x.0).collect::<Vec<_>>()
        );
    }
}
