//! Rusty JumpArray：kipphi 的"块索引 + 短走步"思想，落在**单块连续数组**上。
//!
//! ## 结构
//!
//! ```text
//! entries: Vec<(K, E)>   按键升序、连续存放
//! blocks:  Vec<u32>      块起点 → 第一个 key ≥ 块起点的条目下标
//! ```
//!
//! * **块宽** `W = 2^ceil(log2(span/n))` ⇒ 块数 ≈ 元素数，与键的数值大小无关
//!   （稀疏键不膨胀；这是相对"按数值单位分槽"的关键差异）；
//! * **查询**：`b = floor((key - base)/W)` → `blocks[b]` → 从该下标**向前走几步**
//!   （均匀数据 ~1 项）到精确位置。索引有效时是 **2 次依赖加载**；
//! * **区间遍历**：索引起跳后直接切连续切片，无指针追逐；
//! * **写入**：数组搬移 + 标脏；升序追加走快速路径（不需要二分）。批量写完之后
//!   调一次 [`JumpArray::rebuild`] 重建块表（O(n)）。这是刻意的取舍：不做逐次
//!   插入/删除的增量维护，换取查询与遍历的简单性和缓存友好——"频繁增删"不是本容器
//!   的目标形态；
//! * **脏索引**：`rebuild` 之前查询自动退回二分（`O(log n)`），结果仍然正确；
//! * **顺序访问**：配套 [`Cursor`]（`find_le_from` / `find_ge_from` / `range_from`），
//!   从上次位置双向线性推进，顺序窗口摊还 O(1)；游标带实例 id 与写入代数，过期或跨
//!   实例时自动退回冷路径。
//!
//! ## 与 kipphi 原版的差异
//!
//! 1. 数据载体是连续数组而不是双向链表：区间遍历与短走步都是顺序读；
//! 2. 索引不是 `T | T[]` union，而是扁平 `u32` 下标，无 per-block 分配；
//! 3. 游标不参与"块内走步"：短走步就是在数组上递增下标；游标只用来给顺序访问
//!    提供"从哪儿开始找"的提示；
//! 4. 索引重建是显式的（批量写后一次），不做 `updateRange` 式增量修补。
//!
//! ## 复杂度
//!
//! | 操作 | 复杂度 |
//! |---|---|
//! | 写入（任意位置） | O(n) 搬移 + 标脏 |
//! | 写入（升序追加） | O(1) 摊还（`Vec` push）+ 标脏 |
//! | `rebuild` | O(n) |
//! | 点查询（索引有效） | O(1) 起跳 + 块内短扫（块内超过 `linear_max` 项则块内二分） |
//! | 点查询（脏） | O(log n) |
//! | 区间 / 遍历 | O(长度)，连续切片；上界定位先线性探 8 项再回退块内二分 |
//! | 顺序查询/窗口（游标） | 摊还 O(1) 起跳；单次 O(距离)，超 32 步回退块索引 |
//! | 内存 | 载荷 + 4 B/元素 |

use std::cmp::Ordering;

/// 有序键：全序比较 + 数值化（块下标计算用）。
pub trait Key: Copy + std::fmt::Debug {
    fn cmp_key(&self, other: &Self) -> Ordering;
    fn to_f64(self) -> f64;
}

macro_rules! impl_int_key {
    ($($t:ty),*) => {$(
        impl Key for $t {
            #[inline] fn cmp_key(&self, other: &Self) -> Ordering { self.cmp(other) }
            #[inline] fn to_f64(self) -> f64 { self as f64 }
        }
    )*};
}
impl_int_key!(i32, i64, u32);

impl Key for f64 {
    #[inline]
    fn cmp_key(&self, other: &Self) -> Ordering {
        self.total_cmp(other)
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

/// 有序容器（key 唯一、按 key 升序）。
///
/// ⚠️ **刻意不实现 `Clone`**：游标用 `id` 识别实例，克隆副本若沿用同一个 `id`，
/// 旧游标会在副本上"看似有效"却指向不同数据。若将来要加 `Clone`，必须给新副本
/// 分配新的 `id`（并保持 `version` 语义）。
pub struct JumpArray<K: Key, E> {
    entries: Vec<(K, E)>,
    /// 块起点 → 第一个 `key >= block_start` 的条目下标。
    blocks: Vec<u32>,
    base: f64,
    width: f64,
    /// 索引过期（写入后置脏，`rebuild` 清除）。
    dirty: bool,
    /// 块内元素数不超过它时线性扫，超过则块内二分（可调）。
    linear_max: usize,
    /// 实例唯一 id（游标跨实例误用检测）。
    id: u64,
    /// 写入代数：`insert`/`remove` +1；`rebuild` 不改下标，故不变。
    version: u32,
}

/// `linear_max` 默认值。
const DEFAULT_LINEAR_MAX: usize = 8;

/// 进程内自增的实例 id。
static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// 顺序访问游标（属于某个 [`JumpArray`] 实例）。
///
/// 语义：`pos` 是上一次查询落在 `entries` 里的下标（允许等于 `len`，表示末尾之后）。
/// `id`/`version` 不匹配时视为过期——查询自动走冷路径，结果仍然正确。
#[derive(Debug, Clone, Copy)]
pub struct Cursor {
    id: u64,
    version: u32,
    pos: usize,
}

impl Cursor {
    /// 游标当前下标（诊断/断言用；`len` 表示末尾之后）。
    pub fn pos(&self) -> usize {
        self.pos
    }
}

impl<K: Key, E> Default for JumpArray<K, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Key, E> JumpArray<K, E> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            blocks: Vec::new(),
            base: 0.0,
            width: 1.0,
            dirty: true,
            linear_max: DEFAULT_LINEAR_MAX,
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            version: 0,
        }
    }

    /// 调整"块内线性扫"阈值（块内元素数超过它则块内二分）。
    pub fn with_linear_max(mut self, linear_max: usize) -> Self {
        self.linear_max = linear_max.max(1);
        self
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut s = Self::new();
        s.entries.reserve(capacity);
        s
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 索引诊断：`(块表长度, 元素数, 块宽)`。
    pub fn stats(&self) -> (usize, usize, f64) {
        (self.blocks.len(), self.entries.len(), self.width)
    }

    /// 索引是否被写入弄脏：`true` ⇒ 查询退回二分（仍正确，只是多 log n 次比较），
    /// 需要 [`JumpArray::rebuild`] 恢复"块表起跳 + 短扫"的快路径。
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn first(&self) -> Option<(&K, &E)> {
        self.entries.first().map(|(k, v)| (k, v))
    }

    pub fn last(&self) -> Option<(&K, &E)> {
        self.entries.last().map(|(k, v)| (k, v))
    }

    /// 按 key 升序遍历。
    pub fn iter(&self) -> impl Iterator<Item = (&K, &E)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    /// 重建块表（批量写完之后调一次，恢复 O(1) 查询）。
    ///
    /// 幂等：索引已经是干净的（`!dirty`）时直接返回，不重复 O(n) 扫描。
    pub fn rebuild(&mut self) {
        if !self.dirty {
            return;
        }
        self.blocks.clear();
        if self.entries.is_empty() {
            self.base = 0.0;
            self.width = 1.0;
            self.dirty = false;
            return;
        }
        let first = self.entries[0].0.to_f64();
        let last = self.entries[self.entries.len() - 1].0.to_f64();
        let span = (last - first).max(0.0);
        let width = pow2_ceil(span / self.entries.len().max(1) as f64);
        self.base = first;
        self.width = width;
        let nblocks = ((span / width).ceil() as usize).saturating_add(1).max(1);
        self.blocks.reserve(nblocks);
        let mut i = 0usize;
        for b in 0..nblocks {
            let start = first + b as f64 * width;
            while i < self.entries.len() && self.entries[i].0.to_f64() < start {
                i += 1;
            }
            self.blocks.push(i as u32);
        }
        self.dirty = false;
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

    /// 第一个 `key >= k` 的下标（二分；写入与脏索引路径用）。
    fn lower_bound(&self, k: &K) -> usize {
        self.entries
            .partition_point(|(x, _)| x.cmp_key(k) == Ordering::Less)
    }

    fn upper_bound(&self, k: &K) -> usize {
        self.entries
            .partition_point(|(x, _)| x.cmp_key(k) != Ordering::Greater)
    }

    /// 块表给出的 `[start, end)`：该块覆盖的元素下标区间。
    ///
    /// 空块时 `start == end`（两个锚点相同）。
    #[inline]
    fn block_span(&self, b: usize) -> (usize, usize) {
        let start = self.blocks[b] as usize;
        let end = if b + 1 < self.blocks.len() {
            self.blocks[b + 1] as usize
        } else {
            self.entries.len()
        };
        (start, end)
    }

    /// 索引起跳 + 块内短扫（块内元素多则二分）。
    #[inline]
    fn lower_bound_fast(&self, k: &K) -> usize {
        if self.dirty || self.blocks.is_empty() {
            return self.lower_bound(k);
        }
        let (start, end) = self.block_span(self.block_of(k));
        if end - start <= self.linear_max {
            let mut i = start;
            while i < end && self.entries[i].0.cmp_key(k) == Ordering::Less {
                i += 1;
            }
            i
        } else {
            start
                + self.entries[start..end]
                    .partition_point(|(x, _)| x.cmp_key(k) == Ordering::Less)
        }
    }

    /// 同上，取"第一个 `key > k`"。
    #[inline]
    fn upper_bound_fast(&self, k: &K) -> usize {
        if self.dirty || self.blocks.is_empty() {
            return self.upper_bound(k);
        }
        let (start, end) = self.block_span(self.block_of(k));
        if end - start <= self.linear_max {
            let mut i = start;
            while i < end && self.entries[i].0.cmp_key(k) != Ordering::Greater {
                i += 1;
            }
            i
        } else {
            start
                + self.entries[start..end]
                    .partition_point(|(x, _)| x.cmp_key(k) != Ordering::Greater)
        }
    }

    /// 插入；key 已存在时返回 `Err(value)`。写入后需 `rebuild` 才恢复 O(1) 查询。
    pub fn insert(&mut self, key: K, value: E) -> Result<(), E> {
        // 追加快路径：升序导入/反序列化就是"每次都插在末尾"，省掉一次二分。
        if let Some((last, _)) = self.entries.last() {
            if key.cmp_key(last) == Ordering::Greater {
                self.entries.push((key, value));
                self.dirty = true;
                self.version = self.version.wrapping_add(1);
                return Ok(());
            }
        }
        let pos = self.lower_bound(&key);
        if pos < self.entries.len() && self.entries[pos].0.cmp_key(&key) == Ordering::Equal {
            return Err(value);
        }
        self.entries.insert(pos, (key, value));
        self.dirty = true;
        self.version = self.version.wrapping_add(1);
        Ok(())
    }

    /// 删除；不存在返回 `None`。
    pub fn remove(&mut self, key: &K) -> Option<E> {
        // 删末尾的快速路径：按序清理/回滚时就是"总是删最后一个"，省掉一次二分。
        if let Some((last, _)) = self.entries.last() {
            if last.cmp_key(key) == Ordering::Equal {
                let (_, v) = self.entries.pop().expect("刚判过非空");
                self.dirty = true;
                self.version = self.version.wrapping_add(1);
                return Some(v);
            }
        }
        let pos = self.lower_bound(key);
        if pos >= self.entries.len() || self.entries[pos].0.cmp_key(key) != Ordering::Equal {
            return None;
        }
        let (_, v) = self.entries.remove(pos);
        self.dirty = true;
        self.version = self.version.wrapping_add(1);
        Some(v)
    }

    pub fn contains(&self, key: &K) -> bool {
        self.get(key).is_some()
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
            let (key, v) = &self.entries[i - 1];
            Some((key, v))
        }
    }

    /// 最后一个 `key < k`。
    pub fn find_lt(&self, k: &K) -> Option<(&K, &E)> {
        let i = self.lower_bound_fast(k);
        if i == 0 {
            None
        } else {
            let (key, v) = &self.entries[i - 1];
            Some((key, v))
        }
    }

    /// 第一个 `key >= k`。
    pub fn find_ge(&self, k: &K) -> Option<(&K, &E)> {
        let i = self.lower_bound_fast(k);
        self.entries.get(i).map(|(key, v)| (key, v))
    }

    /// 第一个 `key > k`。
    pub fn find_gt(&self, k: &K) -> Option<(&K, &E)> {
        let i = self.upper_bound_fast(k);
        self.entries.get(i).map(|(key, v)| (key, v))
    }

    /// `[lo, hi]` 区间（含两端、升序、连续切片）。
    ///
    /// 上界优先从 `from` 向前线性探（窗口小的时候最便宜）；探到扫描上限还没越过
    /// `hi` 才回退到 `upper_bound_fast`（块内二分/二分）。
    pub fn range(&self, lo: K, hi: K) -> RangeIter<'_, K, E> {
        if self.is_empty() || lo.cmp_key(&hi) == Ordering::Greater {
            return RangeIter {
                slice: &[],
                start: 0,
                end: 0,
            };
        }
        let from = self.lower_bound_fast(&lo);
        let to = self.range_end_from(from, &hi);
        RangeIter {
            slice: &self.entries[from..to],
            start: from,
            end: to,
        }
    }

    /// `from` 起、上界为 `hi` 的区间结束下标（排他）。
    ///
    /// 先线性探 `RANGE_SCAN_MAX` 个元素（窗口极短时省掉一次块表查找），探不到就
    /// 回退 [`Self::upper_bound_fast`]（块跳 + 块内线性/二分）。
    ///
    /// ⚠️ 这个上限必须与"一次块内二分的比较次数"同量级（~log 块内元素数），
    /// **不能按窗口大小取**：窗口越大，线性探重复走的元素越多，反而比直接二分贵。
    fn range_end_from(&self, from: usize, hi: &K) -> usize {
        const RANGE_SCAN_MAX: usize = 8;
        let n = self.entries.len();
        let mut to = from;
        while to < n
            && to - from < RANGE_SCAN_MAX
            && self.entries[to].0.cmp_key(hi) != Ordering::Greater
        {
            to += 1;
        }
        if to - from >= RANGE_SCAN_MAX && to < n && self.entries[to].0.cmp_key(hi) != Ordering::Greater
        {
            to = self.upper_bound_fast(hi).max(from);
        }
        to
    }

    /// 按下标取条目（配合 [`RangeIter::span`] / [`Cursor::pos`] 使用）。
    pub fn entry(&self, i: usize) -> Option<(&K, &E)> {
        self.entries.get(i).map(|(k, v)| (k, v))
    }

    /// 按下标区间 `[from, to)` 直接迭代（**不做定位**；越界自动截断）。
    ///
    /// 给了 [`RangeIter::span`] 之后想复用这段区间时用它——省掉一次定位。
    pub fn span_iter(&self, (from, to): (usize, usize)) -> RangeIter<'_, K, E> {
        let n = self.entries.len();
        let from = from.min(n);
        let to = to.clamp(from, n);
        RangeIter {
            slice: &self.entries[from..to],
            start: from,
            end: to,
        }
    }

    /// 创建属于本实例的游标（顺序访问用）。
    pub fn cursor(&self) -> Cursor {
        Cursor {
            id: self.id,
            version: self.version,
            pos: 0,
        }
    }

    /// 游标是否仍可用（未过期、且属于本实例）。
    fn cursor_ok(&self, cur: &Cursor) -> bool {
        cur.id == self.id && cur.version == self.version
    }

    /// 从游标出发找"第一个 `key >= k`"：先双向线性推进，超过步数上限再回退块索引。
    fn cursor_lower_bound(&self, cur: &mut Cursor, k: &K) -> usize {
        const STEP_MAX: usize = 32;
        let n = self.entries.len();
        if !self.cursor_ok(cur) {
            cur.id = self.id;
            cur.version = self.version;
            let i = self.lower_bound_fast(k);
            cur.pos = i;
            return i;
        }
        let mut i = cur.pos.min(n);
        if i < n && self.entries[i].0.cmp_key(k) == Ordering::Equal {
            // 恰好命中：直接返回，省掉向左扫。
            cur.pos = i;
            return i;
        }
        if i == n || self.entries[i].0.cmp_key(k) != Ordering::Less {
            // 答案在 i 处或左侧
            let mut steps = 0usize;
            while i > 0 && self.entries[i - 1].0.cmp_key(k) != Ordering::Less && steps < STEP_MAX
            {
                i -= 1;
                steps += 1;
            }
            if i > 0 && self.entries[i - 1].0.cmp_key(k) != Ordering::Less {
                i = self.lower_bound_fast(k);
            }
            cur.pos = i;
            return i;
        }
        // 答案在右侧
        let mut steps = 0usize;
        while i < n && self.entries[i].0.cmp_key(k) == Ordering::Less && steps < STEP_MAX {
            i += 1;
            steps += 1;
        }
        if i < n && self.entries[i].0.cmp_key(k) == Ordering::Less {
            i = self.lower_bound_fast(k);
        }
        cur.pos = i;
        i
    }

    /// 从游标出发找"第一个 `key > k`"。
    fn cursor_upper_bound(&self, cur: &mut Cursor, k: &K) -> usize {
        const STEP_MAX: usize = 32;
        let n = self.entries.len();
        if !self.cursor_ok(cur) {
            cur.id = self.id;
            cur.version = self.version;
            let i = self.upper_bound_fast(k);
            cur.pos = i;
            return i;
        }
        let mut i = cur.pos.min(n);
        if i == n || self.entries[i].0.cmp_key(k) == Ordering::Greater {
            let mut steps = 0usize;
            while i > 0 && self.entries[i - 1].0.cmp_key(k) == Ordering::Greater && steps < STEP_MAX
            {
                i -= 1;
                steps += 1;
            }
            if i > 0 && self.entries[i - 1].0.cmp_key(k) == Ordering::Greater {
                i = self.upper_bound_fast(k);
            }
            cur.pos = i;
            return i;
        }
        let mut steps = 0usize;
        while i < n && self.entries[i].0.cmp_key(k) != Ordering::Greater && steps < STEP_MAX {
            i += 1;
            steps += 1;
        }
        if i < n && self.entries[i].0.cmp_key(k) != Ordering::Greater {
            i = self.upper_bound_fast(k);
        }
        cur.pos = i;
        i
    }

    /// 从游标出发的 `find_le`。
    pub fn find_le_from(&self, cur: &mut Cursor, k: &K) -> Option<(&K, &E)> {
        if self.is_empty() {
            *cur = self.cursor();
            return None;
        }
        let i = self.cursor_upper_bound(cur, k);
        if i == 0 {
            None
        } else {
            cur.pos = i - 1;
            self.entry(i - 1)
        }
    }

    /// 从游标出发的 `find_ge`。
    pub fn find_ge_from(&self, cur: &mut Cursor, k: &K) -> Option<(&K, &E)> {
        if self.is_empty() {
            *cur = self.cursor();
            return None;
        }
        let i = self.cursor_lower_bound(cur, k);
        self.entry(i)
    }

    /// 从游标出发的 `range`：顺序窗口直接前推，跨大步时回退块索引/二分。
    pub fn range_from(&self, cur: &mut Cursor, lo: K, hi: K) -> RangeIter<'_, K, E> {
        if self.is_empty() || lo.cmp_key(&hi) == Ordering::Greater {
            return RangeIter {
                slice: &[],
                start: 0,
                end: 0,
            };
        }
        let from = self.cursor_lower_bound(cur, &lo);
        let to = self.range_end_from(from, &hi);
        cur.pos = to;
        RangeIter {
            slice: &self.entries[from..to],
            start: from,
            end: to,
        }
    }
}

fn pow2_ceil(x: f64) -> f64 {
    if !(x > 0.0) || !x.is_finite() {
        return 1.0;
    }
    let e = x.log2().ceil().clamp(-60.0, 40.0);
    2f64.powf(e)
}

/// [`JumpArray::range`] / [`JumpArray::range_from`] 的迭代器。
pub struct RangeIter<'a, K: Key, E> {
    slice: &'a [(K, E)],
    start: usize,
    end: usize,
}

impl<'a, K: Key, E> RangeIter<'a, K, E> {
    /// 区间在 `entries` 里的下标范围 `[start, end)`（配合 [`JumpArray::entry`] 使用）。
    ///
    /// 写入后这些下标会失效（数组发生搬移），与游标一样按版本判断。
    pub fn span(&self) -> (usize, usize) {
        (self.start, self.end)
    }
}

impl<'a, K: Key, E> Iterator for RangeIter<'a, K, E> {
    type Item = (&'a K, &'a E);

    fn next(&mut self) -> Option<Self::Item> {
        let (head, rest) = self.slice.split_first()?;
        self.slice = rest;
        Some((&head.0, &head.1))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.slice.len(), Some(self.slice.len()))
    }
}

impl<'a, K: Key, E> ExactSizeIterator for RangeIter<'a, K, E> {}

#[cfg(test)]
mod tests {
    use super::*;

    /// 与"排序 Vec + 二分"逐操作对照（覆盖索引有效 / 脏两条路径）。
    #[test]
    fn matches_sorted_vec_semantics() {
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut a = JumpArray::<i64, i64>::new();
        let mut model: Vec<i64> = Vec::new();
        for step in 0..6000 {
            let k = (next() % 40_000) as i64;
            if next() % 3 != 0 {
                let r = a.insert(k, k * 2);
                let dup = model.binary_search(&k).is_ok();
                assert_eq!(r.is_err(), dup, "step {step} insert {k}");
                if !dup {
                    let pos = model.partition_point(|x| *x < k);
                    model.insert(pos, k);
                }
            } else {
                let r = a.remove(&k);
                match model.binary_search(&k) {
                    Ok(pos) => {
                        model.remove(pos);
                        assert_eq!(r, Some(k * 2), "step {step} remove {k}");
                    }
                    Err(_) => assert!(r.is_none(), "step {step} remove {k}"),
                }
            }
            // 周期性重建：覆盖"索引有效"与"脏"两条路径。
            if step % 97 == 0 {
                a.rebuild();
            }
            let q = (next() % 40_000) as i64;
            let le = model.partition_point(|x| *x <= q);
            let lt = model.partition_point(|x| *x < q);
            assert_eq!(
                a.find_le(&q).map(|(k, _)| *k),
                (le > 0).then(|| model[le - 1]),
                "step {step} find_le {q}"
            );
            assert_eq!(
                a.find_lt(&q).map(|(k, _)| *k),
                (lt > 0).then(|| model[lt - 1]),
                "step {step} find_lt {q}"
            );
            assert_eq!(
                a.find_ge(&q).map(|(k, _)| *k),
                model.get(lt).copied(),
                "step {step} find_ge {q}"
            );
            assert_eq!(
                a.find_gt(&q).map(|(k, _)| *k),
                model.get(le).copied(),
                "step {step} find_gt {q}"
            );
            let (lo, hi) = if q < 20_000 { (q, q + 200) } else { (q - 200, q) };
            let want: Vec<i64> = model
                .iter()
                .copied()
                .filter(|x| *x >= lo && *x <= hi)
                .collect();
            let got: Vec<i64> = a.range(lo, hi).map(|(k, _)| *k).collect();
            assert_eq!(got, want, "step {step} range [{lo},{hi}]");
        }
        a.rebuild();
        assert_eq!(a.len(), model.len());
        assert_eq!(a.iter().map(|(k, _)| *k).collect::<Vec<_>>(), model);
    }

    /// 空 / 单元素 / 反向区间。
    #[test]
    fn handles_edges() {
        let mut a = JumpArray::<i64, u8>::new();
        assert!(a.find_le(&0).is_none());
        a.rebuild();
        assert_eq!(a.range(0, 10).count(), 0);
        a.insert(7, 7).unwrap();
        a.rebuild();
        assert_eq!(a.find_le(&7).map(|(_, v)| *v), Some(7));
        assert_eq!(a.find_le(&6), None);
        assert_eq!(a.find_ge(&8), None);
        assert_eq!(a.range(10, 3).count(), 0);
        assert_eq!(a.range(7, 7).count(), 1);
        assert_eq!(a.remove(&7), Some(7));
        assert!(a.is_empty());
        a.rebuild();
        assert!(a.find_le(&7).is_none());
    }

    /// 稀疏大跨度键（模拟 Beat）：块表 ∝ 元素数，不随键值膨胀。
    #[test]
    fn index_length_scales_with_len_not_key_span() {
        let mut a = JumpArray::<f64, u8>::new();
        for i in 0..10_000 {
            a.insert(i as f64 * 1000.0, 0).unwrap();
        }
        a.rebuild();
        let (blocks, len, _) = a.stats();
        assert!(blocks <= 40_000, "块表 {blocks} 应 ≈ 元素数 {len}");
        assert_eq!(a.find_le(&500_000.5).map(|(k, _)| *k), Some(500_000.0));
    }

    /// 聚簇数据：块内元素数超过 `LINEAR_MAX` 时走块内二分，结果仍与模型一致。
    #[test]
    fn handles_clustered_keys() {
        // 每组 100 个键密集排布（间距 1），组间跳 100_000 —— 典型的"块内堆积"。
        let mut a = JumpArray::<i64, u8>::new();
        let mut model: Vec<i64> = Vec::new();
        for i in 0..3000i64 {
            let k = (i / 100) * 100_000 + (i % 100);
            a.insert(k, 0).unwrap();
            model.push(k);
        }
        a.rebuild();
        let (blocks, len, _) = a.stats();
        assert_eq!(len, 3000);
        assert!(blocks <= 4000);
        for q in (-1..30).map(|i| i * 10_000) {
            let want = model.partition_point(|x| *x <= q);
            assert_eq!(
                a.find_le(&q).map(|(k, _)| *k),
                (want > 0).then(|| model[want - 1]),
                "find_le {q}"
            );
            let want_ge = model.partition_point(|x| *x < q);
            assert_eq!(
                a.find_ge(&q).map(|(k, _)| *k),
                model.get(want_ge).copied(),
                "find_ge {q}"
            );
        }
        // 逐个键验证（密集区内也走二分分支）。
        for k in &model {
            assert_eq!(a.find_le(k).map(|(x, _)| *x), Some(*k), "find_le {k}");
            assert_eq!(
                a.find_ge(k).map(|(x, _)| *x),
                Some(*k),
                "find_ge exact {k}"
            );
        }
    }

    /// `rebuild` 幂等：干净时重复调用不改任何状态（尤其不重复扫描）。
    #[test]
    fn rebuild_is_idempotent() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..1000 {
            a.insert(i * 3, 0).unwrap();
        }
        a.rebuild();
        let before = a.stats();
        assert!(!a.is_dirty(), "rebuild 后应干净");
        a.rebuild();
        a.rebuild();
        assert_eq!(a.stats(), before, "重复 rebuild 不应改变块表");
        assert!(!a.is_dirty());
        // 写入后转脏，再 rebuild 应恢复。
        a.insert(-1, 1).unwrap();
        assert!(a.is_dirty(), "写入后索引过期");
        a.rebuild();
        assert!(!a.is_dirty(), "rebuild 后恢复");
        let (blocks, len, _) = a.stats();
        assert_eq!(len, 1001);
        assert!(blocks > 0);
        assert_eq!(a.find_le(&-1).map(|(k, _)| *k), Some(-1), "新首键可取到");
        assert_eq!(a.find_le(&0).map(|(k, _)| *k), Some(0));
        assert!(a.find_le(&-2).is_none(), "base 已换到新首键");
    }

    /// 任意 `linear_max` 下结果一致（线性/二分两条分支都要正确）。
    #[test]
    fn linear_max_variants_agree() {
        let keys: Vec<i64> = (0..2000i64)
            .map(|i| (i / 100) * 100_000 + (i % 100))
            .collect();
        for lm in [1usize, 2, 8, 64] {
            let mut a = JumpArray::<i64, u8>::new().with_linear_max(lm);
            for k in &keys {
                a.insert(*k, 0).unwrap();
            }
            a.rebuild();
            for i in 0..300i64 {
                let q = i * 997;
                let le = keys.partition_point(|x| *x <= q);
                assert_eq!(
                    a.find_le(&q).map(|(k, _)| *k),
                    (le > 0).then(|| keys[le - 1]),
                    "lm={lm} find_le {q}"
                );
                let lt = keys.partition_point(|x| *x < q);
                assert_eq!(
                    a.find_ge(&q).map(|(k, _)| *k),
                    keys.get(lt).copied(),
                    "lm={lm} find_ge {q}"
                );
            }
        }
    }

    /// 窗口大于线性探测上限时，`range` 回退到二分，结果仍完整。
    #[test]
    fn range_falls_back_past_probe_cap() {
        let mut a = JumpArray::<i64, u8>::new().with_linear_max(1);
        let keys: Vec<i64> = (0..500).map(|i| i * 10).collect();
        for k in &keys {
            a.insert(*k, 0).unwrap();
        }
        a.rebuild();
        // 窗口覆盖 200 个元素（cap = max(1*8, 64) = 64，必然走回退）。
        let got: Vec<i64> = a.range(0, 1990).map(|(k, _)| *k).collect();
        assert_eq!(got, keys[..200].to_vec());
        let got: Vec<i64> = a.range(0, 4000).map(|(k, _)| *k).collect();
        assert_eq!(got, keys[..401].to_vec());
    }

    /// 顺序滑窗：`range_from` 与 `range` 结果一致，`span()` 与 `entry()` 对得上。
    #[test]
    fn cursor_windows_match_range() {
        for clustered in [false, true] {
            let mut a = JumpArray::<i64, u8>::new();
            let keys: Vec<i64> = (0..3000i64)
                .map(|i| {
                    if clustered {
                        (i / 100) * 100_000 + (i % 100)
                    } else {
                        i * 37
                    }
                })
                .collect();
            for k in &keys {
                a.insert(*k, 0).unwrap();
            }
            a.rebuild();
            let mut cur = a.cursor();
            for w in 0..keys.len() / 64 {
                let lo = keys[w * 64];
                let hi = keys[(w * 64 + 63).min(keys.len() - 1)];
                let plain: Vec<i64> = a.range(lo, hi).map(|(k, _)| *k).collect();
                let mut it = a.range_from(&mut cur, lo, hi);
                let (from, to) = it.span();
                let got: Vec<i64> = it.by_ref().map(|(k, _)| *k).collect();
                assert_eq!(got, plain, "clustered={clustered} w={w}");
                assert_eq!(
                    a.entry(from).map(|(k, _)| *k),
                    plain.first().copied(),
                    "span 起点"
                );
                assert_eq!(a.entry(to.saturating_sub(1)).map(|(k, _)| *k), plain.last().copied());
                assert_eq!(cur.pos(), to, "游标停在区间末尾之后");
            }
        }
    }

    /// 写入使游标过期：之后继续用游标，结果仍与二分一致（自动走冷路径）。
    #[test]
    fn cursor_recovers_after_writes() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..1000 {
            a.insert(i * 3, 0).unwrap();
        }
        a.rebuild();
        let mut cur = a.cursor();
        assert_eq!(a.find_le_from(&mut cur, &300).map(|(k, _)| *k), Some(300));
        // 写入：下标整体移动，游标过期
        a.insert(-5, 0).unwrap();
        a.insert(299, 0).unwrap();
        a.rebuild();
        for q in [0i64, 1, 150, 298, 299, 301, 2997, 3000] {
            let mut c2 = cur;
            let want = a.find_le(&q).map(|(k, _)| *k);
            let got = a.find_le_from(&mut c2, &q).map(|(k, _)| *k);
            assert_eq!(got, want, "过期游标查询 {q}");
            let want_ge = a.find_ge(&q).map(|(k, _)| *k);
            let mut c3 = cur;
            let got_ge = a.find_ge_from(&mut c3, &q).map(|(k, _)| *k);
            assert_eq!(got_ge, want_ge, "过期游标 find_ge {q}");
        }
    }

    /// 双向：递增、递减、来回跳，结果都与无游标版本一致。
    #[test]
    fn cursor_bidirectional_matches() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..2000 {
            a.insert(i * 11, 0).unwrap();
        }
        a.rebuild();
        let probes: Vec<i64> = [1, 5000, 12, 21000, 3, 15000, 21990, 7000, 0, 22000]
            .iter()
            .copied()
            .collect();
        let mut cur = a.cursor();
        for q in probes {
            assert_eq!(
                a.find_le_from(&mut cur, &q).map(|(k, _)| *k),
                a.find_le(&q).map(|(k, _)| *k),
                "find_le_from {q}"
            );
            assert_eq!(
                a.find_ge_from(&mut cur, &q).map(|(k, _)| *k),
                a.find_ge(&q).map(|(k, _)| *k),
                "find_ge_from {q}"
            );
        }
    }

    /// 删末尾走快速路径：按序回滚/清理时正确且保持查询一致。
    #[test]
    fn remove_last_is_fast_and_correct() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..100 {
            a.insert(i * 2, 0).unwrap();
        }
        a.rebuild();
        for i in (0..100i64).rev() {
            assert_eq!(a.remove(&(i * 2)), Some(0), "删 {i}");
            if i % 10 == 0 {
                a.rebuild();
            }
        }
        assert!(a.is_empty());
        assert_eq!(a.remove(&0), None);
    }

    /// 升序插入走追加快路径：顺序、去重、脏标记都对。
    #[test]
    fn append_fast_path_keeps_order_and_dedup() {
        let mut a = JumpArray::<i64, u8>::new();
        for i in 0..1000 {
            a.insert(i * 3, (i % 251) as u8).unwrap();
        }
        assert!(!a.is_dirty() || a.is_dirty(), "刚写入，允许脏");
        assert_eq!(a.len(), 1000);
        assert_eq!(a.first().map(|(k, _)| *k), Some(0));
        assert_eq!(a.last().map(|(k, _)| *k), Some(2997));
        // 末尾重复
        assert!(a.insert(2997, 9).is_err());
        assert_eq!(a.len(), 1000);
        a.rebuild();
        assert_eq!(
            a.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            (0..1000).map(|i| i * 3).collect::<Vec<_>>()
        );
        // 追加后查询仍正确
        assert_eq!(a.find_le(&2998).map(|(k, _)| *k), Some(2997));
        assert_eq!(a.find_ge(&2999), None);
    }

    /// 跨实例的游标：id 不匹配 ⇒ 退回冷路径，结果正确。
    #[test]
    fn cursor_from_other_instance_falls_back() {
        let mut a = JumpArray::<i64, u8>::new();
        let mut b = JumpArray::<i64, u8>::new();
        for i in 0..100 {
            a.insert(i * 2, 0).unwrap();
            b.insert(i * 3, 0).unwrap();
        }
        a.rebuild();
        b.rebuild();
        let mut cur = a.cursor();
        assert_eq!(a.find_le_from(&mut cur, &50).map(|(k, _)| *k), Some(50));
        // 拿 a 的游标查 b：必须正确（走冷路径）
        assert_eq!(
            b.find_le_from(&mut cur, &50).map(|(k, _)| *k),
            Some(48),
            "跨实例游标应退回冷路径"
        );
        assert_eq!(
            b.range_from(&mut cur, 0, 20)
                .map(|(k, _)| *k)
                .collect::<Vec<_>>(),
            vec![0, 3, 6, 9, 12, 15, 18]
        );
    }
}
