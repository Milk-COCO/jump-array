//! `JumpArray`：kipphi 的"块索引 + 短走步"思想，落在**键值分数组（SoA）**的连续存储上。
//!
//! ## 结构
//!
//! ```text
//! keys:   Vec<K>    按 key 升序、连续存放
//! vals:   Vec<E>    与 keys 同下标一一对应
//! blocks: Vec<u32>  块起点 → 第一个 key ≥ 块起点的下标
//! ```
//!
//! * **块宽** `W = 2^ceil(log2(span/n))` ⇒ 块数 ≈ 元素数，与键的数值大小无关；
//! * **查询**：`b = floor((key - base)/W)` → `blocks[b]` → 在 **keys 数组**上
//!   向前走几步（块内元素数 ≤ `linear_max` 时线性扫，否则块内二分），命中后
//!   用同一下标取 `vals`；
//! * **区间遍历**：索引起跳后切出两个等长切片，zip 迭代——连续、无指针追逐；
//! * **键值分数组**：块内扫描/二分只碰键数组（无值载荷）；且没有 `(K,E)` 元组的
//!   对齐填充（例如 `(i64,u32)` 在 AoS 里占 16 B，这里只占 8+4=12 B）；
//! * **写入**：两个数组同步搬移 + 标脏。批量写完之后调一次 [`JumpArray::rebuild`]
//!   重建块表（O(n)）。升序追加走 `push` 快路径；
//! * **脏索引**：`rebuild` 之前查询自动退回二分（O(log n)），结果仍然正确；
//! * **小数组**：元素数 ≤ `binary_below`（默认 2048）时直接二分，跳过块索引；
//! * **二级锚**（默认关闭）：`with_minor_anchors(true)` 时给"块内元素数 > 8"的
//!   密集块再加 8 格锚点，块内定位从 ~7 次比较降到 1 次加载 + 短扫。实测对聚簇
//!   随机查询略好、对顺序查询明显更差，故默认不开；
//! * **顺序访问**：配套 [`Cursor`]（`find_le_from` / `find_ge_from` / `range_from`），
//!   从上次位置双向线性推进，顺序窗口摊还 O(1)；游标带实例 id 与写入代数，过期或跨
//!   实例时自动退回冷路径。
//!
//! ## 与 kipphi 原版的差异
//!
//! 1. 数据载体是分数组而不是双向链表：区间遍历与短走步都是顺序读；
//! 2. 索引不是 `T | T[]` union，而是扁平 `u32` 下标，无 per-block 分配；
//! 3. 游标不参与"块内走步"：短走步就是在键数组上递增下标；游标只给顺序访问提供提示；
//! 4. 索引重建是显式的（批量写后一次），不做 `updateRange` 式增量修补。
//!
//! ## 复杂度
//!
//! | 操作 | 复杂度 |
//! |---|---|
//! | 写入（任意位置） | O(n) 搬移 + 标脏 |
//! | 写入（升序追加） | O(1) 摊还 + 标脏 |
//! | `rebuild` | O(n)（幂等：干净时 O(1)） |
//! | 点查询（索引有效） | O(1) 起跳 + 块内短扫 |
//! | 点查询（脏 / 小数组） | O(log n) |
//! | 区间 / 遍历 | O(长度)，连续切片 |
//! | 顺序查询/窗口（游标） | 摊还 O(1) 起跳；单次 O(距离)，超 32 步回退块索引 |
//! | 内存 | 载荷（无对齐填充）+ 4 B/元素 |

use std::cmp::Ordering;

pub mod soa;
pub use soa::{Cursor, JumpArray, RangeIter};

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
