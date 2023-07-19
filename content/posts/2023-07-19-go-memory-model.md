---
title: "Go 内存模型与分配机制"
date: 2023-07-19T13:10:33+08:00
draft: false
subtitle: ""
author: "ZhongsJie"
tags:
  - Go
  - Memory
categories:
  - Go
showToc: false # 显示目录
TocOpen: false # 自动展开目录
disableShare: true # 底部不显示分享栏
cover:
    image: "/images/2023-07/go内存模型.png"
    caption: ""
    alt: ""
    relative: false
---


> Go内存模型指定了一个goroutine中变量的读取条件，可以保证观察不同goroutine中对同一变量的写入产生的值。

## 虚拟内存

>虚拟内存技术是操作系统实现的一种高效的物理内存管理方式

![image-20200720222628264](/images/2023-07/虚拟地址.png)
- 虚拟内存通过**页表**映射到物理内存上，页表记录是否在物理内存上（**有效位**），以及物理内存页的地址
- 操作系统为每个进程提供了一个独立的页表，因此也就是一个独立的虚拟空间地址，多个虚拟页面可以映射到同一个共享物理页面上。
- **地址翻译**：一个N元素的虚拟地址空间的元素和一个M元素的物理地址空间中元素之间的映射
- 虚拟内存：利用磁盘空间**虚拟出一块逻辑内存**，用作虚拟内存的磁盘空间被称为交换空间

1. 操作系统内存管理中，一个重要概念虚拟内存:
	- 扩大地址空间
	- 内存保护
	- 公平内存分配
	- 当进程通信时，可采用虚存共享的方式实现
	- 不需要在实际物理内存的连续空间，**可以利用碎片
2. 虚拟内存的**代价**：
	- 管理需要建立很多数据结构，占用额外的内存
	- 虚拟地址到物理地址的转换，增加了指令的执行时间
	- 页面的换入换出需要磁盘I/O
	- 一页中只有部分数据，会浪费内存

## Go内存模型

>参考[tcmalloc](https://github.com/google/tcmalloc/blob/master/docs/design.md)设计，「修改由多个goroutine同时访问的数据的程序必须序列化这种访问。 要序列化访问，请使用channel操作或其他同步原语（sync和sync/atomic）保护数据。 别自作聪明。」

1. tomalloc主要有以下特点
	- 减少系统调用，避免上线文切换
	- 每个线程有缓存，避免了锁竞争
	- 复杂的设计让内存碎片化，并让内存利用率降低，tomalloc做了一定优化

### 概要
> go的早期版本里 <= 1.10 ，内存是线性分配的，就是先申请一块大内存，然后再划分各种小内存，在>=1.11版本中，golang使用稀疏(分段)内存

![内存模型](/images/2023-07/go内存模型.png)

1. 内存模型描述了程序执行的要求，程序执行由 goroutine 执行组成，而 goroutine 执行又由**内存操作**组成。
2. 内存操作：
	- 种类：表明是普通的数据读取、普通的数据写入，还是原子数据访问、互斥操作、通道操作等同步操作
	- 在程序中的位置
	- 正在访问的内存位置或变量
	- 操作读取或写入的值
3.  `mcache`、`mspan`、`mcentral` 和 `mheap` 是内存管理的四大组件，`mcache` 管理线程在本地缓存的 `mspan`，而 `mcentral` 管理着全局的 `mspan` 为所有 `mcache` 提供所有线程
	- `mheap`：全局的内存起源，访问要加全局锁
	- `mcentral`：每种对象大小规格（全局共划分为 68 种）对应的缓存，锁的粒度也仅限于同一种规格以内 （中心缓存）
	- `mcache`：GPM关系中每个P持有一份的内存缓存，mcache 的数量就是P 的数量，访问时无锁 （线程缓存）
![](/images/2023-07/Pasted%20image%2020230714130617.png)

### **mcache** （线程缓存）
> 本地缓存mcache就是从中央索引中，每一种类型的span都拿出来一个空闲的span来，放在本地队列P中
```go
type mcache struct {  
   _ sys.NotInHeap  // 不会分配到GC堆或者栈上
  
	// 会在每次访问malloc时都会被访问，所以为了更加高效缓存将其按组放在这里
   nextSample uintptr // 分配多少大小的堆时触发堆采样
   scanAlloc  uintptr // 分配的可扫描堆字节数
  
   // 小对象缓存, 当申请对象大小为 `<16KB` 的时候，会使用 `Tiny allocator` 分配器
   tiny       uintptr  
   tinyoffset uintptr  
   tinyAllocs uintptr  
  
  // 下方成员不会在每次 malloc 时被访问
   alloc [numSpanClasses]*mspan  // 当前P的分配规格信息
   stackcache [_NumStackOrders]stackfreelist  
  
   flushGen atomic.Uint32  // 表示上次刷新mcache的sweepgen（清扫生成)
}
```
1. `mcache.alloc` 是一个数组，值为 `*spans` 类型，它是 go 中管理内存的基本单元。对于`16-32 kb`大小的内存都会使用这个数组里的的 `spans` 中分配。
### **mspan** （管理单元）
1.  `mspan`：最小的管理单元。`mspan` 大小为 `page` （页最小的存储单元）的整数倍，且从 `8B` 到 `80KB` 被划分为 `67` 种不同的规格，分配对象时，会根据大小映射到不同规格的 mspan，从中获取空间.
2. 源码里定义的虽然是 `_NumSizeClasses = 68` 类，但其中包含一个大小为 `0` 的规格，此规格表示**大对象**，即 `>32KB`，这种对象只会分配到`heap`上，所以不可能出现在 `mcache.alloc` 中。
```go
type mspan struct {  
   _    sys.NotInHeap
   // 前后节点 ,双向链表
   next *mspan    
   prev *mspan   
	...
  
   startAddr uintptr   
   npages    uintptr // number of pages in span  
  
	// Object n starts at address n*elemsize + (start << pageShift).  
	freeindex uintptr  
	// 最多可以存放多少span
	nelems uintptr
   ...
   
   // 标识 mspan 等级
   spanclass spanClass
   // 标记span中的elem哪些是“被使用”的，哪些是未被使用的；清除后将释放 `allocBits` ，并将 `allocBits` 的值设置为 `gcmarkBits` 
	allocBits  *gcBits  
	gcmarkBits *gcBits
}
```
### **mcentral** （中心缓存）
> 当申请一个 `16b` 大小的内存时，如果 `mcache` 中无可用大小内存时，则它找一个最合适的规则 `mcentral` 查找

```go
type mcentral struct {  
   _         sys.NotInHeap  
   spanclass spanClass  
	// one of swept in-use spans, and one of unswept in-use span 在每轮GC期间都扮演着不同的角色。`mheap_.sweepgen` 在每轮gc期间都会递增2。
   partial [2]spanSet // list of spans with a free object  
   full    [2]spanSet // list of spans with no free objects  
}
```

1. mcentral也是存放在全局变量mheap中mheap_.central 并且在64位linux下有136 个 = 68 * 2，也就是说每个规格(spanclass)的mcentral都存在两份，其中一个用了存放需要扫描的对象（scan spanClass），另一个存放没有指针的不需要扫描的对象（noscan spanClass）
2. 每个 mcentral 对应一种 spanClass
3. 每个 mcentral 下聚合了该 spanClass 下的 mspan
4. mcentral 下的 mspan 分为两个链表，分别为有空间 mspan 链表 partial 和满空间 mspan 链表 full
5. mcentral不是内存，只是一个索引（目录）
### **mheap** （页堆）

>mheap.go文件是Go语言运行时包中（runtime）的一个文件，作用是实现Go语言的堆内存管理。其中定义了mheap结构体和相关的方法，用于在运行时环境中跟踪、分配和释放堆内存。

```go
type mheap struct {  
   _ sys.NotInHeap  
	// `lock` 全局锁，保证并发，所以尽量避免从`mheap`中分配
   lock mutex  
  
   pages pageAlloc // page allocation data structure  
  
   sweepgen uint32 
   // `allspans` 所有的 spans 都是通过 `mheap_` 申请，所有申请过的 `mspan` 都会记录在 `allspans`，可以随着堆的增长重新分配和移动
   allspans []*mspan
   // 堆arena 映射。它指向整个可用虚拟地址空间的每个 arena 帧的堆元数据,由一个L1级映射和多个L2级映射组成, 当有大量的的 arena 帧时将节省空间
   arenas [1 << arenaL1Bits]*[1 << arenaL2Bits]*heapArena
   ...
   allArenas []arenaIdx
   sweepArenas []arenaIdx
   markArenas []arenaIdx
   ...
   // 每种规格大小的块对应一个 mcentral。pad 是一个字节填充，用来避免伪共享（false sharing）
   central [numSpanClasses]struct {  
	   mcentral mcentral  
	   pad      [(cpu.CacheLinePadSize - unsafe.Sizeof(mcentral{})%cpu.CacheLinePadSize) % cpu.CacheLinePadSize]byte  
	}
	...
}
	
```
1. mheap结构体是Go语言堆内存管理核心
	- arenas
	- central
2. mheap在golang源码中是一个全局唯一变量，位置`$GOROOT/src/runtime/mheap.go/mheap_`  ，其加载顺序在 schedinit->mallocinit->mheap_.init()
3. Go中其被作为全局变量`mheap_`存储`var mheap_ mheap`
#### heapArea

1. heapArena标记为`notinheap`，表示对象自身存储在Go heap之外。其通过mheap_.arenas index 来访问，heapArena对象也直接从操作系统分配的
2. 每个 heapArena 包含 8192 个页，大小为 8192 * 8KB = 64 MB
3. heapArena 记录了**页到 mspan 的映射**. GC 时，通过地址偏移找到页很方便，但找到其所属的 mspan 不容易. 因此需要通过这个映射信息进行辅助.
4. heapArena 是 mheap 向虚拟内存申请内存的单位
5. 所有的heapArena组成了mheap（Go的堆内存）

## 内存分配

> **Go语言中采用了分级分配的策略**。将一个heapArena中划分成许多大小相等的小格子，空间大小相同的格子划分为一个等级。最终都会调用mallocgc方法，new(T)，&T{}，make(xxxx)

1. 堆上所有的对象内存分配都会通过`runtime.newobject`进行分配，运行时根据对象大小将它们分为微对象、小对象和大对象：
	- **tiny微对象（0, 16B）**：先使用微型分配器，再依次尝试线程缓存、中心缓存和堆分配内存；多个小于16B的无指针微对象的内存分配请求，会合并向Tiny微对象空间申请，微对象的 16B 内存空间从 spanClass 为 4 或 5（无GC扫描）的mspan中获取。
	- **small小对象[16B, 32KB]**：先向mcache申请，mcache内存空间不够时，向mcentral申请，mcentral不够，则向页堆mheap申请，再不够就向操作系统申请。
	- **large大对象(32KB, +∞)**：大对象直接向页堆mheap申请。
2. 对于内存的释放，遵循逐级释放的策略。当ThreadCache的缓存充足或者过多时，则会将内存退还给CentralCache。当CentralCache内存过多或者充足，则将低命中内存块退还PageHeap。

### mallocgc

1. 对于微对象的分配流程：
	（1）从 P 专属 mcache 的 tiny 分配器取内存（无锁）
	（2）nextFreeFast 根据所属的 spanClass，从 P 专属 mcache 缓存的 mspan 中取内存（无锁）
	（3） 根据所属的 spanClass 从对应的 mcentral 中取 mspan 填充到 mcache，然后从 mspan 中取内存（spanClass 粒度锁）
	（4）根据所属的 spanClass，从 mheap 的页分配器 pageAlloc 取得足够数量空闲页组装成 mspan 填充到 mcache，然后从 mspan 中取内存（全局锁）
	（5）mheap 向操作系统申请内存，更新页分配器的索引信息，然后重复（4）.
1. 对于小对象的分配流程是跳过（1）步，执行上述流程的（2）-（5）步；
2. 对于大对象的分配流程是跳过（1）-（3）步，执行上述流程的（4）-（5）步.
```go
func mallocgc(size uintptr, typ *_type, needzero bool) unsafe.Pointer {
	// ...
	assistG := deductAssistCredit(size)
	// 获取m
	mp := acquirem()
	// ... 获取mcache
	c := getMCache(mp)
	// ... mspan
	var span *mspan
	var x unsafe.Pointer
	// 是否为小对象 32kb
	if size <= maxSmallSize {
		// 小于 16 B 且无指针，则视为tiny对象
		if noscan && size < maxTinySize {
			// 1. 分配tiny对象如下
		} else {
			// 2. 分配小对象
		}
	} else {
		// 3. 分配大对象
	}
}
```

### tiny对象分配

```go
    noscan := typ == nil || typ.ptrdata == 0    
    // ...        
    if noscan && size < maxTinySize {        
      // tiny 内存块中，从 offset 往后有空闲位置          
      off := c.tinyoffset          
      // ...   调整off参数并对齐
      // 如果当前 tiny 内存块空间还够用，则直接分配并返回
      if off+size <= maxTinySize && c.tiny != 0 {
	      // 分配空间
	      x = unsafe.Pointer(c.tiny + off)
        c.tinyoffset = off + size
        c.tinyAllocs++
        mp.mallocing = 0
        releasem(mp)
        return x
	  }
	  // 分配一个新的tiny内存块
		span = c.alloc[tinySpanClass]
		// 从 mcache 的 span 中尝试获取空间   
		v := nextFreeFast(span)  
		if v == 0 {  
			// 通过mcentral,mheap兜底
			// 同样是获取mcache中的缓存,但是更加耗时
			// 如果mcache中没获取到则获取mcentral中的mspan用于分配(调用refill方法)
			// 如果mcentral也没有则去找mheap.
			// 这里的tinySpanClass,是序号为2的spanClass,即大小为16字节.同时也等于macTinySize
		   v, span, shouldhelpgc = c.nextFree(tinySpanClass)  
		}  
		// ...  
		size = maxTinySize
    // ...
    }
```

#### newFreeFast

```go
func nextFreeFast(s *mspan) gclinkptr {  
   theBit := sys.TrailingZeros64(s.allocCache) // 在 bit map 上寻找到首个 object 空位
   if theBit < 64 {  
      result := s.freeindex + uintptr(theBit)  
      if result < s.nelems {  
         freeidx := result + 1  
         if freeidx%64 == 0 && freeidx != s.nelems {  
            return 0  
         }  
         s.allocCache >>= uint(theBit + 1)  
         // 偏移
         s.freeindex = freeidx  
         s.allocCount++  
         // 返回获取object 空位的内存地址
         return gclinkptr(result*s.elemsize + s.base())  
      }  
   }  
   return 0  
}
```

#### nextFree
```go
func (c *mcache) nextFree(spc spanClass) (v gclinkptr, s *mspan, shouldhelpgc bool) {
	s = c.alloc[spc]
	// ...    
	// 从 mcache 的 span 中获取 object 空位的偏移量    
	freeIndex := s.nextFreeIndex()
	if freeIndex == s.nelems {
		// ...        
		// 倘若 mcache 中 span 已经没有空位，则调用 refill 方法从 mcentral 或者 mheap 中获取新的 span            
		c.refill(spc)
		// ...        
		// 再次从替换后的 span 中获取 object 空位的偏移量        
		s = c.alloc[spc]
		freeIndex = s.nextFreeIndex()
	}
	// ...    
	v = gclinkptr(freeIndex*s.elemsize + s.base())
	s.allocCount++
	// ...    
	return
}
```
### small对象分配
1. 根据对象大小，向上计算所需最小spanClass
2. 首先从p的mcache中取对应spanClass的span链表，如果有空闲的内存单元，则返回(nextFreeFast)
3. 如果没有，则向mcentral申请，如果还没有则向mheap申请。(nextFreeFast)
4. 最后清理空闲内存
```go
	// 获取spanclass信息
	var sizeclass uint8  
	if size <= smallSizeMax-8 {  
	   sizeclass = size_to_class8[divRoundUp(size, smallSizeDiv)]  
	} else {  
	   sizeclass = size_to_class128[divRoundUp(size-smallSizeMax, largeSizeDiv)]  
	}  
	// 根据对应的spanclass，分配给每个对象的空间大小
	size = uintptr(class_to_size[sizeclass])  
	spc := makeSpanClass(sizeclass, noscan)  
	span = c.alloc[spc]  
	v := nextFreeFast(span)  
	if v == 0 {  
	   v, span, shouldhelpgc = c.c(spc)  
	}  
	x = unsafe.Pointer(v)  
	if needzero && span.needzero != 0 {  
	   memclrNoHeapPointers(x, size)  
	}
```
### large对象分配
```go
	shouldhelpgc = true  
	// 直接调用mheap进行分配. makeSpanClass(0, noscan)
	span = c.allocLarge(size, noscan)  
	span.freeindex = 1  
	span.allocCount = 1  
	size = span.elemsize  
	x = unsafe.Pointer(span.base())
```


## 参考
1. https://blog.csdn.net/qq_25490573/article/details/130027162
2. https://cloud.tencent.com/developer/article/2077196
3. http://www.guoxiaolong.cn/blog/?id=12004
4. https://blog.haohtml.com/archives/29385#%E5%8F%82%E8%80%83%E8%B5%84%E6%96%99
5. https://mp.weixin.qq.com/s/2TBwpQT5-zU4Gy7-i0LZmQ