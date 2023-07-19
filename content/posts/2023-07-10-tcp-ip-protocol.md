---
title: "TCP/IP 协议的那些东西"
date: 2023-07-12T22:12:14+08:00
draft: false
subtitle: ""
author: "ZhongsJie"
tags:
  - TCP/IP
categories:
  - Network
showToc: true # 显示目录
TocOpen: true # 自动展开目录
disableShare: false # 底部不显示分享栏
cover:
    image: ""
    caption: ""
    alt: ""
    relative: false
---

## TCP/IP 协议的那些东西


> 本文主要是基于《TCP/IP 详解 卷1：协议》以及一些资料的一个学习总结。

### 概述

> 网络中的整体传输流程可以简要总结为：数据首先会封装到TCP的Segment中，然后TCP的Segment封装到IP的Packet中，最后封装为以太网Ethernet的Frame，各个层解析自己的协议以及数据信息，最后将数据交给更高层的协议处理

![OSI Model](/images/2023-07/image-OSI-model.png)

-  **应用层** ：为特定应用程序提供数据传输服务，例如 HTTP、DNS 等协议。数据单位为报文。
-  **传输层** ：为进程提供通用数据传输服务。由于应用层协议很多，定义通用的传输层协议就可以支持不断增多的应用层协议。传输层包括两种协议：
	- **传输控制协议 TCP**，提供面向**连接、可靠**的数据传输服务，数据单位为报文段；
	- **用户数据报协议 UDP**，提供无连接、尽最大努力的数据传输服务，数据单位为用户数据报。TCP 主要提供完整性服务，UDP 主要**提供及时性服务**。
-  **网络层** ：为主机提供数据传输服务。而传输层协议是为主机中的进程提供数据传输服务。网络层把传输层传递下来的报文段或者用户数据报封装成分组，**IP协议**
-  **数据链路层** ：网络层针对的还是主机之间的数据传输服务，而主机之间可以有很多链路，链路层协议就是为同一链路的主机提供数据传输服务。数据链路层把网络层传下来的分组**封装成帧**。
-  **物理层** ：考虑的是怎样在传输媒体上传输数据比特流，而不是指具体的传输媒体。物理层的作用是尽可能屏蔽传输媒体和通信手段的差异，使数据链路层感觉不到这些差异。

---
### TCP头格式

![TCP Header](/images/2023-07/image-TCP-Header.jpg)

1. TCP的包没有IP地址, 只有源端口和目标端口, 加上IP首部中的源端IP地址和目的端IP地址唯一确定一个TCP连接(**四元组**,加上协议是五元组)
    - 一组IP地址和端口成为Socket
    ![五元组](/images/2023-07/image-socket.png)
2. Sequence Number是包的序号，报文段中的的第一个数据字节，用来解决网络包乱序问题。
2. Acknowledgement Number就是ACK——用于确认收到，用来解决不丢包的问题。
3. Window又叫Advertised-Window，也就是著名的滑动窗口（Sliding Window），用于解决流量控制
4. 在TCP首部中有6个标志比特（TCP Flag）。它们中的多个可同时被设置为 1
    - URG 紧急指针（urgent pointer）有效
    - ACK 确认序号有效。
    - PSH 接收方应该尽快将这个报文段交给应用层
    - RST 重建连接。
    - SYN 同步序号用来发起一个连接。
    - FIN 发端完成发送任务。

---
### TCP的连接和终止

> TCP为应用层提供全双工服务, 这意味数据能在两个方向上独立地进行传输

![三次握手&四次挥手](/images/2023-07/image-tcp-open-close.jpg)


1. 三次握手
    - 要初始化Sequence Number 的初始值。通信的双方要互相通知对方自己的初始化的Sequence Number（缩写为ISN：Inital Sequence Number）——所以叫SYN，全称Synchronize Sequence Numbers。这个号要作为以后的数据通信的序号，以保证应用层接收到的数据不会因为网络上的传输的问题而乱序（TCP会用这个序号来拼接数据）。
    - 建立连接需要解决的三个问题
      - 要使每一方能够确知对方的存在
      - 允许双方协商一些参数
      - 能够对运输实体资源进行分配
    - 为什么要握手三次
      - 建立连接的时候，不允许出现半打开状态（半建立状态）下传送消息（TCP的规范）
      - 防止**已失效**的连接请求报文段突然又传送到服务器端（客户端已失效的连接到服务端认为是重新连接从而造成服务端的资源浪费）

2. 四次挥手
    - TCP是全双工的，发送方和接收方都需要Fin和Ack。如果两边同时断连接，那就会就进入到CLOSING状态，然后到达TIME_WAIT状态。
    - 服务器端进入**ClOSE-WAIT（关闭等待）**状态。TCP服务器进程这时应通知高层应用进程，这时的TCP连接处于**半关闭状态**。服务器端若发送数据，客户端仍要接收，这个状态会持续一段时间。
    - RST包用于强制关闭TCP链接

3. 建立连接的超时
    - 连接建立过程中，任一端出现接收不到对端ACK的情况时，都将导致本端重传初始SYN segment。 在Linux下，**默认重试次数为5次**，重试的间隔时间从1s开始每次都翻售，5次的重试时间间隔为1s, 2s, 4s, 8s, 16s，总共31s，第五次32s后知道超时。
    - 重传间隔：
      - 若当前为第N次重传，则与上一次发送或重传的间隔时间为2的(N -1)次方秒，但最长间隔不超过120秒；内核的TCP_TIMEOUT_INIT/TCP_RTO_MIN/TCP_RTO_MAX宏分别定义了间隔时间的初始值/最小值/最大值
    - 重传次数：
      - 客户端由tcp_syn_retries指定，对应的proc文件为`/proc/sys/net/ipv4/tcp_syn_retries`，sysctl参数为`net.ipv4.tcp_syn_retries`；
      - 服务端由tcp_synack_retries指定，对应的proc文件为`/proc/sys/net/ipv4/tcp_synack_retries`，sysctl参数为`net.ipv4.tcp_synack_retries`
    - 修改TCP内核参数： `sudo vim /etc/sysctl.conf`
  
4. MSL和TIME_WAIT(RFC793定义MSL为2分钟)
    - MSL(最长报文段寿命)指明TCP报文在Internet上最长生存时间
    - 从TIME_WAIT状态到CLOSED状态，有一个**超时设置**，这个超时设置是 2*MSL
      - 为了让本连接持续时间内所产生的所有报文都从网络中消失，使得下一个连接不会出现旧的连接请求报文
      - TIME_WAIT确保有足够的时间让对端收到了ACK,如果server端没收到确认报文，那么就会重传

---
### TCP超时重传

![超时重传](/images/2023-07/image-tcp-retransimit.png)
1. TCP管理几个定时器：
    - 重传定时器使用于当希望收到另一端的确认
    - 坚持(persist)定时器使窗口大小信息保持不断流动
    - 保活(keepalive)定时器可检测到一个空闲连接的另一端何时崩溃或重启
    - 2MSL定时器测量一个连接处于TIME_WAIT状态的时间

2. 超时时间设置（时间驱动, 根据不同算法计算出RTT时间）
    - RTT：Round-Trip Time 往返时延，数据从网络一端传送到另一端所需的时间，也就是包的往返时间。
    - RTO：Retransmission Timeout 超时重传时间。
    - 超时重传时间 RTO 的值需略大于报文往返 RTT 的值

3. 快速重传（Fast Retransmit）机制，不以时间为驱动，而是以数据驱动重传
    - 快速重传的工作方式是当收到三个相同的 ACK 报文时，会在定时器过期之前，重传丢失的报文段。
    - 对此有两种选择：
      - 一种是仅重传timeout的包（SACK方法，TCP头部「选项」字段里加一个SACK的东西，它可以将缓存的信息发送给发送方。在Linux下，可以通过tcp_sack参数打开这个功能）
      - 另一种是重传timeout后所有的数据

---
### TCP流量控制
> TCP头里有一个字段叫Window，又叫Advertised-Window，这个字段是接收端告诉发送端自己还有多少缓冲区可以接收数据。于是发送端就可以根据这个接收端的处理能力来发送数据，而不会导致接收端处理不过来

![tcp-cache](/images/2023-07/image-tcp-cache.png)

1. 利用滑动窗口实现流量控制，就是让发送方的发送速率不要太快，要让接收方来得及接收。
2. 发送方的发送窗口不能超过接收方给出的接收窗口的数值
3. 流量控制往往指点对点通信量的控制，是个端对端的问题
4. zero window: TCP使用了Zero Window Probe技术，缩写为ZWP，也就是说，发送端在窗口变成0后，会发ZWP的包给接收方，让接收方来ack他的Window大小


### TCP拥塞控制

> 所谓拥塞控制就是防止过多的数据注入到网络中，这样可以使网络中的路由器或链路不致过载, 当拥塞发生的时候，要做自我牺牲

拥塞控制主要是四个算法：1）慢开始，2）拥塞避免，3）拥塞发生，4）快恢复

1. 发送方维护一个拥塞窗口cwnd的变量，值取决于网络的拥塞程度，并且值是动态的
    - 拥塞窗口的原则是，只要网络中没有拥塞，拥塞窗口就扩大一些。当出现网络拥塞的时候，窗口会减小
    - 判断是否出现拥塞的原则，是否接收到确认信息（是否发生重传）
2. 发送方将拥塞窗口作为发送窗口：swnd = cwnd
3. 维护一个慢开始门限ssthread的状态变量
    - cwnd < ssthread: 慢开始
    - cwnd > ssthread: 拥塞避免
    - cwnd = ssthread: 慢开始或拥塞避免
4. 发送方收到3个重复确认，知道只是丢失了部分字段，不会进入慢开始，而是执行快恢复
    

![congestion-controll](/images/2023-07/image-tcp-congestion-control.png)
**慢开始**

1. 刚加入网络的时候一点点提速，直到到达ssthread慢开始门限
2. 连接建好的开始先初始化cwnd = 1，表明可以传一个MSS大小的数据。
3. 每当收到一个ACK，cwnd++; 呈线性上升
4. 每当过了一个RTT，cwnd = cwnd*2; 呈指数让升

**拥塞避免**

1. 收到一个ACK时，cwnd = cwnd + 1/cwnd
2. 当每过一个RTT时，cwnd = cwnd + 1

**拥塞状态**
1. 等到RTO超时，重传数据包。TCP认为这种情况太糟糕，反应也很强烈。
    - sshthresh =  cwnd /2
    - cwnd 重置为 1
    - 进入慢启动过程
2. Fast Retransmit算法，也就是在收到3个duplicate ACK时就开启重传，而不用等到RTO超时，随后进入到快恢复


**快恢复**
1. 发送方将慢开始门限ssthread和cwnd窗口都设置为当前的一半，并且开始执行拥塞避免算法

