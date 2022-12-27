---
title: "Nginx指南"
date: 2022-12-09T17:36:50+08:00
tags: ["Development", "Nginx"]
author: ZhongsJie
draft: false
---
 
## **Nginx** 介绍
> “Nginx 是一款轻量级的 HTTP 服务器，采用事件驱动的异步非阻塞处理方式框架，这让其具有极好的 IO 性能，时常用于服务端的**反向代理**和**负载均衡**。”

![Nginx](/images/2022-12/image-nginx.png)

1. 正向代理
    - 正向代理“代理”的是客户端，而且客户端是知道目标的，而目标是不知道客户端是通过VPN访问的。
2. 反向代理
    - 反向代理“代理”的是服务器端，而且这一个过程对于客户端而言是透明的。


### **Nginx的Master-Worker模式**

1.  Master
    - 读取并配置nginx.conf, 管理Worker进程
2.  Worker
    - 维护一个线程（避免线程切换），处理连接和请求；注意Worker进程的个数由配置文件决定，一般和CPU个数相关（有利于进程切换），配置几个就有几个Worker进程。

### **Nginx热部署**

1.  修改配置文件nginx.conf后，重新生成新的worker进程，当然会以新的配置进行处理请求，而且新的请求必须都交给新的worker进程，至于老的worker进程，等把那些以前的请求处理完毕后，kill掉

### **Keepalived+Nginx实现高可用**

1.  Keepalived是一个高可用解决方案，主要是用来防止服务器单点发生故障
2.  keepalived是以VRRP协议为实现基础的，VRRP全称Virtual Router Redundancy Protocol，即虚拟路由冗余协议。
3.  keepalived主要有三个模块，分别是core、check和vrrp。core模块为keepalived的核心，负责主进程的启动、维护以及全局配置文件的加载和解析。check负责健康检查，包括常见的各种检查方式。vrrp模块是来实现VRRP协议的。


