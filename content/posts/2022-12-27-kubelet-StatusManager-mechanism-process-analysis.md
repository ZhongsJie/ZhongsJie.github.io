---
title: "Kubelet StatusManager机制流程分析"
date: 2022-12-27T22:37:43+08:00
draft: false
subtitle: ""
author: "ZhongsJie"
tags:
  - Kubernetes
  - Kubelet
categories:
  - Kubernetes
showToc: true # 显示目录
TocOpen: false # 自动展开目录
disableShare: true # 底部不显示分享栏
cover:
    image: "/images/2022-12/image-StatusManager.png"
    caption: ""
    alt: ""
    relative: true
---

## Kubelet StatusManager机制流程分析

> 主要功能将Pod状态信息同步到ApiServer，并不会主动监控Pod状态，提供接口供其他Manager调用，当其他组件需要改变 pod 的状态时会将 pod 的 status 信息发送到 statusManager 进行同步。主要使用方`probeManager`，`podWorkers`

![StatusManager](/images/2022-12/image-StatusManager.png)

```go
type manager struct {  
   kubeClient clientset.Interface  
   podManager kubepod.Manager  
   // Map from pod UID to sync status of the corresponding pod.  
   // statusManager 的 cache，保存 pod 与状态的对应关系；
   podStatuses      map[types.UID]versionedPodStatus  
   podStatusesLock  sync.RWMutex  
   // 当其他组件调用 statusManager 更新 pod 状态时，会将 pod 的状态信息发送到podStatusesChannel 中；
   podStatusChannel chan podStatusSyncRequest  
   // Map from (mirror) pod UID to latest status version successfully sent to the API server.  
   // apiStatusVersions must only be accessed from the sync thread.  
   apiStatusVersions map[kubetypes.MirrorPodUID]uint64  
   // 删除 pod 的接口
   podDeletionSafety PodDeletionSafetyProvider  
  
   podStartupLatencyHelper PodStartupLatencyStateHelper  
}
```

## Start
```go
// 设置定时触发，时间为10s
syncTicker := time.NewTicker(syncPeriod).C

go wait.Forever(func() {  
   for {  
      select {  
      // 监听到一个pod状态变更的场景
      case syncRequest := <-m.podStatusChannel:  
         klog.V(5).InfoS("Status Manager: syncing pod with status from podStatusChannel",  
            "podUID", syncRequest.podUID,  
            "statusVersion", syncRequest.status.version,  
            "status", syncRequest.status.status)  
         m.syncPod(syncRequest.podUID, syncRequest.status) 
	         -> func (m *manager) needsUpdate(uid types.UID, status versionedPodStatus) bool {
		         // 1. 判断版本号信息
		         // 2. 获取pod
		         // 3. 判断是否处于删除状态
			         -> PodResourcesAreReclaimed // 检查 pod 在 node 上占用的所有资源是否已经被回收
	         }
      // 触发定时器（定时器，syncPeriod 默认为 10s）
      case <-syncTicker:  
         klog.V(5).InfoS("Status Manager: syncing batch")  
         // remove any entries in the status channel since the batch will handle them  
         for i := len(m.podStatusChannel); i > 0; i-- {  
            <-m.podStatusChannel  
         }  
         m.syncBatch()  
      }  
   }  
}, 0)
```


### syncPod

1. 基本流程
	- 判断是否需要同步状态， 判断版本号信息是否已经增加，若不需要同步则继续检查 pod 是否处于删除状态
	- 合并状态信息并更新记录到cache
	- 如果可以删除pod，执行删除动作
```go
func (m *manager) syncPod(uid types.UID, status versionedPodStatus) {
	// 判断是否需要同步状态，以及是否处于删除状态
	if !m.needsUpdate(uid, status) {
		...
	}
	// 判断ResolvedPodUID是否一致，不一致则为删除后重建出来的pod，需要删除statusmanager保存的旧状态信息
	if len(translatedUID) > 0 && translatedUID != kubetypes.ResolvedPodUID(uid) {
		// 删除保存的状态信息，以及启动延时处理的状态  
	   m.deletePodStatus(uid)  
	   return  
	}
	// 根据实际运行状态以及其他组件设置的状态合并出最终状态信息
	mergedStatus := mergePodStatus(pod.Status, status.status, m.podDeletionSafety.PodCouldHaveRunningContainers(pod))
	// 更新pod状态信息以及所记录状态
	
	// 如果pod处理删除状态，删除pod以及记录信息
	if m.canBeDeleted(pod, status.status) {
		// 设置GracePeriodSecond为0，同时避免删除一个同名同命名空间的资源，传递Uid做precondition
		...
		err = m.kubeClient.CoreV1().Pods(pod.Namespace).Delete(context.TODO(), pod.Name, deleteOptions)
		...
		m.deletePodStatus(uid)
	}
}

```
### syncBatch
> 定期将statusManager 缓存 podStatuses 中的数据同步到 apiserver，定时同步的时候会清理channel内容
1. 基本流程
	- 获取podManager中数据 `m.podManager.GetUIDTranslations()`
	- 清理孤立的版本信息
	- 首先调用 needsUpdate 检查 pod 的状态是否与 apiStatusVersions 中的一致，然后调用 needsReconcile 检查 pod 的状态是否与 podManager 中的一致，若不一致则将需要同步的 pod 加入到 updatedStatuses 列表中
	- 调用syncPod同步状态
```go
// 所有面向公共函数都需要使用ResolvedPodUID
podToMirror, mirrorToPod := m.podManager.GetUIDTranslations()
func() {
	for uid, status := range m.podStatuses {
		syncedUID := kubetypes.MirrorPodUID(uid)
		...
		if m.needsUpdate(types.UID(syncedUID), status) {  
		   updatedStatuses = append(updatedStatuses, podStatusSyncRequest{uid, status})  
		} else if m.needsReconcile(uid, status.status) {  
			// 删除状态，强制更新一次状态
		   delete(m.apiStatusVersions, syncedUID)  
		   updatedStatuses = append(updatedStatuses, podStatusSyncRequest{uid, status})  
		}
	}
}
```

#### needsUpdate

> 判断状态是否已经过时，非线程安全类

1. 版本信息`latest < status.version`
2. 是否可以被删除`m.canBeDeleted(pod, status.status)`
	
#### needsReconcile
>kubelet的状态应该是pod的真实状态，如果apiserver（从podManager中获取的）和kubelet状态(statusManager)不一致的，需要发送一个update信号来强制reconcile
1. 基本流程
	- 通过 uid 从 podManager 中获取 pod 对象；`m.podManager.GetPodByUID(uid)`
	- 检查 pod 是否为 static pod，若为 static pod 则获取其对应的 mirrorPod；`kubetypes.IsStaticPod(pod)`
	- 格式化 pod status subResource；`normalizeStatus(pod, podStatus)`
	- 检查 podManager 中的 status 与 statusManager cache 中的 status 是否一致，若不一致则以 statusManager 为准进行同步；`isPodStatusByKubeletEqual(podStatus, &status)`


## SetPodStatus
>设置 status subResource 并会触发同步操作, SetPodStatus 方法主要会用在 kubelet 的主 `syncLoop` 中，并在`syncPod` 方法中创建 pod 时使用

```go
m.updateStatusInternal(pod, status, pod.DeletionTimestamp != nil)
```

## SetContainerReadiness
> 使用给定的readiness状态更新缓存的容器状态，主要被用在 `proberManager`

```go
// 更新pod
updateConditionFunc(v1.PodReady, GeneratePodReadyCondition(&pod.Spec, status.Conditions, status.ContainerStatuses, status.Phase))  
// 更新容器
updateConditionFunc(v1.ContainersReady, GenerateContainersReadyCondition(&pod.Spec, status.ContainerStatuses, status.Phase))  
// 更新内部状态
m.updateStatusInternal(pod, status, false)
```

## SetContainerStartup
> 主要被用在 `proberManager`
```go
// startup探针状态
containerStatus, _, _ = findContainerStatus(&status, containerID.String())  
containerStatus.Started = &started  
  
m.updateStatusInternal(pod, status, false)
```

## TerminatePod
> 主要会用在 kubelet 的主 `syncLoop` 
1. 一旦pod被初始化，任何状态的丢失都被视为是失败
2. `status.containerStatuses`和 `status.initContainerStatuses` 中 container 的 state 置为Terminated 状态
3. 调用`m.updateStatusInternal(pod, status, true)`


## updateStatusInternal（核心逻辑）
> 触发状态更新的核心逻辑，`updateStatusInternal` 会更新 `statusManager` 的 `cache` 并会将 newStatus 发送到`m.podStatusChannel` 中，然后 `statusManager` 会调用 `syncPod` 方法同步到 apiserver。

```go
func (m *manager) updateStatusInternal(pod *v1.Pod, status v1.PodStatus, forceUpdate bool) bool {
	// 获取旧状态信息（优先从缓存中获取）
	var oldStatus v1.PodStatus
	...
	// 检查container以及initContainer状态信息是否合法（禁止状态从终止态到非终止态）
	err := checkContainerStateTransition(oldStatus.ContainerStatuses, status.ContainerStatuses, pod.Spec.RestartPolicy)
	...
	// 为 status 更新ContainersReady、PodReady、PodInitialized、PodHasNetwork、PodScheduled对应conditions时间
	updateLastTransitionTime(&status, &oldStatus, v1.ContainersReady)
	// 更新status开始时间并格式化
	...
	// 添加到缓存中
	m.podStatuses[pod.UID] = newStatus  
	select {  
	// 发送同步请求
	case m.podStatusChannel <- podStatusSyncRequest{pod.UID, newStatus}:  
		// 触发一次更新
	   return true  
	default:  
	   // 当channel满了之后跳过当前同步请求
	   return false  
	}
	
}
```

