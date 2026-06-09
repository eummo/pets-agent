# Cron 生产化调度/队列方案

当前 cron 已完成：

- job/run state 默认写入 SQLite `cron_jobs` / `cron_run_state`。
- file store 仅作为 `cron.jobStore=file` 的本地轻量模式。
- 调度 tick 在进程内执行，使用 `cron.leaderLeasePath` 文件租约做同机 leader guard。
- `/cron/status` 返回 `running`、`leader`、job 数量。
- `system.jsonl` 记录 `cron.leader.acquired`、`cron.leader.lost`、`cron.leader.skipped`、`cron.tick.error`、job 执行和 delivery 事件。

这套实现适合单实例或同机轻量多进程。跨主机多副本部署前，需要把“谁触发 job”和“如何抢占 leader”迁到共享后端或外部调度器。

## 触发条件

满足任一条件时启动生产化迁移：

- 运维要求 `>=2` 个服务副本同时运行 cron。
- 单实例压测或实际运行显示 cron tick / delivery 影响主服务响应。
- 需要跨主机 leader election、执行审计、失败重试或延迟队列。
- cron job 数量或执行耗时导致 tick 经常出现 `cron.tick.skipped`。

## 推荐路线

### 方案 A：外部调度器 + 当前 HTTP 管理/触发接口

适用：job 数量少、执行频率低、运维已有 Kubernetes CronJob、GitHub Actions、云函数或企业内部调度平台。

做法：

- 保留 SQLite `cron_jobs` 作为配置源。
- 外部调度器按固定频率调用受控触发入口或内部 worker。
- 应用内 `setInterval` 可关闭或仅保留开发模式。
- 调度器平台负责多副本互斥、重试和告警。

优点：改造最小，运维边界清晰。

风险：需要额外调度平台；动态 job 的精确调度能力取决于平台。

### 方案 B：Postgres/Redis leader election + SQLite/Postgres job state

适用：服务自身需要管理动态 cron，且多副本部署由同一应用集群承接。

做法：

- 用 Redis `SET NX PX` 或 Postgres advisory lock 替代本地文件租约。
- job/run state 继续 SQLite 仅限单机；跨主机时迁到 Postgres 或 Redis-backed store。
- leader 获取、续租、失去 leader 必须写 system event，并接入运维告警。

优点：保留应用内动态调度体验。

风险：需要引入共享存储依赖；leader 续租、时钟漂移和网络分区要做演练。

### 方案 C：队列化执行

适用：job 执行耗时长、需要失败重试、并发限制、死信队列或跨 worker 扩容。

做法：

- cron tick 只负责 enqueue due jobs。
- worker 从队列消费并执行 MessageGateway / delivery。
- run state 记录 enqueue、started、finished、retry、dead-letter。
- 可选后端：Redis queue、Postgres queue、云消息队列。

优点：执行与调度解耦，弹性和审计更强。

风险：实现和运维成本最高；需要设计幂等与重复消费处理。

## 当前项目建议

短期保持当前实现，不引入 Redis/Postgres，条件是：

- 部署保持单实例，或同机多进程只作为开发/临时场景。
- 定时任务数量低，执行时长不会压垮主服务。
- `cron.leader.*` 和 `cron.tick.error` 已纳入 system 日志巡检。

进入 2026 H2 多副本路线图时，优先选择：

1. 有现成调度平台：走方案 A。
2. 没有外部调度平台，但已有 Redis/Postgres：走方案 B。
3. cron 执行需要重试/死信/扩容：走方案 C。

## 迁移步骤

1. 确认目标部署拓扑：单实例、同机多进程、跨主机多副本。
2. 选择共享后端：外部调度平台、Redis、Postgres 或队列。
3. 保留 `CronJobStore` / `CronScheduler` 契约，新增后端实现，不改 gateway 或 channel adapter。
4. 在 staging 同时跑当前 file leader 和新 leader 观测，但只允许一个触发执行。
5. 验证：
   - 同一 job 在多副本下只执行一次。
   - leader 崩溃后能在 TTL 内恢复。
   - 管理接口并发创建/更新 job 不丢写。
   - `nextRunAt`、`lastResult`、delivery 事件和 conversation log 正常。
   - `npm run check` 和 `npm run smoke` 全绿。
6. 切生产前备份 SQLite，并确认恢复演练可用：`docs/sqlite-backup-restore.md`。

## 验收标准

- `/cron/status` 能显示当前实例是否 leader，且外部监控能识别 leader 缺失。
- 共享后端能证明跨主机只触发一次 due job。
- `cron.leader.acquired`、`cron.leader.lost`、`cron.tick.error`、`cron.job.failed`、`cron.delivery.failed` 进入告警或日志巡检。
- 故障演练覆盖 leader 进程崩溃、共享后端短暂不可用、job 执行超时和 delivery 失败。
- JSONL 不记录 secrets、authorization headers、access tokens、refresh tokens。
