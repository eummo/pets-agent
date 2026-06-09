# 优化待办清单

更新日期：2026-06-06

本清单只保留尚未落地或仍需拍板的优化项。已完成内容沉淀到 `docs/changelog.md`，本文件不再重复。

## 待拍板

### Cron 多副本生产化路线

位置：`docs/cron-production-plan.md`

现状：cron job/run state 默认已进入 SQLite，文件 store 仅作为显式轻量模式；调度触发仍依赖进程内 `setInterval` 和本地文件租约。生产化路线已拆成外部调度器、Redis/Postgres leader election、队列化执行三种方案。

待拍板：

- 2026 H2 是否需要跨主机多副本运行 cron。
- 若需要，多副本优先走外部调度平台、Redis/Postgres leader election，还是队列化执行。
- 是否把 `cron.leader.*` / `cron.tick.error` / `cron.job.failed` / `cron.delivery.failed` 接入统一告警。
