# pets-agent 风险与计划(2026-06-06)

`npm run check` 全绿:431 tests / 41 files / typecheck+lint+build+deterministic smoke ✓；`npm run smoke` 真实模型回归全绿。架构沿 docs/architecture.md 的契约/适配器分层,本评估只列**风险 + 阶段计划**,不重复优点。

---

## 风险(按 P0 → P2)

| P   | 项                                                          | 位置                                                     | 风险                                         |
| --- | ----------------------------------------------------------- | -------------------------------------------------------- | -------------------------------------------- |
| P1  | Cron job/run state 已进 SQLite,多副本生产化路线已拆分待拍板 | `cron/cronScheduler.ts` + `docs/cron-production-plan.md` | 跨主机部署前需选择共享调度/队列或共享 leader |

---

## 阶段计划

**Phase 4 · 多进程/规模化(按需)**

- 触发条件:单进程压测 200 并发 fail 或运维需要 ≥2 副本
- 按 `docs/cron-production-plan.md` 选择外部调度器、Redis/Postgres leader election 或队列化执行
- Fastify cluster + sticky session
- `/metrics` Prometheus 端点

## 待拍板

1. Phase 4 多进程是否在 2026 H2 路线图?不做则 Phase 1.1/1.2 可更轻。
2. Cron 是否在 2026 H2 继续迁到外部队列 / Redis / Postgres 共享调度后端?
