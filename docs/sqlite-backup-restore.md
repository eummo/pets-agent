# SQLite 备份与恢复演练

本项目默认把生产状态写入 `config/runtime.json` 的 `dbPath`，包括：

- `roles`
- `feedback`
- `conversation_sessions`
- `conversation_histories`
- `conversation_history_archives`
- `cron_jobs`
- `cron_run_state`

JSONL 日志仍作为审计日志保存，不作为业务状态恢复来源。显式启用的 file store 只适合本地轻量模式，应单独备份对应 JSON 文件。

完整数据生命周期见 `docs/persistence-lifecycle.md`。

## 备份

服务运行中可以执行 SQLite online backup，备份会先写到目标文件，再做 `PRAGMA integrity_check`：

```bash
npm run db:backup -- --db .harness/state/agent.db --out .harness/backups/agent.db
```

备份完成后可再次校验：

```bash
npm run db:verify -- --db .harness/backups/agent.db
```

## 恢复演练

恢复会覆盖目标数据库，必须显式传 `--force`。恢复前先停止服务，并保留当前库的二次备份。

```bash
npm run db:backup -- --db .harness/state/agent.db --out .harness/backups/pre-restore.db
npm run db:restore -- --backup .harness/backups/agent.db --db .harness/state/agent.db --force
npm run db:verify -- --db .harness/state/agent.db
```

恢复完成后启动服务，检查：

- `GET /readyz` 返回 SQLite ready。
- 启动横幅中的 `db=` 指向恢复后的库。
- `/cron/status?userId=<admin>` 能读取 cron job/run state。
- 最近 `.harness/logs/system.jsonl` 没有 SQLite 打开或迁移错误。

## 压缩与归档

conversation history archives 已有保留期清理任务，默认保留 180 天。长期运行后，重型归档导出或压缩应作为离线任务执行，不放进主服务请求路径。备份窗口内如需减小文件体积，可先停止服务，再用 SQLite 运维工具执行 VACUUM；执行前后都要保留备份并运行 `npm run db:verify`。
