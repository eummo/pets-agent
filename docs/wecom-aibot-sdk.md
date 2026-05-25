# 企业微信智能机器人接入文档

## 接入模式：WebSocket 长连接

本项目使用企业微信智能机器人的 **WebSocket 长连接模式**（非 Webhook 短连接），通过官方 SDK `@wecom/aibot-node-sdk` 接入。

### 长连接 vs Webhook 对比

| 特性 | Webhook（短连接） | WebSocket（长连接） |
|------|-------------------|---------------------|
| 连接方式 | 每次回调建立新连接 | 复用已建立的长连接 |
| 延迟 | 较高 | 低 |
| 服务端要求 | 需要公网可访问的 URL | 无需固定公网 IP |
| 加解密 | 需要对消息加解密 | 无需加解密 |
| 流式回复 | 通过 response_url 回调 | 主动通过 WebSocket 推送 |
| 复杂度 | 低 | 较高（心跳保活、断线重连） |

长连接模式的优势：无需公网 IP、无需 AES 加解密、天然支持流式回复。

## 官方 SDK

```bash
npm install @wecom/aibot-node-sdk
```

SDK 仓库：https://github.com/WecomTeam/aibot-node-sdk

### 核心能力

- WebSocket 长连接 + 自动认证 + 心跳保活 + 断线重连
- 消息收发（text/image/mixed/voice/file/video）
- 流式回复（支持 Markdown）
- 模板卡片消息
- 主动推送消息
- 事件回调（enter_chat / template_card_event / feedback_event）
- 文件下载解密（AES-256-CBC，每个文件独立 aeskey）
- 媒体素材分片上传
- 完整 TypeScript 类型声明

### 配置

| 参数 | 说明 |
|------|------|
| `botId` | 智能机器人 BotID（企业微信后台获取） |
| `secret` | 长连接专用密钥（企业微信后台获取） |

> 注意：长连接模式的 `botId` + `secret` 与 Webhook 模式的 `token` + `encodingAesKey` 不同，切换模式会导致原配置失效。

### 快速开始

```ts
import { WSClient, generateReqId } from '@wecom/aibot-node-sdk';
import type { WsFrame } from '@wecom/aibot-node-sdk';

const wsClient = new WSClient({
  botId: 'your-bot-id',
  secret: 'your-bot-secret',
});

wsClient.connect();

wsClient.on('authenticated', () => {
  console.log('认证成功');
});

wsClient.on('message.text', (frame: WsFrame) => {
  const content = frame.body.text?.content;
  const streamId = generateReqId('stream');
  wsClient.replyStream(frame, streamId, '正在处理...', false);
  // ... 后续流式更新
  wsClient.replyStream(frame, streamId, '处理完成！', true);
});
```

## 消息帧结构

### WsFrame<T>

```ts
interface WsFrame<T = any> {
  cmd?: string;
  headers: { req_id: string; [key: string]: any };
  body?: T;
  errcode?: number;
  errmsg?: string;
}
```

### BaseMessage

```ts
interface BaseMessage {
  msgid: string;
  aibotid: string;
  chatid?: string;              // 群聊 ID（群聊时返回）
  chattype: 'single' | 'group'; // 会话类型
  from: { userid: string };     // 发送者
  create_time?: number;
  response_url?: string;
  msgtype: string;
  quote?: QuoteContent;
}
```

## 关键 API

### WSClient 方法

| 方法 | 说明 |
|------|------|
| `connect()` | 建立 WebSocket 连接 |
| `disconnect()` | 断开连接 |
| `replyStream(frame, streamId, content, finish?)` | 流式文本回复 |
| `replyWelcome(frame, body)` | 回复欢迎语（5s 内） |
| `replyTemplateCard(frame, templateCard)` | 回复模板卡片 |
| `replyStreamWithCard(frame, streamId, content, finish?, options?)` | 流式+卡片组合 |
| `updateTemplateCard(frame, templateCard)` | 更新模板卡片（5s 内） |
| `sendMessage(chatid, body)` | 主动推送消息 |
| `uploadMedia(fileBuffer, options)` | 上传临时素材 |
| `replyMedia(frame, mediaType, mediaId)` | 被动回复媒体消息 |
| `sendMediaMessage(chatid, mediaType, mediaId)` | 主动发送媒体消息 |
| `downloadFile(url, aesKey)` | 下载文件并 AES 解密 |

### 事件列表

| 事件 | 说明 |
|------|------|
| `connected` | 连接建立 |
| `authenticated` | 认证成功 |
| `disconnected` | 连接断开 |
| `reconnecting` | 正在重连 |
| `error` | 发生错误 |
| `message.text` | 收到文本消息 |
| `message.image` | 收到图片消息 |
| `message.mixed` | 收到图文混排消息 |
| `message.voice` | 收到语音消息 |
| `message.file` | 收到文件消息 |
| `message.video` | 收到视频消息 |
| `event.enter_chat` | 进入会话事件 |
| `event.template_card_event` | 模板卡片按钮事件 |
| `event.feedback_event` | 用户反馈事件 |

## 对话隔离

每个消息帧中包含 `chattype`（single/group）和 `chatid`（仅群聊），用于对话隔离：

| 场景 | chattype | chatid | ConversationSessionKey |
|------|----------|--------|----------------------|
| 用户在群聊 X | group | groupX | { channel: "wechat-work", userId: "userA", chatId: "groupX", workspacePath } |
| 用户在群聊 Y | group | groupY | { channel: "wechat-work", userId: "userA", chatId: "groupY", workspacePath } |
| 用户单聊 | single | undefined | { channel: "wechat-work", userId: "userA", workspacePath } |

同一用户在不同群聊中拥有独立的对话上下文。

## 频率限制

- 回复 + 主动推送：30 条/分钟，1000 条/小时（每个会话）
- 流式消息超时：从首次发送开始 10 分钟内必须 finish
- 欢迎语/卡片更新：5 秒内回复
- 同一用户同一机器人最多 3 条消息同时交互中

## 流式回复机制

```
1. 用户 @机器人发消息
2. 收到 aibot_msg_callback (req_id=xxx)
3. 生成唯一的 stream.id = generateReqId('stream')
4. replyStream(frame, streamId, "正在查询...", false)  → 创建新流式消息
5. replyStream(frame, streamId, "查询中...", false)    → 刷新内容
6. replyStream(frame, streamId, "最终结果", true)      → 结束流式消息
```

针对同一次消息回调的所有流式回复都使用相同的 `req_id`。
