#!/usr/bin/env python3
"""pets-agent TUI 交互测试框架"""
import subprocess, os, pty, select, time, sys

SLEEP = time.sleep

class TUITest:
    def __init__(self, cmd=['npx', 'tsx', 'src/index.ts']):
        self.cmd = cmd
        self.proc = None
        self.master = None
        self.output = ''

    def start(self):
        self.master, slave = pty.openpty()
        self.proc = subprocess.Popen(
            self.cmd, stdin=slave, stdout=slave, stderr=slave,
            start_new_session=True
        )
        os.close(slave)
        SLEEP(0.5)

    def write(self, text, delay=0.3):
        os.write(self.master, text.encode('utf-8'))
        SLEEP(delay)

    def read(self, timeout=2.0):
        data = b''
        while True:
            r, _, _ = select.select([self.master], [], [], timeout)
            if not r: break
            try:
                d = os.read(self.master, 4096)
                if not d: break
                data += d
            except OSError: break
        text = data.decode('utf-8', errors='replace')
        self.output += text
        return text

    def expect(self, text, timeout=5.0):
        """发送文本后等待响应"""
        self.write(text + '\n', delay=0.5)
        out = self.read(timeout)
        return out

    def destroy(self):
        os.close(self.master)
        try: self.proc.terminate()
        except: pass

    def run(self):
        print(f"[TEST] Starting: {' '.join(self.cmd)}")
        self.start()

        # === 基础命令测试 ===
        print("\n=== 基础命令 ===")

        # 1. 启动
        out = self.read(1.5)
        assert 'Pets Agent' in out or 'pets-agent' in out.lower(), "启动失败"
        print("  ✓ 启动成功")

        # 2. /help
        self.write('/help\n')
        out = self.read()
        assert '可用命令' in out or '/tasks' in out, "/help 失败"
        print("  ✓ /help")

        # 3. 测试 /tasks
        self.write('/tasks\n')
        out = self.read()
        assert '任务' in out or 'No tasks' in out or '暂无' in out, "/tasks 失败"
        print("  ✓ /tasks")

        # 4. /history
        self.write('/history\n')
        out = self.read()
        assert '[done]' in out or '历史' in out or '暂无' in out, "/history 失败"
        print("  ✓ /history")

        # 5. /clear
        self.write('/clear\n')
        self.read(0.5)
        print("  ✓ /clear")

        # === 对话测试 ===
        print("\n=== 对话测试 ===")
        # === 退出 ===
        print("\n=== 退出 ===")
        self.write('/quit\n')
        SLEEP(0.5)
        self.destroy()

        print(f"\n[PASS] 全部测试通过 ({len(self.output)} 字符)")
        return True

if __name__ == '__main__':
    t = TUITest()
    try:
        t.run()
    except AssertionError as e:
        print(f"\n[FAIL] {e}")
        print("\n=== OUTPUT ===")
        print(t.output[-2000:])  # 最后 2000 字符
        t.destroy()
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        t.destroy()
        sys.exit(1)
