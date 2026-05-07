#!/bin/bash
rm -f /tmp/todo-test.log
cd /home/jadenli/code/pets-agent
npx tsx test-full.ts >> /tmp/todo-test.log 2>&1 &
PID=$!
echo "Started PID $PID at $(date)" >> /tmp/todo-test.log
sleep 90
echo "Killing PID $PID at $(date)" >> /tmp/todo-test.log
kill $PID 2>/dev/null
echo "=== LOG ===" >> /tmp/todo-test.log
cat /tmp/todo-test.log
