#!/bin/bash

# Create or attach to tmux monitor session
SESSION_NAME="monitor"

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Remove and recreate..."
    tmux kill-session -t "$SESSION_NAME"
fi
# Create new detached session
tmux new-session -d -s "$SESSION_NAME"

# Split into 2 vertical panes (left and right)
tmux split-window -h -t "$SESSION_NAME"

# Split the left pane (pane 0) horizontally into 2 panes (top and bottom)
tmux split-window -h -t "$SESSION_NAME:0.0"

# Split the right pane (pane 1) horizontally into 2 panes (top and bottom)
tmux split-window -v -t "$SESSION_NAME:0.2"

tmux send-keys -t "$SESSION_NAME:0.0" "clear; while true; do tput cup 0 0; nvidia-smi -i 0 -q -d POWER,CLOCK; sleep 0.01; done" Enter
tmux send-keys -t "$SESSION_NAME:0.1" "clear; while true; do tput cup 0 0; nvidia-smi -i 1 -q -d POWER,CLOCK; sleep 0.01; done" Enter
tmux send-keys -t "$SESSION_NAME:0.2" "nvtop" Enter
tmux send-keys -t "$SESSION_NAME:0.3" "glances" Enter

echo "Session '$SESSION_NAME' created"