#!/usr/bin/env bash
# Live demo for the presentation. Run on the Mac, on college wifi, from the project folder:
#
#   bash remote/demo.sh          start the demo servers on the GPU host (if not already running),
#                                open the tunnel and the page. Keep the window open; Ctrl+C ends it.
#   bash remote/demo.sh stop     stop the demo servers on the GPU host.
#
# The servers run on the CPU only (GPU 2 is never touched) with the final model (scnet2),
# offering vocals, drums, piano and everything else.
#   http://localhost:8765  2-second window (best quality, ~2 s delay)
#   http://localhost:8766  1-second window (~1 s delay)
set -u
HOST=${HOST:-university-gpu}

if [ "${1:-}" = "stop" ]; then
  ssh -o ConnectTimeout=10 "$HOST" 'cd ~/audio_csesa && for p in 8765 8766; do
      P=$(cat logs/stream_$p.pid 2>/dev/null)
      if [ -n "$P" ] && tr "\0" " " 2>/dev/null < /proc/$P/cmdline | grep -q stream_server.py; then
        kill $P
        for i in $(seq 30); do [ -e /proc/$P ] || break; sleep 0.5; done
        echo "stopped demo server on port $p"
      else echo "no demo server on port $p"; fi
    done' < /dev/null
  exit
fi

echo "Starting the demo servers on the GPU host..."
ssh -o ConnectTimeout=10 "$HOST" 'cd ~/audio_csesa && for cfg in "2 8765" "1 8766"; do
    set -- $cfg
    P=$(cat logs/stream_$2.pid 2>/dev/null)
    ours=0
    [ -n "$P" ] && tr "\0" " " 2>/dev/null < /proc/$P/cmdline | grep -q stream_server.py && ours=1
    if [ $ours = 1 ] && [ "$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$2/)" = 200 ]; then
      echo "  port $2: already running"
    else
      if [ $ours = 1 ]; then          # ours but not answering: restart it
        kill $P; for i in $(seq 30); do [ -e /proc/$P ] || break; sleep 0.5; done
      fi
      CUDA_VISIBLE_DEVICES= setsid nohup .venv/bin/python scripts/stream_server.py \
        --ckpt checkpoints/scnet2/best.pt --window $1 --threads 16 --host 127.0.0.1 --port $2 \
        > logs/stream_$2.log 2>&1 < /dev/null &
      echo $! > logs/stream_$2.pid
      echo "  port $2: starting"
    fi
  done
  for i in $(seq 30); do
    a=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/); b=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8766/)
    [ "$a" = 200 ] && [ "$b" = 200 ] && echo "  both servers ready" && exit 0
    sleep 2
  done
  echo "  servers did not come up - see ~/audio_csesa/logs/stream_8765.log"; exit 1' < /dev/null
case $? in
  0) ;;
  255) echo "Cannot reach the GPU server. Are you on college wifi?"; exit 1 ;;
  *) exit 1 ;;
esac

(sleep 3; open "http://localhost:8765" 2>/dev/null) &
echo
echo "Demo is live:  http://localhost:8765  (2 s window)    http://localhost:8766  (1 s window)"
echo "Keep this window open during the presentation. Ctrl+C closes the tunnel."
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
    -L 8765:localhost:8765 -L 8766:localhost:8766 "$HOST"
