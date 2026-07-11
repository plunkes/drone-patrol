#!/usr/bin/env python3
"""
Plotter ao vivo da telemetria do drone (Webots).

Escuta os pacotes UDP/JSON enviados por drone_patrol.py e plota, em tempo
real, altitude, atitude (roll/pitch/yaw), erro de yaw e as saidas de
controle. Roda em OUTRO terminal, em paralelo com a simulacao.

Uso:
    python3 tools/drone_plot.py                # porta padrao 5005
    python3 tools/drone_plot.py --port 5005 --window 20 --dump telem.csv

O plotter e o controlador sao independentes: pode iniciar/parar qualquer um
dos dois a qualquer momento (UDP e fire-and-forget).

Requisitos: matplotlib  (pip install matplotlib)
"""

import argparse
import collections
import json
import socket
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def parse_args():
    p = argparse.ArgumentParser(description="Plotter ao vivo da telemetria do drone.")
    p.add_argument("--host", default="127.0.0.1", help="IP para escutar")
    p.add_argument("--port", type=int, default=5005, help="porta UDP (default 5005)")
    p.add_argument("--window", type=float, default=20.0,
                   help="janela de tempo mostrada, em segundos (default 20)")
    p.add_argument("--dump", default=None,
                   help="opcional: grava todos os pacotes em CSV neste caminho")
    return p.parse_args()


# Series exibidas, agrupadas por subplot: (titulo, ylabel, [(chave, legenda)])
PANELS = [
    ("Altitude", "m", [("altitude", "altitude"), ("target_altitude", "alvo")]),
    ("Atitude", "rad", [("roll", "roll"), ("pitch", "pitch"), ("yaw", "yaw")]),
    ("Erro de yaw", "rad", [("yaw_err", "yaw_err")]),
    ("Saidas de controle", "-", [("roll_input", "roll_in"),
                                 ("pitch_input", "pitch_in"),
                                 ("yaw_input", "yaw_in"),
                                 ("vertical_input", "vert_in")]),
]


def main():
    args = parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.setblocking(False)
    print(f"[PLOT] Escutando UDP em {args.host}:{args.port} ... (Ctrl+C p/ sair)")

    # buffers deslizantes por chave; maxlen alto o bastante p/ a janela
    maxlen = 20000
    t_buf = collections.deque(maxlen=maxlen)
    series = {key: collections.deque(maxlen=maxlen)
              for _, _, items in PANELS for key, _ in items}

    dump_f = None
    dump_keys = None
    if args.dump:
        dump_f = open(args.dump, "w")

    fig, axes = plt.subplots(len(PANELS), 1, sharex=True, figsize=(10, 8))
    fig.canvas.manager.set_window_title("Drone telemetry")
    lines = {}
    for ax, (title, ylabel, items) in zip(axes, PANELS):
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        for key, label in items:
            (lines[key],) = ax.plot([], [], label=label, linewidth=1.2)
        ax.legend(loc="upper right", fontsize=8, ncol=len(items))
    axes[-1].set_xlabel("tempo (s)")
    fig.tight_layout()

    def drain_socket():
        """Le todos os pacotes pendentes sem bloquear."""
        nonlocal dump_keys
        got = False
        while True:
            try:
                raw, _ = sock.recvfrom(65535)
            except BlockingIOError:
                break
            except OSError:
                break
            try:
                d = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            t_buf.append(d.get("t", 0.0))
            for key in series:
                series[key].append(d.get(key, float("nan")))
            if dump_f is not None:
                if dump_keys is None:
                    dump_keys = sorted(d.keys())
                    dump_f.write(",".join(dump_keys) + "\n")
                dump_f.write(",".join(str(d.get(k, "")) for k in dump_keys) + "\n")
            got = True
        return got

    def update(_frame):
        drain_socket()
        if not t_buf:
            return list(lines.values())
        t_now = t_buf[-1]
        t_min = t_now - args.window
        for ax, (_, _, items) in zip(axes, PANELS):
            for key, _label in items:
                lines[key].set_data(t_buf, series[key])
            ax.set_xlim(t_min, max(t_now, t_min + args.window))
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
        return list(lines.values())

    _anim = FuncAnimation(fig, update, interval=50, blit=False,
                          cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        if dump_f is not None:
            dump_f.close()
            print(f"[PLOT] CSV salvo em '{args.dump}'.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
