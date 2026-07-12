from controller import Robot
import math
import json
import socket
import colorsys


TARGET_ALTITUDE = 2.0          # setpoint de altitude para o hover (m)

# Constantes de controle (padrao Mavic 2 Pro, achadas empiricamente).
# Logica P + amortecimento por gyro; vertical usa funcao cubica.
K_VERTICAL_THRUST = 68.5       # empuxo base que sustenta o drone no ar
K_VERTICAL_OFFSET = 0.6        # bias no alvo de altitude (compensa o "sag")
K_VERTICAL_P = 3.0             # ganho P da altitude
K_VERTICAL_D = 5.0             # amortece a taxa de subida (mata overshoot)
K_ROLL_P = 50.0               # ganho P do roll
K_PITCH_P = 30.0              # ganho P do pitch

# Trava de proa (yaw). Sem isso o drone gira/deriva sozinho no hover.
K_YAW_HOLD_P = 3.0            # erro de proa -> torque de correcao
K_YAW_HOLD_D = 1.5           # amortece a rotacao (yaw rate)

# MODE "HOVER"  -> paira parado na posicao de decolagem
# MODE "PATROL" -> percorre a lista WAYPOINTS em loop
MODE = "PATROL"

# Rota (x, y) em coordenadas do mundo.
WAYPOINTS = [
    (-22.0,  -22.0),
    (5.0, -22.0),
    (5.0, 0.0)
]
WAYPOINT_TOLERANCE = 0.6       # dist (m) para considerar o waypoint alcancado

# Controle de posicao horizontal (PD).  Sem isto o drone so nivela a
# atitude e desliza para sempre com a velocidade residual da subida.
# Segura o ponto de decolagem em HOVER e persegue o waypoint em PATROL.
K_POS_P = 0.4                 # erro de posicao (m) -> inclinacao
K_VEL_D = 1.5                 # amortece a velocidade horizontal (mata o glide)
MAX_POS_ERROR = 3.0          # satura o erro de posicao (evita tilt extremo)
MAX_TILT_DISTURBANCE = 1.0    # limite de inclinacao por eixo (roll/pitch)
MAX_YAW_DISTURBANCE = 1.3      # limite de giro por passo

# Log de trajetoria na raiz do projeto (junto do telem.csv).
# CWD do controlador = controllers/drone_patrol -> sobe 2 niveis.
LOG_FILE_PATH = "../../log_trajetoria.txt"

# Telemetria ao vivo: envia estado por UDP p/ o plotter (drone_plot.py).
# Fire-and-forget: se ninguem escuta, nao trava a simulacao.
TELEMETRY_ENABLE = True
TELEMETRY_HOST = "127.0.0.1"
TELEMETRY_PORT = 5005

# Blob e "detectado" se area (nº de pixels da cor) > MIN_BLOB_AREA.
MIN_BLOB_AREA = 40
VISION_STRIDE = 2              # amostra 1 a cada N pixels (performance)
ALERT_COOLDOWN = 3.0           # s entre alertas repetidos do mesmo alvo

# Deteccao de cor em HSV (robusta a sombra/contraluz).  O matiz (hue) se
# mantem sob variacao de luz; satura/valor rejeitam cinza e preto.
HUE_RED_LOW = 15.0            # vermelho: hue <= 15 ou >= 345 graus
HUE_RED_HIGH = 345.0
HUE_GREEN_LOW = 80.0         # verde: 80..160 graus
HUE_GREEN_HIGH = 160.0
SAT_MIN = 0.28               # abaixo disso = lavado/cinza -> ignora
VAL_MIN = 0.15               # abaixo disso = escuro demais -> ignora


def clamp(value, low, high):
    return max(low, min(value, high))


def normalize_angle(angle):
    """Mantem o angulo em [-pi, pi] (evita salto de erro em +/-180)."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


class VisionDetector:
    """Detecta intrusos processando o buffer BGRA da camera.

    Procura dois alvos por cor:
        - vermelho (cubo/objeto de interesse)
        - verde    (calca do pedestre = intruso)
    Retorna area do blob e centroide (u, v) de cada cor.
    """

    def __init__(self, camera):
        self.camera = camera
        self.width = camera.getWidth()
        self.height = camera.getHeight()

    @staticmethod
    def _hsv(r, g, b):
        """(hue graus, sat 0-1, val 0-1) a partir de RGB 0-255."""
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        return h * 360.0, s, v

    @staticmethod
    def _is_red(r, g, b):
        h, s, v = VisionDetector._hsv(r, g, b)
        return (h <= HUE_RED_LOW or h >= HUE_RED_HIGH) and s >= SAT_MIN and v >= VAL_MIN

    @staticmethod
    def _is_green(r, g, b):
        h, s, v = VisionDetector._hsv(r, g, b)
        return HUE_GREEN_LOW <= h <= HUE_GREEN_HIGH and s >= SAT_MIN and v >= VAL_MIN

    def scan(self):
        """Varre a imagem (com stride) e devolve dict por cor com
        {'area', 'u', 'v'}. area=0 significa nada relevante."""
        image = self.camera.getImage()
        if image is None:
            return None

        w, h, s = self.width, self.height, VISION_STRIDE
        acc = {"red":   {"n": 0, "su": 0, "sv": 0},
               "green": {"n": 0, "su": 0, "sv": 0}}

        for v in range(0, h, s):
            row = v * w * 4               # BGRA -> 4 bytes por pixel
            for u in range(0, w, s):
                i = row + u * 4
                b = image[i]
                g = image[i + 1]
                r = image[i + 2]
                if self._is_red(r, g, b):
                    a = acc["red"]
                elif self._is_green(r, g, b):
                    a = acc["green"]
                else:
                    continue
                a["n"] += 1
                a["su"] += u
                a["sv"] += v

        result = {}
        for color, a in acc.items():
            if a["n"] > 0:
                result[color] = {"area": a["n"],
                                 "u": a["su"] / a["n"],
                                 "v": a["sv"] / a["n"]}
            else:
                result[color] = {"area": 0, "u": 0, "v": 0}
        return result


class TrajectoryLogger:
    """Grava a trajetoria no formato:  <x> <y> <z>  (uma linha por passo)."""

    def __init__(self, path):
        self.path = path
        self._f = open(path, "w")

    def write(self, x, y, z):
        self._f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    def close(self):
        if self._f and not self._f.closed:
            self._f.flush()
            self._f.close()
            print(f"[I/O] Log salvo em '{self.path}'.")


class Telemetry:
    """Envia o estado de controle por UDP (JSON) p/ um plotter externo.

    UDP nao-bloqueante: se o plotter nao estiver rodando, os pacotes sao
    descartados sem travar a simulacao."""

    def __init__(self, host, port, enable=True):
        self.enable = enable
        self.addr = (host, port)
        self.sock = None
        if enable:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setblocking(False)
            print(f"[TELEMETRY] Enviando UDP p/ {host}:{port}")

    def send(self, data):
        if not self.enable:
            return
        try:
            self.sock.sendto(json.dumps(data).encode("utf-8"), self.addr)
        except OSError:
            pass    # buffer cheio / sem rota -> ignora, nao trava o loop

    def close(self):
        if self.sock:
            self.sock.close()


class KeyboardListener:
    """Escuta o teclado. Retorna True se 'F' (ou 'f') foi pressionada."""

    def __init__(self, robot, timestep):
        self.keyboard = robot.getKeyboard()
        self.keyboard.enable(timestep)

    def quit_requested(self):
        key = self.keyboard.getKey()
        while key != -1:
            if key in (ord('F'), ord('f')):
                return True
            key = self.keyboard.getKey()
        return False



class VigilanceDrone:
    def __init__(self):
        self.robot = Robot()                                  
        self.timestep = int(self.robot.getBasicTimeStep())
        self.dt = self.timestep / 1000.0

        # --- sensores ---
        self.gps = self.robot.getDevice("gps");            self.gps.enable(self.timestep)
        self.imu = self.robot.getDevice("inertial unit");  self.imu.enable(self.timestep)
        self.gyro = self.robot.getDevice("gyro");          self.gyro.enable(self.timestep)
        self.camera = self.robot.getDevice("camera");      self.camera.enable(self.timestep)

        # gimbal (estabilizacao visual da camera)
        self.cam_roll = self.robot.getDevice("camera roll")
        self.cam_pitch = self.robot.getDevice("camera pitch")

        # --- motores (helices em modo velocidade) ---
        self.motors = [self.robot.getDevice(n) for n in
                       ("front left propeller", "front right propeller",
                        "rear left propeller",  "rear right propeller")]
        for m in self.motors:
            m.setPosition(float("inf"))
            m.setVelocity(1.0)

        # --- modulos ---
        self.vision = VisionDetector(self.camera)
        self.logger = TrajectoryLogger(LOG_FILE_PATH)
        self.keys = KeyboardListener(self.robot, self.timestep)

        self.target_yaw = None          # proa travada, definida no 1o passo
        self.last_alert = {"red": -1e9, "green": -1e9}
        self.sim_time = 0.0
        self.waypoint_index = 0         # waypoint atual da rota (modo PATROL)
        self.prev_alt = None            # altitude anterior (p/ estimar vz)
        self.prev_xy = None             # (x,y) anterior (p/ estimar vx,vy)
        self.hover_xy = None            # ponto de decolagem travado (HOVER)
        self.debug_inputs = (0.0, 0.0, 0.0, 0.0)  # roll/pitch/yaw/vert p/ log
        self.telemetry = Telemetry(TELEMETRY_HOST, TELEMETRY_PORT,
                                   enable=TELEMETRY_ENABLE)

    # -----------------------------------------------------------
    def compute_motor_commands(self, target=None):
        """Roda os 4 PIDs e mistura a saida nos quatro motores (mixagem
        padrao do quadrirotor em X).

        target=None  -> hover (proa travada, sem deslocamento).
        target=(tx,ty) -> gira para encarar o ponto e avanca ate ele.
        Retorna a distancia horizontal ate o alvo (0.0 em hover)."""
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        roll_rate, pitch_rate, yaw_rate = self.gyro.getValues()
        x, y, altitude = self.gps.getValues()

        # velocidades por diferenca finita (p/ amortecer altitude e posicao)
        vz = 0.0 if self.prev_alt is None else (altitude - self.prev_alt) / self.dt
        if self.prev_xy is None:
            vx = vy = 0.0
        else:
            vx = (x - self.prev_xy[0]) / self.dt
            vy = (y - self.prev_xy[1]) / self.dt
        self.prev_alt = altitude
        self.prev_xy = (x, y)

        if self.target_yaw is None:
            self.target_yaw = yaw       # trava a proa inicial como setpoint
        if self.hover_xy is None:
            self.hover_xy = (x, y)      # trava o ponto de decolagem

        # --- alvo de posicao e de proa ---
        # HOVER: segura o ponto de decolagem, mantem a proa inicial.
        # PATROL: persegue o waypoint e encara-o.
        if target is None:
            hold_x, hold_y = self.hover_xy
            desired_yaw = self.target_yaw
        else:
            hold_x, hold_y = target
            desired_yaw = math.atan2(hold_y - y, hold_x - x)
        distance = math.hypot(hold_x - x, hold_y - y)

        # --- controle de posicao horizontal (PD em coords do corpo) ---
        # Projeta erro de posicao e velocidade nos eixos frente/esquerda
        # do drone (girados pelo yaw) e gera as inclinacoes de correcao.
        c, s = math.cos(yaw), math.sin(yaw)
        dpx, dpy = hold_x - x, hold_y - y
        e_fwd  = clamp(dpx * c + dpy * s, -MAX_POS_ERROR, MAX_POS_ERROR)
        e_left = clamp(-dpx * s + dpy * c, -MAX_POS_ERROR, MAX_POS_ERROR)
        v_fwd  = vx * c + vy * s
        v_left = -vx * s + vy * c
        # aceleracao desejada = P*erro - D*velocidade (freia o glide)
        a_fwd  = K_POS_P * e_fwd  - K_VEL_D * v_fwd
        a_left = K_POS_P * e_left - K_VEL_D * v_left
        # frente = pitch negativo; esquerda = roll positivo (conv. Mavic)
        pitch_disturbance = -clamp(a_fwd, -MAX_TILT_DISTURBANCE, MAX_TILT_DISTURBANCE)
        roll_disturbance  =  clamp(a_left, -MAX_TILT_DISTURBANCE, MAX_TILT_DISTURBANCE)

        # --- controle de atitude (P + amortecimento por gyro) ---
        # Formula padrao Mavic: Kp*angulo + taxa_angular + disturbio.
        # angulo saturado em +-1 rad para limitar a autoridade.
        roll_input = K_ROLL_P * clamp(roll, -1.0, 1.0) + roll_rate + roll_disturbance
        pitch_input = K_PITCH_P * clamp(pitch, -1.0, 1.0) + pitch_rate + pitch_disturbance

        # --- trava de proa (P sobre o erro de yaw + D sobre yaw_rate) ---
        # Sem isso o drone deriva/gira; D amortece a oscilacao de yaw.
        yaw_err = normalize_angle(desired_yaw - yaw)
        yaw_input = clamp(K_YAW_HOLD_P * yaw_err - K_YAW_HOLD_D * yaw_rate,
                          -MAX_YAW_DISTURBANCE, MAX_YAW_DISTURBANCE)

        # --- altitude: funcao cubica do erro saturado + amortecimento da
        #     taxa de subida (mata o overshoot/oscilacao na subida) ---
        clamped_diff_alt = clamp(TARGET_ALTITUDE - altitude + K_VERTICAL_OFFSET,
                                 -1.0, 1.0)
        vertical_input = K_VERTICAL_P * (clamped_diff_alt ** 3) - K_VERTICAL_D * vz

        base = K_VERTICAL_THRUST + vertical_input    # empuxo comum aos 4 motores

        # Mixagem padrao do Mavic 2 Pro (geometria em X)
        fl = base - roll_input + pitch_input - yaw_input
        fr = base + roll_input + pitch_input + yaw_input
        rl = base - roll_input - pitch_input + yaw_input
        rr = base + roll_input - pitch_input - yaw_input

        # sentido de giro das helices (2 horario, 2 anti-horario)
        self.motors[0].setVelocity(fl)
        self.motors[1].setVelocity(-fr)
        self.motors[2].setVelocity(-rl)
        self.motors[3].setVelocity(rr)

        # gimbal compensa a inclinacao para manter a imagem estavel
        self.cam_roll.setPosition(-0.115 * roll_rate)
        self.cam_pitch.setPosition(-0.1 * pitch_rate)

        self.debug_inputs = (roll_input, pitch_input, yaw_input, vertical_input)
        # snapshot completo p/ telemetria/plotter
        self.telemetry.send({
            "t": round(self.sim_time, 3),
            "altitude": altitude, "target_altitude": TARGET_ALTITUDE, "vz": vz,
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "yaw_err": yaw_err, "desired_yaw": desired_yaw,
            "roll_rate": roll_rate, "pitch_rate": pitch_rate, "yaw_rate": yaw_rate,
            "roll_input": roll_input, "pitch_input": pitch_input,
            "yaw_input": yaw_input, "vertical_input": vertical_input,
            "x": x, "y": y, "distance": distance,
            "vx": vx, "vy": vy, "v_fwd": v_fwd, "v_left": v_left,
            "pitch_dist": pitch_disturbance, "roll_dist": roll_disturbance,
            "fl": fl, "fr": fr, "rl": rl, "rr": rr,
        })

        return distance

    # -----------------------------------------------------------
    def run_vision(self):
        """Etapa 2: processa a imagem e emite ALERTA MONITORAMENTO."""
        det = self.vision.scan()
        if det is None:
            return
        _, _, yaw = self.imu.getRollPitchYaw()
        x, y, z = self.gps.getValues()

        labels = {"red": "Objeto suspeito (vermelho)",
                  "green": "INTRUSO (pedestre)"}
        for color in ("green", "red"):
            info = det[color]
            if info["area"] >= MIN_BLOB_AREA:
                if self.sim_time - self.last_alert[color] >= ALERT_COOLDOWN:
                    self.last_alert[color] = self.sim_time
                    print("=" * 60)
                    print(f"[ALERTA MONITORAMENTO] {labels[color]} detectado!")
                    print(f"    Area do blob : {info['area']} px  "
                          f"@ imagem(u={info['u']:.0f}, v={info['v']:.0f})")
                    print(f"    Posicao GPS  : x={x:.2f}  y={y:.2f}  z={z:.2f}")
                    print(f"    Orientacao   : yaw={math.degrees(yaw):+.1f} deg")
                    print("=" * 60)

    # -----------------------------------------------------------
    def run(self):
        print(f"=== Sistema de Vigilancia iniciado (Mavic 2 Pro | modo {MODE}) ===")
        print("Pressione 'F' na janela 3D para salvar o log e encerrar.\n")
        log_every = max(1, int(0.5 / self.dt))   # log de PID a cada ~0.5 s
        step = 0

        while self.robot.step(self.timestep) != -1:
            self.sim_time += self.dt
            step += 1

            # --- Etapa 1: controle + navegacao ---
            if MODE == "PATROL":
                target = WAYPOINTS[self.waypoint_index]
                distance = self.compute_motor_commands(target)
                if distance < WAYPOINT_TOLERANCE:      # chegou -> proximo ponto
                    self.waypoint_index = (self.waypoint_index + 1) % len(WAYPOINTS)
                    print(f"[NAV] Waypoint {self.waypoint_index} -> "
                          f"{WAYPOINTS[self.waypoint_index]}")
            else:
                self.compute_motor_commands()          # hover

            # --- Etapa 1: debug do controle no console ---
            if step % log_every == 0:
                ri, pi, yi, vi = self.debug_inputs
                _, _, alt = self.gps.getValues()
                # print(f"[CTRL] alt={alt:+.2f} | roll_in={ri:+.2f} | "
                      # f"pitch_in={pi:+.2f} | yaw_in={yi:+.2f} | vert_in={vi:+.2f}")

            # --- Etapa 2: visao ---
            self.run_vision()

            # --- Etapa 3: log de trajetoria ---
            x, y, z = self.gps.getValues()
            self.logger.write(x, y, z)

            # --- Etapa 3: tecla 'F' encerra e salva ---
            if self.keys.quit_requested():
                print("\nTecla 'F' pressionada. Encerrando...")
                break

        # fecha o arquivo com seguranca (flush + close).
        # Ao retornar de run(), o controlador termina e a simulacao para.
        self.logger.close()
        self.telemetry.close()


if __name__ == "__main__":
    VigilanceDrone().run()
