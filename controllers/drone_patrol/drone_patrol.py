"""
Controlador do Drone Mavic 2 Pro - Sistema de Vigilancia Autonomo (SSC0714)
===========================================================================
Arquitetura modular (diretriz do trabalho):

    PID              -> Etapa 1: estabilizacao de voo (Kp, Ki, Kd)
    VisionDetector   -> Etapa 2: deteccao de intrusos por Visao Computacional
    TrajectoryLogger -> Etapa 3: gravacao da trajetoria em .txt
    KeyboardListener -> Etapa 3: tecla 'F' encerra e salva
    VigilanceDrone   -> orquestra tudo

NAO usa Supervisor (Etapa 2): a deteccao e feita processando a imagem da
camera do proprio drone, nao lendo posicoes do Scene Tree.

Comportamento inicial (Etapa 1): HOVER estavel na posicao de decolagem.
"""

from controller import Robot
import math

# ===============================================================
# CONFIGURACOES GERAIS
# ===============================================================
TARGET_ALTITUDE = 2.0          # setpoint de altitude para o hover (m)

# --- Constantes do controlador PADRAO do Mavic 2 Pro (Webots) ---
# Valores identicos ao mavic2pro.py oficial -> hover comprovadamente estavel.
K_VERTICAL_THRUST = 68.5       # empuxo base que sustenta o drone no ar
K_VERTICAL_OFFSET = 0.6        # bias no alvo de altitude (compensa o "sag")

# --- NAVEGACAO / TRAJETORIA ---
# MODE "HOVER"  -> paira parado na posicao de decolagem
# MODE "PATROL" -> percorre a lista WAYPOINTS em loop
MODE = "PATROL"

# Rota (x, y) em coordenadas do mundo. Arena 60x60 centrada em (3.17,1.76)
# -> x uteis ~[-27, 33]. Drone decola em (-22.89, 0). Retangulo a esquerda,
# longe dos paineis internos (x~10). Veja no README como pegar mais pontos.
WAYPOINTS = [
    (-22.0,  -22.0),
    (5.0, -22.0),
    (5.0, 0.0)
]
WAYPOINT_TOLERANCE = 0.6       # dist (m) para considerar o waypoint alcancado

# Ganhos de deslocamento horizontal (inclina o drone p/ ir ao waypoint)
K_FORWARD_P = 0.05              # dist -> inclinacao de pitch (avanco)
MAX_PITCH_DISTURBANCE = 1.0    # limite de nose-down (evita mergulho)
MAX_YAW_DISTURBANCE = 0.4      # limite de giro por passo

LOG_FILE_PATH = "log_trajetoria.txt"

# --- Etapa 2: limiares de Visao Computacional ---
# Blob e "detectado" se area (nº de pixels da cor) > MIN_BLOB_AREA.
MIN_BLOB_AREA = 40
VISION_STRIDE = 2              # amostra 1 a cada N pixels (performance)
ALERT_COOLDOWN = 3.0           # s entre alertas repetidos do mesmo alvo

# ===============================================================
# GANHOS PID  (base = controlador padrao do Mavic 2 Pro)
#   Formato dos logs: [Setpoint | Atual | Erro | Saida]
#   Ki=0 -> o controlador oficial nao usa termo integral.
#   Kd nos eixos de atitude = 1.0 -> a "velocidade" do giroscopio e a
#   propria derivada do angulo (amortecimento). Nao mexa nisso ainda.
# ===============================================================
#                    Kp     Ki     Kd
GAINS_ALTITUDE  = (   1.5,  0.3,   2.0 )   # k_vertical_p do padrao
GAINS_ROLL      = (  50.0,  1.0,   3.0 )   # k_roll_p  + amortecimento gyro
GAINS_PITCH     = (  30.0,  1.0,   3.0 )   # k_pitch_p + amortecimento gyro
GAINS_YAW       = (   2.0,  0.1,   3.0 )   # trava a proa inicial


def clamp(value, low, high):
    return max(low, min(value, high))


def normalize_angle(angle):
    """Mantem o angulo em [-pi, pi] (evita salto de erro em +/-180)."""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


# ===============================================================
# MODULO 1 - CONTROLE (PID)
# ===============================================================
class PID:
    """Controlador PID generico com anti-windup e saturacao de saida.

    Guarda o ultimo calculo (setpoint, medida, erro, saida) para os logs
    de debug exigidos na Etapa 1.
    """

    def __init__(self, kp, ki, kd, name="",
                 out_limits=(-float("inf"), float("inf")),
                 integral_limit=10.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.name = name
        self.out_min, self.out_max = out_limits
        self.integral_limit = integral_limit      # anti-windup (satura I)
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True
        # snapshot p/ debug: setpoint, valor atual, erro, saida
        self.debug = (0.0, 0.0, 0.0, 0.0)

    def update(self, setpoint, measured, dt, derivative=None, error_clamp=None):
        # (P) erro instantaneo entre onde queremos e onde estamos
        error = setpoint - measured
        # opcional: satura o erro (o padrao do Mavic limita o erro de
        # altitude a [-1,1] antes de aplicar o ganho vertical)
        if error_clamp is not None:
            error = clamp(error, -error_clamp, error_clamp)

        # (I) acumula o erro no tempo para eliminar erro de regime.
        # O clamp abaixo e o anti-windup: impede o termo integral de
        # crescer sem limite enquanto o drone ainda nao chegou no alvo.
        self._integral += error * dt
        self._integral = clamp(self._integral,
                               -self.integral_limit, self.integral_limit)

        # (D) taxa de variacao do erro -> amortece oscilacao/overshoot.
        # Se 'derivative' for passado (ex.: velocidade do giroscopio), usa-o
        # direto -> reproduz o "+ *_velocity" do controlador oficial e evita
        # ruido de derivada numerica. Senao, calcula d(erro)/dt.
        if derivative is not None:
            deriv = derivative
        else:
            deriv = 0.0 if self._first else (error - self._prev_error) / dt
        self._first = False
        self._prev_error = error

        # Saida = soma ponderada dos tres termos (linha critica do PID)
        output = self.kp * error + self.ki * self._integral + self.kd * deriv
        output = clamp(output, self.out_min, self.out_max)

        self.debug = (setpoint, measured, error, output)
        return output

    def log(self):
        sp, val, err, out = self.debug
        print(f"[PID {self.name:>4}] Setpoint={sp:+.3f} | Atual={val:+.3f} | "
              f"Erro={err:+.3f} | Saida={out:+.3f}")


# ===============================================================
# MODULO 2 - VISAO COMPUTACIONAL
# ===============================================================
class VisionDetector:
    """Detecta intrusos processando o buffer BGRA da camera.

    Sem Supervisor: apenas pixels. Procura dois alvos por cor:
        - vermelho (cubo/objeto de interesse)
        - verde    (calca do pedestre = intruso)
    Retorna area do blob e centroide (u, v) de cada cor.
    """

    def __init__(self, camera):
        self.camera = camera
        self.width = camera.getWidth()
        self.height = camera.getHeight()

    @staticmethod
    def _is_red(r, g, b):
        return r > 120 and g < 80 and b < 80

    @staticmethod
    def _is_green(r, g, b):
        return g > 120 and r < 90 and b < 90

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


# ===============================================================
# MODULO 3 - I/O (LOG EM ARQUIVO + TECLADO)
# ===============================================================
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


# ===============================================================
# MODULO 4 - ORQUESTRACAO DO DRONE
# ===============================================================
class VigilanceDrone:
    def __init__(self):
        self.robot = Robot()                                  # <- NAO Supervisor
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
        self.pid_alt   = PID(*GAINS_ALTITUDE, name="ALT")
        self.pid_roll  = PID(*GAINS_ROLL,     name="ROLL")
        self.pid_pitch = PID(*GAINS_PITCH,    name="PTCH")
        self.pid_yaw   = PID(*GAINS_YAW,      name="YAW")

        self.vision = VisionDetector(self.camera)
        self.logger = TrajectoryLogger(LOG_FILE_PATH)
        self.keys = KeyboardListener(self.robot, self.timestep)

        self.target_yaw = None          # proa travada, definida no 1o passo
        self.last_alert = {"red": -1e9, "green": -1e9}
        self.sim_time = 0.0
        self.waypoint_index = 0         # waypoint atual da rota (modo PATROL)

    # -----------------------------------------------------------
    def compute_motor_commands(self, target=None):
        """Roda os 4 PIDs e mistura a saida nos quatro motores (mixagem
        padrao do quadrirotor em X).

        target=None  -> hover (proa travada, sem deslocamento).
        target=(tx,ty) -> gira para encarar o ponto e avanca ate ele.
        Retorna a distancia horizontal ate o alvo (0.0 em hover)."""
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        roll_rate, pitch_rate, _ = self.gyro.getValues()
        x, y, altitude = self.gps.getValues()

        if self.target_yaw is None:
            self.target_yaw = yaw       # trava a proa inicial como setpoint

        # --- direcao/distancia ate o waypoint (navegacao) ---
        distance = 0.0
        pitch_disturbance = 0.0
        if target is None:
            desired_yaw = self.target_yaw          # hover: mantem a proa
        else:
            tx, ty = target
            dx, dy = tx - x, ty - y
            distance = math.hypot(dx, dy)
            desired_yaw = math.atan2(dy, dx)       # aponta o nariz p/ o alvo
            yaw_err_nav = normalize_angle(desired_yaw - yaw)
            # so avanca quando ja esta razoavelmente alinhado (cos>0);
            # nose-down proporcional a distancia -> voa para frente
            alignment = max(0.0, math.cos(yaw_err_nav))
            pitch_disturbance = -clamp(K_FORWARD_P * distance * alignment,
                                       0.0, MAX_PITCH_DISTURBANCE)

        # --- PID de atitude (setpoint = nivelado, 0 rad) ---
        # O PID leva o angulo a zero: saida = Kp*(0-ang) + Kd*(-rate).
        # Negamos para obter o "*_input" no mesmo sinal do controlador
        # padrao ( = Kp*ang + rate ), usado na mixagem abaixo.
        roll_input  = -self.pid_roll.update(0.0, roll, self.dt, derivative=-roll_rate)
        pitch_input = -self.pid_pitch.update(0.0, pitch, self.dt, derivative=-pitch_rate)
        pitch_input += pitch_disturbance           # comando de avanco

        # --- PID de yaw: encara o alvo (ou trava a proa em hover) ---
        yaw_err = normalize_angle(desired_yaw - yaw)   # trata o wrap +/-pi
        yaw_input = clamp(self.pid_yaw.update(0.0, -yaw_err, self.dt),
                          -MAX_YAW_DISTURBANCE, MAX_YAW_DISTURBANCE)

        # --- PID de altitude: incremento de empuxo (erro saturado em +-1
        #     como no padrao; offset compensa a perda de sustentacao) ---
        vertical_input = self.pid_alt.update(
            TARGET_ALTITUDE + K_VERTICAL_OFFSET, altitude, self.dt, error_clamp=1.0)

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
        rr_, pr_, _ = self.gyro.getValues()
        self.cam_roll.setPosition(-0.115 * rr_)
        self.cam_pitch.setPosition(-0.1 * pr_)

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

            # --- Etapa 1: debug dos PIDs no console ---
            if step % log_every == 0:
                self.pid_alt.log()
                self.pid_roll.log()
                self.pid_pitch.log()
                self.pid_yaw.log()

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


if __name__ == "__main__":
    VigilanceDrone().run()
