from controller import Robot
import math
import json
import socket
import colorsys
import heapq


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

# Rota (x, y, altitude) em coordenadas do mundo. Loop em torno do edificio,
# com altitude variando por waypoint (movimento vertical).
WAYPOINTS = [
    (-6.0, -10.0, 2.5),
    (8.0, -10.0, 4.0),
    (24.0, -10.0, 2.5),
    (32.0, 2.0, 6.0),
    (32.0, 18.0, 3.0),
    (24.0, 30.0, 5.0),
    (8.0, 30.0, 2.5),
    (-6.0, 30.0, 4.0),
    (-12.0, 18.0, 3.0),
    (-12.0, 2.0, 6.0),
]
WAYPOINT_TOLERANCE = 0.6       # dist (m) para considerar o waypoint alcancado
SUBWP_TOLERANCE = 1.0          # dist (m) p/ avancar entre pontos do caminho A*

# Mapa de obstaculos CONHECIDOS (nao inclui anomalias: cubos/pessoas).
# Deve refletir o mundo. rects = (xmin,xmax,ymin,ymax); circles = (cx,cy,r).
GRID_BOUNDS = (-20.0, 40.0, -18.0, 38.0)   # (xmin,xmax,ymin,ymax) do grid
GRID_CELL = 1.0                            # tamanho da celula (m)
OBSTACLE_INFLATE = 0.8                     # folga: raio do drone + margem
STATIC_RECTS = [
    (-0.3, 20.3, -0.3, 14.3),   # edificio (preenchido)
    (22.0, 30.0, 4.0, 12.0),    # anexo
    (23.0, 25.0, -7.0, -5.0),   # caixote (box_1)
    (1.0, 3.0, -11.0, -9.0),    # box_2
    (11.0, 13.0, 26.0, 28.0),   # box_3
    (26.0, 28.0, 24.0, 26.0),   # box_4
    (-16.0, -14.0, 12.0, 14.0),  # box_5
    (20.0, 22.0, -12.0, -10.0),  # box_6
]
STATIC_CIRCLES = [
    (16.0, -6.0, 0.5),          # arvore 1
    (32.0, 12.0, 0.6),          # arvore 2
    (-4.0, 20.0, 0.5),          # arvore 3
    (6.0, 30.0, 0.5),           # arvore 4
    (0.0, -6.0, 0.6),           # arvore 5
    (-9.0, 6.0, 0.5),           # arvore 6
    (-9.0, 26.0, 0.6),          # arvore 7
    (14.0, 22.0, 0.5),          # arvore 8
    (28.0, -2.0, 0.5),          # arvore 9
    (2.0, 20.0, 0.6),           # arvore 10
    (18.0, 30.0, 0.5),          # arvore 11
    (-12.0, 10.0, 0.5),         # arvore 12
]

# Parada + reorientacao em cada waypoint (freia -> gira p/ o proximo -> segue)
WAYPOINT_ARRIVE_SPEED = 0.25   # m/s abaixo disso = "parado" no waypoint
WAYPOINT_PAUSE = 1.5           # s de permanencia parado em cada waypoint
WAYPOINT_YAW_TOL = 0.10        # rad (~6 deg) alinhamento p/ seguir

# Controle de posicao horizontal (PD).  Sem isto o drone so nivela a
# atitude e desliza para sempre com a velocidade residual da subida.
# Segura o ponto de decolagem em HOVER e persegue o waypoint em PATROL.
K_POS_P = 0.4                 # erro de posicao (m) -> inclinacao
K_VEL_D = 1.5                 # amortece a velocidade horizontal (mata o glide)
MAX_POS_ERROR = 3.0          # satura o erro de posicao (evita tilt extremo)
MAX_TILT_DISTURBANCE = 1.0    # limite de inclinacao por eixo (roll/pitch)
MAX_YAW_DISTURBANCE = 1.3      # limite de giro por passo

# Desvio reativo (campo potencial) com o anel de sensores de distancia.
# Empurra o drone para longe de qualquer obstaculo dentro de REACT_RANGE,
# mantendo pelo menos SAFE_DISTANCE de folga.
SENSOR_COUNT = 8
REACT_RANGE = 2.0            # m: comeca a repelir abaixo disto
SAFE_DISTANCE = 0.5          # m: folga minima desejada de qualquer parede
K_REPULSION = 2.0           # ganho da repulsao (forte perto de SAFE_DISTANCE)
REACT_MIN_ALT = 1.0         # m: so liga o desvio reativo acima desta altitude

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

# Tratamento de anomalias (cubo vermelho / pessoa verde).
ANOMALY_COOLDOWN = 30.0       # s antes de reagir de novo ao mesmo tipo
INSPECT_TIMEOUT = 8.0         # s max encarando o cubo antes de desistir
INSPECT_CENTER_TOL = 0.10     # rad: cubo centrado -> registra
FOLLOW_DURATION = 10.0        # s seguindo a pessoa
FOLLOW_LOST_TIMEOUT = 3.0     # s sem ver a pessoa -> considera perdida
FOLLOW_APPROACH = 2.0         # m a frente: alvo de aproximacao no follow
REGISTER_PERIOD = 1.0        # s entre registros continuos (follow)
REGISTER_NOMINAL_RANGE = 3.0  # m: standoff estimado quando nao ha profundidade
ANOMALY_LOG_PATH = "../../anomalias.txt"

# APPROACH: ao detectar, aproxima ate <= APPROACH_RANGE antes de registrar.
APPROACH_RANGE = 10.0         # m: distancia maxima p/ registrar a anomalia
APPROACH_TIMEOUT = 25.0       # s max tentando aproximar
APPROACH_LOST_TIMEOUT = 3.0   # s sem ver o alvo -> desiste

# Camera apontada p/ baixo (rad). Sem isto a camera olha no horizonte e nao
# enxerga anomalias no chao. Tambem e o angulo usado na telemetria de alcance.
CAM_PITCH_DOWN = 0.35         # ~20 deg abaixo da horizontal

# Altura assumida do alvo (m) p/ estimar alcance pelo plano do chao.
TARGET_HEIGHT = {"red": 0.25, "green": 0.9}

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


class Planner:
    """Planejador DELIBERATIVO: Theta* (A* de "angulo livre") sobre um grid
    de ocupacao dos obstaculos CONHECIDOS do mapa.

    Por que Theta* e nao A* puro?
      O A* classico so anda de celula em celula (8 direcoes), entao o
      caminho sai "escadinha": muitos zigue-zagues de 45 deg que o drone
      teria de seguir. O Theta* corrige isso: ao expandir um vizinho, ele
      TENTA liga-lo direto ao AVO (o pai do no atual) em vez do pai.
      Se existe linha de visada livre entre avo e vizinho, o caminho pula
      o intermediario. O resultado sao poucos pontos e retas em qualquer
      angulo -- muito melhor para um veiculo que voa livre no plano.

    Camadas de navegacao:
      - DELIBERATIVO (esta classe): planeja com o mapa conhecido, antes de
        voar. Obstaculos estaticos (edificio, arvores, caixote).
      - REATIVO (_repulsion no drone): sensores em tempo real, para o que
        o mapa NAO sabe. As anomalias (cubos, pessoas) NAO entram aqui.

    Representacao:
      - Grid de ocupacao booleano `occ[linha][coluna]`, celula de `cell` m.
      - Obstaculos descritos como retangulos e circulos em coords do mundo.
      - Cada obstaculo e INFLADO por `inflate` (raio do drone + folga), de
        modo que um caminho por celulas livres ja garante a distancia de
        seguranca. E o truque padrao do "configuration space": infla o
        obstaculo e trata o robo como um ponto.

    Saida de plan(): lista de pontos (x, y) do mundo, do primeiro sub-alvo
    ate o objetivo exato. O drone persegue um ponto de cada vez.
    """

    def __init__(self, bounds, cell, rects, circles, inflate):
        self.xmin, self.xmax, self.ymin, self.ymax = bounds
        self.cell = cell
        self.cols = int((self.xmax - self.xmin) / cell) + 1
        self.rows = int((self.ymax - self.ymin) / cell) + 1
        # Pre-computa a ocupacao uma unica vez: o mapa e estatico, entao
        # nao ha motivo para testar obstaculo a cada consulta do A*.
        self.occ = [[self._blocked(*self._cw(c, r), rects, circles, inflate)
                     for c in range(self.cols)] for r in range(self.rows)]

    # --- conversoes celula <-> mundo ---
    def _cw(self, c, r):
        """Celula -> mundo (canto/centro logico da celula)."""
        return (self.xmin + c * self.cell, self.ymin + r * self.cell)

    def _wc(self, x, y):
        """Mundo -> celula (arredonda p/ a celula mais proxima)."""
        return (int(round((x - self.xmin) / self.cell)),
                int(round((y - self.ymin) / self.cell)))

    @staticmethod
    def _blocked(x, y, rects, circles, inf):
        """Ponto do mundo cai dentro de algum obstaculo JA INFLADO por
        `inf`? Inflar aqui = garantir a folga sem checar o corpo do drone
        depois (robo vira um ponto)."""
        for x0, x1, y0, y1 in rects:
            if x0 - inf <= x <= x1 + inf and y0 - inf <= y <= y1 + inf:
                return True
        for cx, cy, rad in circles:
            if math.hypot(x - cx, y - cy) <= rad + inf:
                return True
        return False

    def _free(self, c, r):
        """Celula existe no grid e nao esta ocupada."""
        return 0 <= c < self.cols and 0 <= r < self.rows and not self.occ[r][c]

    def _nearest_free(self, c, r):
        """Se start/goal caiu dentro da inflacao (ex.: waypoint colado num
        muro), procura em aneis crescentes a celula livre mais proxima.
        Sem isso o A* nao teria de onde partir / aonde chegar."""
        if self._free(c, r):
            return (c, r)
        for rad in range(1, 15):
            for dc in range(-rad, rad + 1):
                for dr in range(-rad, rad + 1):
                    if max(abs(dc), abs(dr)) == rad and self._free(c + dc, r + dr):
                        return (c + dc, r + dr)
        return (c, r)

    def _los(self, a, b):
        """LINHA DE VISADA -- o coracao do Theta*.
        Percorre as celulas do segmento a->b com Bresenham. Se todas estao
        livres, o drone pode voar reto de a ate b e o caminho pode cortar
        o intermediario. Se alguma esta ocupada, nao ha visada."""
        (c0, r0), (c1, r1) = a, b
        dc, dr = abs(c1 - c0), abs(r1 - r0)
        sc = 1 if c1 > c0 else -1
        sr = 1 if r1 > r0 else -1
        err = dc - dr
        c, r = c0, r0
        while True:
            if not self._free(c, r):
                return False
            if (c, r) == (c1, r1):
                return True
            e2 = 2 * err
            if e2 > -dr:
                err -= dr
                c += sc
            if e2 < dc:
                err += dc
                r += sr

    def _neighbors(self, node):
        """Os 8 vizinhos livres (inclui diagonais)."""
        c, r = node
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                if (dc or dr) and self._free(c + dc, r + dr):
                    yield (c + dc, r + dr)

    def _dist(self, a, b):
        """Distancia euclidiana em metros. Serve de custo g E de heuristica
        h (admissivel: nunca superestima, pois a reta e o menor caminho)."""
        return math.hypot(a[0] - b[0], a[1] - b[1]) * self.cell

    def plan(self, start, goal):
        """Theta* de start ate goal (coords do mundo).

        Estrutura A* padrao: fila de prioridade por f = g + h, onde
          g[n] = custo real acumulado ate n
          h[n] = distancia em linha reta de n ao objetivo (heuristica)
        A diferenca do Theta* esta no relaxamento das arestas (abaixo).

        Se nao existir rota, devolve [goal]: segue em linha reta e deixa a
        camada reativa (sensores) resolver -- degradacao segura."""
        s = self._nearest_free(*self._wc(*start))
        g = self._nearest_free(*self._wc(*goal))

        gscore = {s: 0.0}     # custo real conhecido ate cada no
        parent = {s: s}       # de quem cada no veio (define o caminho)
        openh = [(self._dist(s, g), s)]   # fila de prioridade por f
        closed = set()        # nos ja expandidos (nao reabrir)

        while openh:
            _, cur = heapq.heappop(openh)
            if cur == g:                  # objetivo alcancado
                break
            if cur in closed:             # entrada obsoleta do heap
                continue
            closed.add(cur)

            for nb in self._neighbors(cur):
                if nb in closed:
                    continue
                # --- relaxamento Theta* (o que o diferencia do A*) ---
                # Tenta pendurar o vizinho direto no AVO (pai do atual).
                # Se ha visada avo->vizinho, o caminho vira uma reta longa
                # e o no atual e descartado como intermediario inutil.
                p = parent[cur]
                if self._los(p, nb):
                    cand, ng = p, gscore[p] + self._dist(p, nb)
                else:
                    # sem visada: comporta-se como A* normal (via o atual)
                    cand, ng = cur, gscore[cur] + self._dist(cur, nb)

                # so aceita se for um caminho mais barato do que o conhecido
                if nb not in gscore or ng < gscore[nb]:
                    gscore[nb] = ng
                    parent[nb] = cand
                    heapq.heappush(openh, (ng + self._dist(nb, g), nb))

        if g not in parent:
            return [goal]                 # sem rota -> reta + camada reativa

        # Refaz o caminho de tras p/ frente seguindo os pais. Como o Theta*
        # ja "pulou" os intermediarios, sobram poucos pontos (os vertices).
        path = [g]
        while path[-1] != s:
            path.append(parent[path[-1]])
        path.reverse()

        pts = [self._cw(c, r) for c, r in path]
        pts[-1] = goal        # ultimo ponto = objetivo exato, nao o centro da celula
        # Descarta o primeiro (celula onde o drone JA esta), senao ele
        # tentaria voar de volta ao centro da propria celula.
        return pts[1:] if len(pts) > 1 else pts


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


class AnomalyRegister:
    """Registra anomalias detectadas (a "foto"): tempo, tipo, posicao
    estimada no mundo e pose do drone. Grava em arquivo e no console."""

    def __init__(self, path):
        self.path = path
        self._f = open(path, "w")
        self._f.write("# t tipo  wx wy wz  drone_x drone_y drone_z  yaw_deg range_m\n")

    def register(self, t, kind, world, drone_pose, rng, announce=True):
        wx, wy, wz = world
        dx, dy, dz, yaw = drone_pose
        rng_s = f"{rng:.2f}" if rng is not None else "NA"
        self._f.write(f"{t:.2f} {kind} {wx:.3f} {wy:.3f} {wz:.3f} "
                      f"{dx:.3f} {dy:.3f} {dz:.3f} {math.degrees(yaw):.1f} {rng_s}\n")
        self._f.flush()
        if announce:
            label = "CUBO VERMELHO" if kind == "red" else "PESSOA"
            print("=" * 60)
            print(f"[ANOMALIA] {label} registrada (t={t:.1f}s)")
            print(f"    Posicao estimada : x={wx:.2f} y={wy:.2f} z={wz:.2f}"
                  f"  (range={rng_s} m)")
            print(f"    Observado de     : x={dx:.2f} y={dy:.2f} z={dz:.2f}"
                  f"  yaw={math.degrees(yaw):+.1f} deg")
            print("=" * 60)

    def close(self):
        if self._f and not self._f.closed:
            self._f.flush()
            self._f.close()
            print(f"[I/O] Anomalias salvas em '{self.path}'.")


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

        # anel de sensores de distancia (desvio reativo). ds{i} a i*45 deg
        # no referencial do corpo (0 = frente, +CCW).
        self.range_sensors = []
        self.sensor_angles = []
        for i in range(SENSOR_COUNT):
            s = self.robot.getDevice(f"ds{i}")
            s.enable(self.timestep)
            self.range_sensors.append(s)
            self.sensor_angles.append(i * 2.0 * math.pi / SENSOR_COUNT)

        # --- motores (helices em modo velocidade) ---
        self.motors = [self.robot.getDevice(n) for n in
                       ("front left propeller", "front right propeller",
                        "rear left propeller",  "rear right propeller")]
        for m in self.motors:
            m.setPosition(float("inf"))
            m.setVelocity(1.0)

        self.cam_width = self.camera.getWidth()
        self.cam_height = self.camera.getHeight()
        self.cam_fov = self.camera.getFov()   # FOV horizontal (rad)

        # --- modulos ---
        self.vision = VisionDetector(self.camera)
        self.logger = TrajectoryLogger(LOG_FILE_PATH)
        self.anomalies = AnomalyRegister(ANOMALY_LOG_PATH)
        self.planner = Planner(GRID_BOUNDS, GRID_CELL, STATIC_RECTS,
                               STATIC_CIRCLES, OBSTACLE_INFLATE)
        self.keys = KeyboardListener(self.robot, self.timestep)

        self.target_yaw = None          # proa travada, definida no 1o passo
        self.last_alert = {"red": -1e9, "green": -1e9}
        self.sim_time = 0.0
        self.waypoint_index = 0         # waypoint atual da rota (modo PATROL)
        self.patrol_state = "CRUISE"    # CRUISE -> PAUSE -> CRUISE
        self.path = None                # caminho A* atual (lista de pontos xy)
        self.path_idx = 0               # ponto do caminho sendo perseguido
        self.pause_start = 0.0          # instante em que comecou a parada
        self.last_speed = 0.0           # |v| horizontal (maquina de estados)
        self.last_yaw_err = 0.0         # erro de proa atual
        self.prev_alt = None            # altitude anterior (p/ estimar vz)
        self.prev_xy = None             # (x,y) anterior (p/ estimar vx,vy)
        self.hover_xy = None            # ponto de decolagem travado (HOVER)
        self.target_alt = TARGET_ALTITUDE  # altitude alvo atual (por waypoint)
        self.debug_inputs = (0.0, 0.0, 0.0, 0.0)  # roll/pitch/yaw/vert p/ log
        self.telemetry = Telemetry(TELEMETRY_HOST, TELEMETRY_PORT,
                                   enable=TELEMETRY_ENABLE)

        # --- mission FSM: PATROL / INSPECT (cubo) / FOLLOW (pessoa) ---
        self.mission = "PATROL"
        self.anomaly_cooldown = {"red": -1e9, "green": -1e9}
        self.state_start = 0.0          # inicio do estado INSPECT/FOLLOW
        self.inspect_anchor = None      # posicao travada durante o INSPECT
        self.approach_kind = None       # tipo de anomalia sendo aproximada
        self.last_register = 0.0        # ultimo registro continuo (FOLLOW)
        self.last_seen = 0.0            # ultima vez que viu a pessoa (FOLLOW)

    # -----------------------------------------------------------
    def _repulsion(self, altitude):
        """Campo potencial repulsivo a partir do anel de sensores.
        Retorna (a_fwd, a_left) no referencial do corpo e a menor leitura.
        Cada sensor empurra na direcao oposta ao obstaculo que ve."""
        rep_fwd = rep_left = 0.0
        min_range = REACT_RANGE
        # so no ar: abaixo de REACT_MIN_ALT ignora os sensores (decolagem)
        if altitude < REACT_MIN_ALT:
            return rep_fwd, rep_left, min_range
        for sensor, angle in zip(self.range_sensors, self.sensor_angles):
            d = sensor.getValue()
            if d < min_range:
                min_range = d
            if d >= REACT_RANGE:
                continue
            # mais forte perto de SAFE_DISTANCE; saturado abaixo dele
            dd = max(d, SAFE_DISTANCE * 0.6)
            mag = K_REPULSION * (1.0 / dd - 1.0 / REACT_RANGE)
            rep_fwd -= mag * math.cos(angle)   # empurra p/ longe do sensor
            rep_left -= mag * math.sin(angle)
        return rep_fwd, rep_left, min_range

    # -----------------------------------------------------------
    def compute_motor_commands(self, target=None, face_point=None, target_alt=None):
        """Roda o controle e aciona os 4 motores. Retorna a dist ao alvo.

        target=None      -> segura o ponto de decolagem (hover).
        target=(tx,ty)   -> vai ate o ponto.
        face_point=(x,y) -> encara este ponto (senao encara o 'target').
        target_alt       -> altitude alvo (m); None mantem a ultima."""
        if target_alt is not None:
            self.target_alt = target_alt
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
        if target is None:
            hold_x, hold_y = self.hover_xy
            desired_yaw = self.target_yaw
        else:
            hold_x, hold_y = target
            desired_yaw = math.atan2(hold_y - y, hold_x - x)
        # face_point sobrepoe a proa (usado no reorientar em waypoint)
        if face_point is not None:
            desired_yaw = math.atan2(face_point[1] - y, face_point[0] - x)
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
        # desvio reativo: soma a repulsao dos sensores (seguranca)
        rep_fwd, rep_left, min_range = self._repulsion(altitude)
        a_fwd += rep_fwd
        a_left += rep_left
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
        clamped_diff_alt = clamp(self.target_alt - altitude + K_VERTICAL_OFFSET,
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
        # bias fixo p/ baixo + estabilizacao: sem o bias a camera olha no
        # horizonte e nunca ve as anomalias no chao.
        self.cam_pitch.setPosition(CAM_PITCH_DOWN - 0.1 * pitch_rate)

        self.debug_inputs = (roll_input, pitch_input, yaw_input, vertical_input)
        self.last_speed = math.hypot(vx, vy)     # p/ a maquina de estados
        self.last_yaw_err = yaw_err
        # snapshot completo p/ telemetria/plotter
        self.telemetry.send({
            "t": round(self.sim_time, 3),
            "altitude": altitude, "target_altitude": self.target_alt, "vz": vz,
            "roll": roll, "pitch": pitch, "yaw": yaw,
            "yaw_err": yaw_err, "desired_yaw": desired_yaw,
            "roll_rate": roll_rate, "pitch_rate": pitch_rate, "yaw_rate": yaw_rate,
            "roll_input": roll_input, "pitch_input": pitch_input,
            "yaw_input": yaw_input, "vertical_input": vertical_input,
            "x": x, "y": y, "distance": distance,
            "vx": vx, "vy": vy, "v_fwd": v_fwd, "v_left": v_left,
            "pitch_dist": pitch_disturbance, "roll_dist": roll_disturbance,
            "min_range": min_range,
            "fl": fl, "fr": fr, "rl": rl, "rr": rr,
        })

        return distance

    # -----------------------------------------------------------
    def patrol_step(self):
        """Maquina de estados da patrulha.
        CRUISE: segue o caminho A* (Theta*) ate o waypoint, desviando dos
                obstaculos conhecidos.
        PAUSE : chegou -> freia, gira para o proximo waypoint e espera
                (WAYPOINT_PAUSE s alinhado) antes de seguir."""
        n = len(WAYPOINTS)
        cur = WAYPOINTS[self.waypoint_index]
        nxt = WAYPOINTS[(self.waypoint_index + 1) % n]
        cur_xy, cur_alt = (cur[0], cur[1]), cur[2]
        nxt_xy = (nxt[0], nxt[1])
        x, y, _ = self.gps.getValues()

        if self.patrol_state == "CRUISE":
            # planeja uma vez por trecho (obstaculos estaticos)
            if self.path is None:
                self.path = self.planner.plan((x, y), cur_xy)
                self.path_idx = 0
                print(f"[A*] Rota p/ waypoint {self.waypoint_index} {cur_xy}: "
                      f"{len(self.path)} pontos.")
            sub = self.path[self.path_idx]
            dist_sub = self.compute_motor_commands(sub, target_alt=cur_alt)
            if self.path_idx >= len(self.path) - 1:      # ultimo ponto = waypoint
                arrived = (dist_sub < WAYPOINT_TOLERANCE and
                           self.last_speed < WAYPOINT_ARRIVE_SPEED)
                if arrived:
                    self.patrol_state = "PAUSE"
                    self.pause_start = self.sim_time
                    self.path = None
                    print(f"[NAV] Chegou ao waypoint {self.waypoint_index} {cur} -> parando.")
            elif dist_sub < SUBWP_TOLERANCE:             # avanca no caminho
                self.path_idx += 1
        else:  # PAUSE: segura a posicao no waypoint e encara o proximo
            self.compute_motor_commands(cur_xy, face_point=nxt_xy, target_alt=cur_alt)
            waited = self.sim_time - self.pause_start >= WAYPOINT_PAUSE
            aligned = abs(self.last_yaw_err) < WAYPOINT_YAW_TOL
            if waited and aligned:
                self.waypoint_index = (self.waypoint_index + 1) % n
                self.patrol_state = "CRUISE"
                self.path = None                         # replaneja o proximo trecho
                print(f"[NAV] Seguindo p/ waypoint {self.waypoint_index} "
                      f"{WAYPOINTS[self.waypoint_index]}.")

    # -----------------------------------------------------------
    # Localizacao do alvo a partir da imagem
    def _bearing(self, info, yaw):
        """Azimute (mundo) do alvo a partir do deslocamento em u na imagem."""
        offset = (info["u"] / self.cam_width - 0.5) * self.cam_fov
        return normalize_angle(yaw - offset)

    def _bearing_point(self, info, x, y, yaw):
        """Ponto distante na direcao do alvo (usado como face_point)."""
        b = self._bearing(info, yaw)
        return (x + 50.0 * math.cos(b), y + 50.0 * math.sin(b))

    def _range_to_target(self, info, altitude, kind):
        """Alcance horizontal (m) ao alvo pela HIPOTESE DE PLANO DE CHAO.

        Uma camera unica nao mede profundidade. Mas as anomalias estao no
        chao, numa altura conhecida (TARGET_HEIGHT). Entao:
          - a linha do pixel v da o angulo de depressao abaixo da horizontal
            (bias fixo da camera + deslocamento vertical do pixel);
          - com a altura do drone acima do alvo, o triangulo fecha:
                alcance = (z_drone - z_alvo) / tan(depressao)
        Devolve None se o alvo esta na horizontal ou acima (alcance ~ inf).
        """
        vfov = 2.0 * math.atan(math.tan(self.cam_fov / 2.0) *
                               self.cam_height / self.cam_width)
        # v cresce p/ baixo na imagem -> angulo positivo = abaixo do eixo
        ang_v = (info["v"] / self.cam_height - 0.5) * vfov
        depression = CAM_PITCH_DOWN + ang_v
        dz = altitude - TARGET_HEIGHT.get(kind, 0.0)
        if depression <= 0.05 or dz <= 0.1:
            return None                      # alvo no horizonte -> muito longe
        return dz / math.tan(depression)

    def _estimate_pos(self, info, x, y, z, yaw, kind):
        """Posicao estimada do alvo no mundo + alcance (m).
        Bearing (do pixel u) + alcance (plano de chao) = posicao 2D."""
        b = self._bearing(info, yaw)
        rng = self._range_to_target(info, z, kind)
        d = rng if rng is not None else REGISTER_NOMINAL_RANGE
        return (x + d * math.cos(b), y + d * math.sin(b),
                TARGET_HEIGHT.get(kind, 0.0)), rng

    # Mission FSM
    def mission_step(self, det):
        """Despacha PATROL -> APPROACH -> INSPECT (cubo) / FOLLOW (pessoa).

        APPROACH e comum aos dois tipos: so vale registrar de perto, entao
        o drone primeiro chega a <= APPROACH_RANGE do alvo."""
        x, y, z = self.gps.getValues()
        _, _, yaw = self.imu.getRollPitchYaw()

        if self.mission == "PATROL":
            trig = self._check_trigger(det)
            if trig is None:
                self.patrol_step()
                return
            self.mission = "APPROACH"
            self.approach_kind = trig
            self.state_start = self.sim_time
            self.last_seen = self.sim_time
            label = "Pessoa" if trig == "green" else "Cubo vermelho"
            print(f"[MISSION] {label} detectado -> aproximando "
                  f"(ate {APPROACH_RANGE:.0f} m).")

        if self.mission == "APPROACH":
            self._approach_step(det, x, y, z, yaw)
        elif self.mission == "INSPECT":
            self._inspect_step(det, x, y, z, yaw)
        elif self.mission == "FOLLOW":
            self._follow_step(det, x, y, z, yaw)

    def _approach_step(self, det, x, y, z, yaw):
        """Voa em direcao a anomalia ate ficar a <= APPROACH_RANGE dela.
        So entao passa p/ INSPECT (cubo) ou FOLLOW (pessoa)."""
        kind = self.approach_kind
        info = det.get(kind) if det else None
        seen = info and info["area"] >= MIN_BLOB_AREA

        if seen:
            self.last_seen = self.sim_time
            face = self._bearing_point(info, x, y, yaw)
            rng = self._range_to_target(info, z, kind)
            if rng is not None and rng <= APPROACH_RANGE:
                # chegou perto o bastante -> comportamento por tipo
                self.state_start = self.sim_time
                if kind == "red":
                    self.mission = "INSPECT"
                    self.inspect_anchor = (x, y)
                    print(f"[MISSION] A {rng:.1f} m do cubo -> inspecionando.")
                else:
                    self.mission = "FOLLOW"
                    self.last_register = -1e9
                    print(f"[MISSION] A {rng:.1f} m da pessoa -> seguindo.")
                self.compute_motor_commands(target=(x, y), face_point=face)
                return
            # ainda longe: persegue a posicao estimada do alvo
            world, _ = self._estimate_pos(info, x, y, z, yaw, kind)
            self.compute_motor_commands(target=(world[0], world[1]), face_point=face)
        else:
            self.compute_motor_commands(target=(x, y))   # perdeu: segura

        elapsed = self.sim_time - self.state_start
        lost = self.sim_time - self.last_seen > APPROACH_LOST_TIMEOUT
        if elapsed > APPROACH_TIMEOUT or lost:
            reason = "perdeu o alvo" if lost else "tempo esgotado"
            print(f"[MISSION] Aproximacao abortada ({reason}).")
            self._resume_patrol(kind)

    def _check_trigger(self, det):
        """Retorna 'green'/'red' se ha anomalia acima do limiar e fora do
        cooldown (pessoa tem prioridade); senao None."""
        if det is None:
            return None
        for color in ("green", "red"):
            info = det.get(color)
            if (info and info["area"] >= MIN_BLOB_AREA and
                    self.sim_time - self.anomaly_cooldown[color] >= ANOMALY_COOLDOWN):
                return color
        return None

    def _inspect_step(self, det, x, y, z, yaw):
        """Cubo: para, encara, registra a "foto", retoma."""
        info = det.get("red") if det else None
        seen = info and info["area"] >= MIN_BLOB_AREA
        if seen:
            face = self._bearing_point(info, x, y, yaw)
            self.compute_motor_commands(target=self.inspect_anchor, face_point=face)
            if abs(self.last_yaw_err) < INSPECT_CENTER_TOL:      # centrado
                world, rng = self._estimate_pos(info, x, y, z, yaw, "red")
                self.anomalies.register(self.sim_time, "red", world, (x, y, z, yaw), rng)
                self._resume_patrol("red")
                return
        else:
            self.compute_motor_commands(target=self.inspect_anchor)

        if self.sim_time - self.state_start > INSPECT_TIMEOUT:
            world, rng = (self._estimate_pos(info, x, y, z, yaw, "red")
                          if seen else ((x, y, 0.0), None))
            self.anomalies.register(self.sim_time, "red", world, (x, y, z, yaw), rng)
            print("[MISSION] Inspecao esgotou o tempo -> registrando e seguindo.")
            self._resume_patrol("red")

    def _follow_step(self, det, x, y, z, yaw):
        """Pessoa: aproxima e segue ate 10s (ou ate perde-la), registrando."""
        info = det.get("green") if det else None
        seen = info and info["area"] >= MIN_BLOB_AREA
        if seen:
            self.last_seen = self.sim_time
            face = self._bearing_point(info, x, y, yaw)
            b = self._bearing(info, yaw)
            tgt = (x + FOLLOW_APPROACH * math.cos(b),
                   y + FOLLOW_APPROACH * math.sin(b))
            self.compute_motor_commands(target=tgt, face_point=face)
            if self.sim_time - self.last_register >= REGISTER_PERIOD:
                self.last_register = self.sim_time
                world, rng = self._estimate_pos(info, x, y, z, yaw, "green")
                self.anomalies.register(self.sim_time, "green", world, (x, y, z, yaw), rng)
        else:
            self.compute_motor_commands(target=(x, y))   # perdeu: segura no lugar

        elapsed = self.sim_time - self.state_start
        lost = self.sim_time - self.last_seen > FOLLOW_LOST_TIMEOUT
        if elapsed >= FOLLOW_DURATION or lost:
            reason = "10s concluidos" if elapsed >= FOLLOW_DURATION else "pessoa perdida"
            print(f"[MISSION] Follow encerrado ({reason}).")
            self._resume_patrol("green")

    def _resume_patrol(self, color):
        self.anomaly_cooldown[color] = self.sim_time
        self.mission = "PATROL"
        self.patrol_state = "CRUISE"
        self.path = None                # replaneja a partir da posicao atual
        print(f"[MISSION] Retomando patrulha (cooldown {color} {ANOMALY_COOLDOWN:.0f}s).")

    # -----------------------------------------------------------
    def run_vision(self, det):
        """Alerta simples de monitoramento (usado no modo HOVER)."""
        if det is None:
            return
        _, _, yaw = self.imu.getRollPitchYaw()
        x, y, z = self.gps.getValues()
        labels = {"red": "Objeto suspeito (vermelho)", "green": "INTRUSO (pedestre)"}
        for color in ("green", "red"):
            info = det[color]
            if info["area"] >= MIN_BLOB_AREA and \
                    self.sim_time - self.last_alert[color] >= ALERT_COOLDOWN:
                self.last_alert[color] = self.sim_time
                print(f"[ALERTA] {labels[color]} @ img(u={info['u']:.0f},v={info['v']:.0f}) "
                      f"| GPS x={x:.2f} y={y:.2f} z={z:.2f} yaw={math.degrees(yaw):+.1f}")

    # -----------------------------------------------------------
    def run(self):
        print(f"=== Sistema de Vigilancia iniciado (Mavic 2 Pro | modo {MODE}) ===")
        print("Pressione 'F' na janela 3D para salvar o log e encerrar.\n")
        log_every = max(1, int(0.5 / self.dt))   # log de PID a cada ~0.5 s
        step = 0

        while self.robot.step(self.timestep) != -1:
            self.sim_time += self.dt
            step += 1

            # --- Etapa 2: visao (uma varredura por passo) ---
            det = self.vision.scan()

            # --- Etapa 1: controle + navegacao + anomalias ---
            if MODE == "PATROL":
                self.mission_step(det)
            else:
                self.compute_motor_commands()          # hover
                self.run_vision(det)

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
        self.anomalies.close()
        self.telemetry.close()


if __name__ == "__main__":
    VigilanceDrone().run()
