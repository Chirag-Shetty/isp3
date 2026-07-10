"""
mmWave Radar Replay Visualizer
Matches the style of the TI IWR6843AOP visualizer.
Uses PyQtGraph OpenGL (same stack as original) for smooth, GPU-accelerated rendering.

Run:  python mmwave_visualizer.py [optional_replay.json]
Deps: pip install pyqtgraph PyOpenGL PyQt5 numpy
"""

import sys, json, math, random
import numpy as np

from PyQt5.QtCore    import Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                              QVBoxLayout, QLabel, QPushButton, QSlider,
                              QFileDialog, QGroupBox, QFormLayout, QComboBox,
                              QCheckBox, QSizePolicy)
from PyQt5.QtGui     import QFont

import pyqtgraph as pg
import pyqtgraph.opengl as gl

# ─── Kelly 22 Colors of Max Contrast (same LUT as original visualizer) ───────
_KELLY_RGB = [
    (255,179,0),(128,62,117),(255,104,0),(166,189,215),(193,0,32),
    (206,162,98),(129,112,102),(0,125,52),(246,118,142),(0,83,138),
    (255,122,92),(83,55,122),(255,142,0),(179,40,81),(244,200,0),
    (127,24,13),(147,170,0),(89,51,21),(241,58,19),(35,44,22),(0,161,194),
]

def get_track_colors(n):
    pool = [tuple(v/255 for v in c) + (1.0,) for c in _KELLY_RGB]
    out  = []
    for i in range(n):
        if i < len(pool):
            out.append(pool[i])
        else:
            r1,g1,b1,_ = pool[random.randint(0,len(pool)-1)]
            r2,g2,b2,_ = pool[random.randint(0,len(pool)-1)]
            nc = ((r1+r2)/2,(g1+g2)/2,(b1+b2)/2,1.0)
            pool.append(nc); out.append(nc)
    return out

TRACK_COLORS = get_track_colors(50)

# ─── Geometry helpers ─────────────────────────────────────────────────────────
def get_box_lines(x, y, z, xr=0.25, yr=0.25, zr=0.5):
    xl,xR = x-xr, x+xr
    yl,yR = y-yr, y+yr
    zl,zR = z-zr, z+zr
    v = np.array([[xl,yl,zl],[xR,yl,zl],[xl,yR,zl],[xR,yR,zl],
                  [xl,yl,zR],[xR,yl,zR],[xl,yR,zR],[xR,yR,zR]])
    pairs = [(0,1),(0,2),(0,4),(3,1),(3,2),(3,7),
             (5,4),(5,7),(5,1),(6,2),(6,4),(6,7)]
    lines = np.empty((len(pairs)*2,3))
    for i,(a,b) in enumerate(pairs):
        lines[i*2]=v[a]; lines[i*2+1]=v[b]
    return lines

def get_boundary_lines(xmin,xmax,ymin,ymax,zmin,zmax):
    v = np.array([[xmin,ymin,zmin],[xmax,ymin,zmin],[xmin,ymax,zmin],[xmax,ymax,zmin],
                  [xmin,ymin,zmax],[xmax,ymin,zmax],[xmin,ymax,zmax],[xmax,ymax,zmax]])
    pairs = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    lines = np.empty((len(pairs)*2,3))
    for i,(a,b) in enumerate(pairs):
        lines[i*2]=v[a]; lines[i*2+1]=v[b]
    return lines

# ─── JSON helpers ─────────────────────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)

def parse_boundary(cfg_lines):
    for line in cfg_lines:
        s = line.strip()
        if s.startswith("boundaryBox"):
            p = s.split()
            if len(p)==7:
                try: return [float(x) for x in p[1:]]
                except: pass
    return [-4,4,0,8,0,3]

def parse_frame(fd):
    pts        = np.array(fd.get("pointCloud",[]), dtype=float)
    tidxs      = np.array(fd.get("trackIndexes",[]), dtype=float)
    tracks     = fd.get("trackData",[])
    num_humans = int(fd.get("numDetectedTracks",0))
    frame_num  = fd.get("frameNum",0)
    heights    = fd.get("heightData",[])
    return pts, tidxs, tracks, num_humans, frame_num, heights


# ─── Main Window ─────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    MAX_TRACKS = 50

    def __init__(self):
        super().__init__()
        self.setWindowTitle("mmWave Radar Replay Visualizer")
        self.resize(1100, 750)

        pg.setConfigOption('background', pg.mkColor(70,72,79))
        pg.setConfigOption('foreground', 'w')
        self._apply_dark_stylesheet()

        self.frames    = []
        self.frame_idx = 0
        self.playing   = False
        self.bounds    = None

        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer)

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root_lay = QHBoxLayout(central)
        root_lay.setSpacing(6); root_lay.setContentsMargins(6,6,6,6)

        # ── Left panel ────────────────────────────────────────────────────────
        left = QWidget(); left.setFixedWidth(215)
        ll   = QVBoxLayout(left)
        ll.setSpacing(8); ll.setContentsMargins(0,0,0,0)
        root_lay.addWidget(left)

        # File
        fb = QGroupBox("File"); fl = QVBoxLayout(fb)
        self.btn_open = QPushButton("📂  Open JSON")
        self.btn_open.clicked.connect(self.open_file)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setWordWrap(True)
        self.lbl_file.setStyleSheet("color:#aaa;font-size:10px;")
        fl.addWidget(self.btn_open); fl.addWidget(self.lbl_file)
        ll.addWidget(fb)

        # ── Human count badge (fixed: smaller font + minHeight + wordWrap) ──
        self.lbl_humans = QLabel("👤  Humans: –")
        self.lbl_humans.setAlignment(Qt.AlignCenter)
        self.lbl_humans.setFont(QFont("Helvetica", 13, QFont.Bold))
        self.lbl_humans.setMinimumHeight(50)
        self.lbl_humans.setWordWrap(True)
        self.lbl_humans.setStyleSheet(
            "background:#3a3a3a;color:#888;border-radius:6px;padding:8px;")
        ll.addWidget(self.lbl_humans)

        # Track info
        ib = QGroupBox("Track Info")
        self.ti_lay = QVBoxLayout(ib)
        self.track_labels = []
        for _ in range(10):
            lbl = QLabel(); lbl.setStyleSheet("color:#ccc;font-size:10px;")
            lbl.hide(); self.track_labels.append(lbl); self.ti_lay.addWidget(lbl)
        ll.addWidget(ib)

        # Controls
        cb = QGroupBox("Controls"); ctl = QFormLayout(cb)
        self.color_mode = QComboBox()
        self.color_mode.addItems(["Track","SNR","Height","Doppler"])
        self.color_mode.currentIndexChanged.connect(self._redraw_current)
        ctl.addRow("Color by:", self.color_mode)
        self.chk_tracks = QCheckBox("Show track boxes"); self.chk_tracks.setChecked(True)
        self.chk_tracks.stateChanged.connect(self._redraw_current)
        ctl.addRow(self.chk_tracks)
        ll.addWidget(cb)
        ll.addStretch()

        # ── Right: GL view + playback bar ─────────────────────────────────────
        right = QWidget(); rl = QVBoxLayout(right)
        rl.setSpacing(4); rl.setContentsMargins(0,0,0,0)
        root_lay.addWidget(right, 1)

        self.lbl_frame = QLabel("Frame – / –")
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        self.lbl_frame.setStyleSheet("color:#aaa;font-size:11px;")
        rl.addWidget(self.lbl_frame)

        # GL Widget
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(pg.mkColor(70,72,79))
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view.setCameraPosition(distance=10, elevation=30, azimuth=45)
        rl.addWidget(self.view, 1)

        # Grid
        self.gz = gl.GLGridItem()
        self.gz.setSize(x=8,y=8); self.gz.setSpacing(x=1,y=1)
        self.view.addItem(self.gz)

        # Scatter
        self.scatter = gl.GLScatterPlotItem(size=5, pxMode=True)
        self.scatter.setData(pos=np.zeros((1,3)), color=(1,1,1,0))
        self.view.addItem(self.scatter)

        # Pre-create track box line items
        self.track_boxes = []
        for _ in range(self.MAX_TRACKS):
            item = gl.GLLinePlotItem(width=2, antialias=True, mode='lines')
            item.hide(); self.view.addItem(item)
            self.track_boxes.append(item)

        # Boundary box
        self.boundary_item = gl.GLLinePlotItem(
            color=(0.5,0.5,1.0,0.4), width=1, antialias=True, mode='lines')
        self.view.addItem(self.boundary_item)

        # ── Playback bar ──────────────────────────────────────────────────────
        bar = QWidget(); bl = QHBoxLayout(bar)
        bl.setContentsMargins(4,2,4,2); bl.setSpacing(4)

        def mk(txt, cb, w=36):
            b=QPushButton(txt); b.setFixedWidth(w)
            b.clicked.connect(cb); return b

        bl.addWidget(mk("⏮",self.go_first))
        bl.addWidget(mk("◀",self.step_back))
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(50)
        self.btn_play.setObjectName("btnPlay")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        bl.addWidget(self.btn_play)
        bl.addWidget(mk("▶",self.step_fwd))
        bl.addWidget(mk("⏭",self.go_last))

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0); self.slider.setMaximum(1)
        self.slider.sliderMoved.connect(self._slider_moved)
        bl.addWidget(self.slider,1)

        bl.addWidget(QLabel("Speed:"))
        self.speed_sl = QSlider(Qt.Horizontal)
        self.speed_sl.setMinimum(1); self.speed_sl.setMaximum(20)
        self.speed_sl.setValue(10); self.speed_sl.setFixedWidth(80)
        bl.addWidget(self.speed_sl)

        rl.addWidget(bar)

    # ── File loading ──────────────────────────────────────────────────────────
    def open_file(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self,"Open Replay JSON","","JSON (*.json);;All (*)")
        if not path: return
        try: raw = load_json(path)
        except Exception as e:
            self.lbl_file.setText(f"Error: {e}"); return

        self.frames = raw.get("data",[])
        self.bounds = parse_boundary(raw.get("cfg",[]))

        # Boundary wireframe
        self.boundary_item.setData(
            pos=get_boundary_lines(*self.bounds), color=(0.5,0.5,1.0,0.4))

        # Refit grid to boundary floor
        xsz = self.bounds[1]-self.bounds[0]
        ysz = self.bounds[3]-self.bounds[2]
        cx  = (self.bounds[0]+self.bounds[1])/2
        cy  = (self.bounds[2]+self.bounds[3])/2
        self.gz.resetTransform()
        self.gz.translate(cx, cy, self.bounds[4])
        self.gz.setSize(x=xsz, y=ysz)

        self.frame_idx = 0
        self.slider.setMaximum(max(1,len(self.frames)-1))
        self.slider.setValue(0)
        self.btn_play.setEnabled(True)
        self.lbl_file.setText(path.replace("\\","/").split("/")[-1])
        self.render_frame(0)

    # ── Rendering ─────────────────────────────────────────────────────────────
    def render_frame(self, idx):
        if not self.frames: return
        idx = max(0, min(idx, len(self.frames)-1))
        self.frame_idx = idx

        fd = self.frames[idx]["frameData"]
        pts, tidxs, tracks, num_humans, frame_num, heights = parse_frame(fd)

        # Point cloud
        if len(pts) > 0:
            xyz  = pts[:,0:3]
            snrs = pts[:,4] if pts.shape[1]>4 else np.ones(len(pts))*10
            with np.errstate(divide='ignore',invalid='ignore'):
                sizes = np.clip(np.log2(np.maximum(snrs,1)), 2, 12)
            colors = self._point_colors(pts, tidxs)
            self.scatter.setData(pos=xyz, color=colors, size=sizes)
        else:
            self.scatter.setData(pos=np.zeros((1,3)), color=(1,1,1,0))

        # Hide all boxes
        for tb in self.track_boxes: tb.hide()

        # Height lookup
        h_map = {int(h[0]):round(h[1],2) for h in heights if len(h)>=2}

        # Clear track labels
        for lbl in self.track_labels: lbl.hide()

        if self.chk_tracks.isChecked():
            for i,t in enumerate(tracks):
                if len(t)<4: continue
                tid = int(t[0]); x,y,z = t[1],t[2],t[3]
                col = TRACK_COLORS[tid % len(TRACK_COLORS)]
                if tid < len(self.track_boxes):
                    self.track_boxes[tid].setData(
                        pos=get_box_lines(x,y,z), color=col, width=2,
                        antialias=True, mode='lines')
                    self.track_boxes[tid].show()
                if i < len(self.track_labels):
                    ht  = h_map.get(tid)
                    hex_col = "#{:02x}{:02x}{:02x}".format(
                        int(col[0]*255),int(col[1]*255),int(col[2]*255))
                    txt = f"T{tid}: ({x:.2f},{y:.2f},{z:.2f})"
                    if ht: txt += f"  h={ht}m"
                    self.track_labels[i].setText(txt)
                    self.track_labels[i].setStyleSheet(
                        f"color:{hex_col};font-size:10px;")
                    self.track_labels[i].show()

        # Human badge
        if num_humans>0:
            self.lbl_humans.setText(f"👤  Humans: {num_humans}")
            self.lbl_humans.setStyleSheet(
                "background:#2a9d8f;color:white;border-radius:6px;padding:8px;")
        else:
            self.lbl_humans.setText("👤  Humans: 0")
            self.lbl_humans.setStyleSheet(
                "background:#3a3a3a;color:#888;border-radius:6px;padding:8px;")

        self.lbl_frame.setText(
            f"Frame {frame_num}   [{idx+1} / {len(self.frames)}]")
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)

    def _point_colors(self, pts, tidxs):
        n = len(pts)
        cols = np.ones((n,4), dtype=float)
        mode = self.color_mode.currentText()

        if mode=="Track":
            for i in range(n):
                ti = int(tidxs[i]) if i<len(tidxs) else 255
                cols[i] = (1,1,1,0.35) if ti in (253,254,255) \
                           else TRACK_COLORS[ti % len(TRACK_COLORS)]
        elif mode=="SNR":
            snrs = pts[:,4] if pts.shape[1]>4 else np.ones(n)
            t = np.clip((snrs-5.0)/45.0, 0, 1)
            cols[:,0]=t; cols[:,1]=0.4+0.6*(1-t); cols[:,2]=1-t; cols[:,3]=0.9
        elif mode=="Height":
            zs = pts[:,2]
            zmin = self.bounds[4] if self.bounds else 0
            zmax = self.bounds[5] if self.bounds else 3
            t = np.clip((zs-zmin)/max(zmax-zmin,0.01), 0, 1)
            cols[:,0]=t; cols[:,1]=1-t; cols[:,2]=0.5; cols[:,3]=0.9
        elif mode=="Doppler":
            dop = pts[:,3] if pts.shape[1]>3 else np.zeros(n)
            t = np.clip((dop+2.0)/4.0, 0, 1)
            cols[:,0]=t; cols[:,1]=0.3; cols[:,2]=1-t; cols[:,3]=0.9
        return cols

    def _redraw_current(self):
        self.render_frame(self.frame_idx)

    # ── Playback ──────────────────────────────────────────────────────────────
    def toggle_play(self):
        if self.playing:
            self.playing=False; self.timer.stop()
            self.btn_play.setText("▶")
            self.btn_play.setStyleSheet(
                "background:#2a9d8f;color:white;border-radius:4px;font-size:14px;")
        else:
            self.playing=True
            self.btn_play.setText("⏸")
            self.btn_play.setStyleSheet(
                "background:#e76f51;color:white;border-radius:4px;font-size:14px;")
            self.timer.start(max(16, 500//max(1,self.speed_sl.value())))

    def _on_timer(self):
        self.timer.setInterval(max(16, 500//max(1,self.speed_sl.value())))
        nxt = self.frame_idx+1
        if nxt>=len(self.frames): nxt=0
        self.render_frame(nxt)

    def step_back(self):  self.render_frame(self.frame_idx-1)
    def step_fwd(self):   self.render_frame(self.frame_idx+1)
    def go_first(self):   self.render_frame(0)
    def go_last(self):    self.render_frame(len(self.frames)-1)
    def _slider_moved(self,v): self.render_frame(v)

    # ── Stylesheet ────────────────────────────────────────────────────────────
    def _apply_dark_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#2b2b2b;color:#ddd;}
            QGroupBox{border:1px solid #555;border-radius:4px;margin-top:8px;
                      font-size:11px;color:#aaa;padding:6px;}
            QGroupBox::title{subcontrol-origin:margin;left:8px;color:#aaa;}
            QPushButton{background:#444;color:white;border-radius:4px;padding:4px 8px;}
            QPushButton:hover{background:#555;}
            QPushButton#btnPlay{background:#2a9d8f;font-size:14px;}
            QSlider::groove:horizontal{background:#444;height:4px;border-radius:2px;}
            QSlider::handle:horizontal{background:#2a9d8f;width:12px;height:12px;
                                       margin:-4px 0;border-radius:6px;}
            QComboBox{background:#3a3a3a;color:#ddd;border:1px solid #555;
                      border-radius:3px;padding:2px;}
            QCheckBox{color:#ccc;}
        """)


# ─── Entry ────────────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    if len(sys.argv)>1:
        import os
        if os.path.isfile(sys.argv[1]): win.open_file(sys.argv[1])
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()