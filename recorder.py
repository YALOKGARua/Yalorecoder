import ctypes
import customtkinter as ctk
import soundfile as sf
import numpy as np
import threading
import time
import os
import sys
import json
import subprocess
import urllib.request
import urllib.error
from datetime import datetime

try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("yalokgar.audiorecorder.1.0")
except Exception:
    pass

APP_VERSION = "1.1.1"
GITHUB_REPO = "YALOKGARua/Yalorecoder"

try:
    import pyaudiowpatch as pyaudio
    AUDIO_OK = True
except ImportError:
    pyaudio = None
    AUDIO_OK = False

HAS_TRAY = False
try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    pass


def _resample(data, src_rate, dst_rate):
    if src_rate == dst_rate or len(data) == 0:
        return data
    ratio = dst_rate / src_rate
    n_out = int(len(data) * ratio)
    x_src = np.linspace(0, 1, len(data))
    x_dst = np.linspace(0, 1, n_out)
    out = np.zeros((n_out, data.shape[1]), dtype=np.float32)
    for ch in range(data.shape[1]):
        out[:, ch] = np.interp(x_dst, x_src, data[:, ch])
    return out


def _sanitize_filename(raw):
    allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _-')
    cleaned = ''.join(c if c in allowed else '_' for c in raw).strip()
    return cleaned[:80] if cleaned else ''


def _strip_silence(audio, rate, threshold_db=-40, min_gap_ms=400, pad_ms=50):
    chunk_len = int(rate * 0.02)
    if chunk_len == 0 or len(audio) < chunk_len:
        return audio

    threshold = 10 ** (threshold_db / 20.0)
    n_chunks = len(audio) // chunk_len
    pad_n = max(1, int(pad_ms / 20))
    gap_n = max(1, int(min_gap_ms / 20))

    rms_arr = np.array([
        np.sqrt(np.mean(audio[i * chunk_len:(i + 1) * chunk_len] ** 2))
        for i in range(n_chunks)
    ])

    voiced = rms_arr > threshold
    expanded = voiced.copy()

    for i in np.where(voiced)[0]:
        lo = max(0, i - pad_n)
        hi = min(len(expanded), i + pad_n + 1)
        expanded[lo:hi] = True

    silence_start = None
    for i in range(len(expanded)):
        if not expanded[i]:
            if silence_start is None:
                silence_start = i
        else:
            if silence_start is not None:
                run_len = i - silence_start
                if run_len < gap_n:
                    expanded[silence_start:i] = True
                silence_start = None

    parts = []
    for i in range(n_chunks):
        if expanded[i]:
            a = i * chunk_len
            b = min(a + chunk_len, len(audio))
            parts.append(audio[a:b])

    return np.concatenate(parts) if parts else audio


class AutoUpdater:

    API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

    def __init__(self, on_progress=None):
        self._on_progress = on_progress or (lambda *a: None)
        self._is_frozen = getattr(sys, 'frozen', False)

    def _parse_version(self, tag):
        cleaned = tag.lstrip('vV').strip()
        parts = []
        for segment in cleaned.split('.'):
            try:
                parts.append(int(segment))
            except ValueError:
                break
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])

    def _is_newer(self, remote_tag):
        return self._parse_version(remote_tag) > self._parse_version(APP_VERSION)

    def _find_setup_asset(self, assets):
        for asset in assets:
            name = asset.get('name', '').lower()
            if 'setup' in name and name.endswith('.exe'):
                return asset
        for asset in assets:
            if asset.get('name', '').lower().endswith('.exe'):
                return asset
        return None

    def run(self):
        if not self._is_frozen:
            return None

        self._on_progress("checking", 0, "Checking for updates...")

        try:
            req = urllib.request.Request(self.API_URL, headers={
                'User-Agent': f'AudioRecorder/{APP_VERSION}',
                'Accept': 'application/vnd.github.v3+json',
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
        except Exception:
            return None

        tag = data.get('tag_name', '')
        if not tag or not self._is_newer(tag):
            return None

        asset = self._find_setup_asset(data.get('assets', []))
        if not asset:
            return None

        download_url = asset['browser_download_url']
        total_size = asset.get('size', 0)
        version_clean = tag.lstrip('vV')

        import tempfile
        setup_path = os.path.join(tempfile.gettempdir(), f"AudioRecorder_Setup_v{version_clean}.exe")

        self._on_progress("downloading", 0, f"Downloading v{version_clean}...")

        try:
            req = urllib.request.Request(download_url, headers={'User-Agent': 'AudioRecorder'})
            with urllib.request.urlopen(req, timeout=30) as response:
                downloaded = 0
                chunk_size = 65536
                with open(setup_path, 'wb') as out_file:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = min(downloaded / total_size, 1.0)
                            mb_done = downloaded / 1048576
                            mb_total = total_size / 1048576
                            self._on_progress("downloading", pct,
                                f"Downloading v{version_clean}  {mb_done:.1f}/{mb_total:.1f} MB")
        except Exception:
            try:
                os.remove(setup_path)
            except OSError:
                pass
            return None

        if total_size and os.path.getsize(setup_path) != total_size:
            try:
                os.remove(setup_path)
            except OSError:
                pass
            return None

        try:
            with open(setup_path, 'rb') as f:
                if f.read(2) != b'MZ':
                    os.remove(setup_path)
                    return None
        except Exception:
            return None

        self._on_progress("installing", 1.0, f"Installing v{version_clean}...")

        proc = subprocess.Popen([
            setup_path, '/VERYSILENT', '/SUPPRESSMSGBOXES',
            '/CLOSEAPPLICATIONS', '/RESTARTAPPLICATIONS',
        ])
        proc.wait()
        return setup_path


class AudioEngine:

    def __init__(self):
        self.is_recording = False
        self.is_paused = False
        self.sample_rate = 48000
        self.channels = 2
        self.start_time = 0
        self.pause_offset = 0
        self.lock = threading.Lock()

        self.capture_mic = True
        self.capture_system = True
        self.mic_gain = 1.0
        self.system_gain = 1.0
        self.noise_threshold = 0.0

        self.mic_level = 0.0
        self.system_level = 0.0

        self._mic_thread = None
        self._sys_thread = None
        self._mic_buf = []
        self._sys_buf = []
        self._mic_rate = 48000
        self._sys_rate = 48000
        self.last_error = None
        self.markers = []

        self._mic_devices = []
        self._loopback_devices = []
        self._mic_selected = None
        self._loopback_selected = None

        self._pa = pyaudio.PyAudio() if AUDIO_OK else None

    def shutdown(self):
        if self.is_recording:
            self.is_recording = False
            time.sleep(0.4)
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None

    def scan_microphones(self):
        if not self._pa:
            return []

        raw_list = []
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0 and not dev.get('isLoopbackDevice', False):
                raw_list.append(dev)

        grouped = {}
        for dev in raw_list:
            name = dev['name']
            if name not in grouped:
                grouped[name] = dev
            else:
                existing_delta = abs(grouped[name]['defaultSampleRate'] - 48000)
                new_delta = abs(dev['defaultSampleRate'] - 48000)
                if new_delta < existing_delta:
                    grouped[name] = dev

        self._mic_devices = list(grouped.values())
        return [(d['index'], d['name']) for d in self._mic_devices]

    def scan_loopback(self):
        if not self._pa:
            return []

        result = []
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if dev.get('isLoopbackDevice', False):
                result.append(dev)
        self._loopback_devices = result
        return [(d['index'], d['name']) for d in result]

    def pick_mic(self, device_index):
        for d in self._mic_devices:
            if d['index'] == device_index:
                self._mic_selected = d
                return

    def pick_loopback(self, device_index):
        for d in self._loopback_devices:
            if d['index'] == device_index:
                self._loopback_selected = d
                self._sys_rate = int(d['defaultSampleRate'])
                return

    def start(self):
        self.is_recording = True
        self.is_paused = False
        self._mic_buf.clear()
        self._sys_buf.clear()
        self.markers.clear()
        self.last_error = None
        self.mic_level = 0
        self.system_level = 0
        self.start_time = time.time()
        self.pause_offset = 0

        if self.capture_mic and self._mic_selected:
            self._mic_thread = threading.Thread(target=self._worker_mic, daemon=True)
            self._mic_thread.start()

        if self.capture_system and self._loopback_selected:
            self._sys_thread = threading.Thread(target=self._worker_loopback, daemon=True)
            self._sys_thread.start()

    def _try_open_stream(self, dev, rates_to_try):
        nch = int(dev['maxInputChannels'])
        for rate in rates_to_try:
            try:
                stream = self._pa.open(
                    format=pyaudio.paFloat32,
                    channels=nch,
                    rate=rate,
                    input=True,
                    input_device_index=dev['index'],
                    frames_per_buffer=512,
                )
                return stream, nch, rate
            except Exception:
                continue
        return None, nch, 0

    def _worker_mic(self):
        dev = self._mic_selected
        native = int(dev['defaultSampleRate'])

        rates = []
        if native == 48000:
            rates = [48000, 44100]
        elif native == 44100:
            rates = [44100, 48000]
        else:
            rates = [native, 48000, 44100]

        stream, nch, actual_rate = self._try_open_stream(dev, rates)
        if stream is None:
            self.last_error = "Cannot open microphone"
            return

        self._mic_rate = actual_rate
        try:
            while self.is_recording:
                if self.is_paused:
                    time.sleep(0.02)
                    continue

                try:
                    raw = stream.read(512, exception_on_overflow=False)
                except Exception:
                    continue

                pcm = np.frombuffer(raw, dtype=np.float32).reshape(-1, nch)
                pcm = pcm * self.mic_gain

                rms = np.sqrt(np.mean(pcm ** 2))

                if self.noise_threshold > 0 and rms < self.noise_threshold:
                    pcm = np.zeros_like(pcm)
                    self.mic_level = max(self.mic_level * 0.5, 0)
                else:
                    self.mic_level = min(rms * 8, 1.0)

                with self.lock:
                    if nch == 1 and self.channels == 2:
                        pcm = np.column_stack([pcm, pcm])
                    elif nch > self.channels:
                        pcm = pcm[:, :self.channels]
                    self._mic_buf.append(pcm)
        except Exception as e:
            self.last_error = f"Mic recording: {e}"
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass

    def _worker_loopback(self):
        dev = self._loopback_selected
        native = int(dev['defaultSampleRate'])

        stream, nch, actual_rate = self._try_open_stream(dev, [native])
        if stream is None:
            self.last_error = "Cannot open system audio"
            return

        self._sys_rate = actual_rate
        try:
            while self.is_recording:
                if self.is_paused:
                    time.sleep(0.02)
                    continue

                try:
                    raw = stream.read(512, exception_on_overflow=False)
                except Exception:
                    continue

                pcm = np.frombuffer(raw, dtype=np.float32).reshape(-1, nch)
                pcm = pcm * self.system_gain
                rms = np.sqrt(np.mean(pcm ** 2))
                self.system_level = min(rms * 8, 1.0)

                with self.lock:
                    if nch < self.channels:
                        pcm = np.column_stack([pcm] * self.channels)[:, :self.channels]
                    elif nch > self.channels:
                        pcm = pcm[:, :self.channels]
                    self._sys_buf.append(pcm)
        except Exception as e:
            self.last_error = f"System recording: {e}"
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass

    def pause(self):
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            self.pause_offset += time.time() - self.start_time

    def resume(self):
        if self.is_recording and self.is_paused:
            self.is_paused = False
            self.start_time = time.time()

    def stop(self):
        self.is_recording = False
        time.sleep(0.35)

        with self.lock:
            mic = np.concatenate(self._mic_buf) if self._mic_buf else None
            sys_a = np.concatenate(self._sys_buf) if self._sys_buf else None

        has_both = mic is not None and sys_a is not None
        if has_both:
            if self._mic_rate != self._sys_rate:
                mic = _resample(mic, self._mic_rate, self._sys_rate)
            self.sample_rate = self._sys_rate
            n = min(len(mic), len(sys_a))
            mixed = mic[:n] + sys_a[:n]
            peak = np.max(np.abs(mixed))
            if peak > 1.0:
                mixed /= peak
            return mixed

        if mic is not None:
            self.sample_rate = self._mic_rate
            return mic
        if sys_a is not None:
            self.sample_rate = self._sys_rate
            return sys_a
        return None

    def elapsed(self):
        if not self.is_recording:
            return 0.0
        if self.is_paused:
            return self.pause_offset
        return self.pause_offset + (time.time() - self.start_time)

    def add_marker(self):
        ts = self.elapsed()
        idx = len(self.markers) + 1
        self.markers.append((ts, f"Mark {idx}"))
        return ts, idx


class PostSaveDialog(ctk.CTkToplevel):

    BG = '#05050d'
    SURFACE = '#0a0a18'
    BORDER = '#152035'
    NEON = '#00e5ff'
    PINK = '#ff2a6d'
    GREEN = '#39ff14'
    ORANGE = '#ff9500'
    PURPLE = '#c050ff'
    TEXT = '#e4e4f0'
    DIM = '#454568'

    def __init__(self, parent, audio_data, sample_rate, markers,
                 categories, rec_name, rec_sources, duration, fmt,
                 recordings_path, on_done):
        super().__init__(parent)

        self._audio = audio_data
        self._rate = sample_rate
        self._markers = list(markers)
        self._categories = list(categories)
        self._rec_name = rec_name
        self._sources = rec_sources
        self._duration = duration
        self._fmt = fmt
        self._base_path = recordings_path
        self._on_done = on_done

        self.title("Save Recording")
        self.geometry("440x520")
        self.resizable(False, False)
        self.configure(fg_color=self.BG)
        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._do_discard)

        self._icon_file = None
        if hasattr(parent, '_icon_path') and os.path.exists(parent._icon_path):
            self._icon_file = parent._icon_path
            try:
                self.iconbitmap(self._icon_file)
                self.after(80, self._apply_icon)
            except Exception:
                pass

        self.after(50, self._center_on_parent)
        self._build_ui()

    def _apply_icon(self):
        if not self._icon_file:
            return
        try:
            self.iconbitmap(self._icon_file)
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            flags = 0x00000010 | 0x00000040 | 0x00000001
            handle = ctypes.windll.user32.LoadImageW(0, self._icon_file, 1, 0, 0, flags)
            if handle:
                ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 0, handle)
                ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 1, handle)
        except Exception:
            pass

    def _center_on_parent(self):
        self.update_idletasks()
        pw, ph = self.master.winfo_width(), self.master.winfo_height()
        px, py = self.master.winfo_x(), self.master.winfo_y()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{px + (pw - w) // 2}+{py + (ph - h) // 2}")

    def _build_ui(self):
        main = ctk.CTkScrollableFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=2, pady=2)

        ctk.CTkLabel(
            main, text="SAVE RECORDING",
            font=ctk.CTkFont(size=16, weight="bold"), text_color=self.NEON
        ).pack(anchor="w", padx=14, pady=(12, 4))

        mins, secs = int(self._duration // 60), int(self._duration % 60)
        info = f"Duration: {mins:02d}:{secs:02d}  \u2022  Markers: {len(self._markers)}"
        ctk.CTkLabel(
            main, text=info,
            font=ctk.CTkFont(size=11), text_color=self.DIM
        ).pack(anchor="w", padx=14, pady=(0, 8))

        ctk.CTkFrame(main, fg_color=self.BORDER, height=1).pack(fill="x", padx=14, pady=4)

        ctk.CTkLabel(
            main, text="Category",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=self.TEXT
        ).pack(anchor="w", padx=14, pady=(8, 3))

        folder_opts = ["\u2014 No category \u2014"] + self._categories
        self._cat_var = ctk.StringVar(value=folder_opts[0])
        self._cat_combo = ctk.CTkComboBox(
            main, values=folder_opts, variable=self._cat_var,
            state="readonly",
            fg_color=self.BG, border_color=self.BORDER,
            button_color=self.NEON, button_hover_color='#009db3',
            dropdown_fg_color=self.SURFACE, dropdown_hover_color='#0a2535',
            font=ctk.CTkFont(size=11)
        )
        self._cat_combo.pack(fill="x", padx=14, pady=(0, 6))

        new_row = ctk.CTkFrame(main, fg_color="transparent")
        new_row.pack(fill="x", padx=14, pady=(0, 8))

        self._new_cat_entry = ctk.CTkEntry(
            new_row, placeholder_text="New category name...",
            fg_color=self.BG, border_color=self.BORDER,
            text_color=self.TEXT, placeholder_text_color=self.DIM,
            font=ctk.CTkFont(size=11), height=30
        )
        self._new_cat_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        ctk.CTkButton(
            new_row, text="+", width=40, height=30,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.PURPLE, hover_color='#9535d0',
            corner_radius=6, command=self._add_category
        ).pack(side="right")

        ctk.CTkFrame(main, fg_color=self.BORDER, height=1).pack(fill="x", padx=14, pady=4)

        self._trim_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            main, text="Remove silence", variable=self._trim_var,
            font=ctk.CTkFont(size=12),
            fg_color=self.ORANGE, hover_color='#cc7700',
            border_color=self.BORDER, checkmark_color='#ffffff'
        ).pack(anchor="w", padx=14, pady=(6, 8))

        if self._markers:
            ctk.CTkFrame(main, fg_color=self.BORDER, height=1).pack(fill="x", padx=14, pady=4)

            ctk.CTkLabel(
                main, text="Markers",
                font=ctk.CTkFont(size=12, weight="bold"), text_color=self.TEXT
            ).pack(anchor="w", padx=14, pady=(6, 2))

            mf = ctk.CTkFrame(main, fg_color=self.SURFACE, corner_radius=8)
            mf.pack(fill="x", padx=14, pady=(0, 8))

            for ts, label in self._markers:
                m, s = int(ts // 60), int(ts % 60)
                row = ctk.CTkFrame(mf, fg_color="transparent")
                row.pack(fill="x", padx=8, pady=2)
                ctk.CTkLabel(
                    row, text=f"\u2691 {m:02d}:{s:02d}",
                    font=ctk.CTkFont(family="Consolas", size=11),
                    text_color=self.ORANGE
                ).pack(side="left")
                ctk.CTkLabel(
                    row, text=f"  {label}",
                    font=ctk.CTkFont(size=11), text_color=self.DIM
                ).pack(side="left")

        btn_row = ctk.CTkFrame(main, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(12, 14))
        btn_row.grid_columnconfigure(0, weight=3)
        btn_row.grid_columnconfigure(1, weight=2)

        ctk.CTkButton(
            btn_row, text="SAVE", height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.GREEN, hover_color='#25d00e',
            text_color='#000000', corner_radius=8,
            command=self._do_save
        ).grid(row=0, column=0, sticky="ew", padx=(0, 4))

        ctk.CTkButton(
            btn_row, text="DISCARD", height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.PINK, hover_color='#d01850',
            text_color='#ffffff', corner_radius=8,
            command=self._do_discard
        ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

    def _add_category(self):
        raw = self._new_cat_entry.get().strip()
        if not raw:
            return
        name = raw[:40]
        if name not in self._categories:
            self._categories.append(name)
            all_opts = ["\u2014 No category \u2014"] + self._categories
            self._cat_combo.configure(values=all_opts)
        self._cat_var.set(name)
        self._new_cat_entry.delete(0, "end")

    def _do_save(self):
        cat = self._cat_var.get()
        if cat.startswith("\u2014"):
            cat = None

        audio = self._audio
        if self._trim_var.get():
            audio = _strip_silence(audio, self._rate)

        self._on_done("save", audio, cat, list(self._categories), self._markers)
        self.grab_release()
        self.destroy()

    def _do_discard(self):
        self._on_done("discard", None, None, list(self._categories), [])
        self.grab_release()
        self.destroy()


class RecorderApp(ctk.CTk):

    BG = '#05050d'
    SURFACE = '#0a0a18'
    SURFACE_ALT = '#111125'
    BORDER = '#152035'
    NEON = '#00e5ff'
    NEON_DIM = '#009db3'
    PINK = '#ff2a6d'
    PINK_DIM = '#d01850'
    GREEN = '#39ff14'
    GREEN_DIM = '#25d00e'
    PURPLE = '#c050ff'
    PURPLE_DIM = '#9535d0'
    ORANGE = '#ff9500'
    TEXT = '#e4e4f0'
    TEXT_DIM = '#454568'
    MUTED = '#6a6a8a'
    CYAN = '#00e5ff'
    MAGENTA = '#ff006e'

    PRIMARY = '#00e5ff'
    PRIMARY_DIM = '#009db3'
    DANGER = '#ff2a6d'
    DANGER_DIM = '#d01850'
    SUCCESS = '#39ff14'
    SUCCESS_DIM = '#25d00e'
    WARNING = '#ff9500'

    def __init__(self):
        super().__init__()

        self.engine = AudioEngine()
        self._mic_list = []
        self._sys_list = []
        self._rec_sources = []
        self._tray_icon = None

        self.title(f"Audio Recorder v{APP_VERSION}  |  YALOKGAR")
        self.geometry("540x780")
        self.minsize(380, 500)
        self.configure(fg_color=self.BG)
        ctk.set_appearance_mode("dark")

        if getattr(sys, 'frozen', False):
            self._bundle_dir = sys._MEIPASS
            self._app_dir = os.path.dirname(sys.executable)
        else:
            self._bundle_dir = os.path.dirname(os.path.abspath(__file__))
            self._app_dir = self._bundle_dir

        icon_path = os.path.join(self._bundle_dir, "icon.ico")
        self._icon_path = icon_path
        if os.path.exists(icon_path):
            self.iconbitmap(icon_path)
            self.iconbitmap(default=icon_path)
            self.after(100, lambda: self._force_icon())
        self.recordings_path = os.path.join(self._app_dir, "Recordings")
        os.makedirs(self.recordings_path, exist_ok=True)

        self._config_path = os.path.join(self._app_dir, "config.json")
        self._categories = self._load_config()
        self._pending_markers = []
        self._last_duration = 0

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_interface()
        self._load_devices()
        self._tick_timer()
        self._tick_meters()
        self._init_tray()

        self.bind('<Unmap>', self._on_window_unmap)
        self.after(3000, self._check_for_updates)

    def _init_tray(self):
        if not HAS_TRAY:
            return
        img = self._draw_tray_icon('#71717a')
        menu = pystray.Menu(
            pystray.MenuItem("Show", self._tray_show, default=True),
            pystray.MenuItem("Record / Stop", self._tray_toggle_rec),
            pystray.MenuItem("Pause / Resume", self._tray_toggle_pause),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", self._tray_exit),
        )
        self._tray_icon = pystray.Icon("AudioRecorder", img, "Audio Recorder", menu)
        threading.Thread(target=self._tray_icon.run, daemon=True).start()

    def _draw_tray_icon(self, color, size=64):
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        pad = size // 6
        draw.ellipse([pad, pad, size - pad, size - pad], fill=color)
        inner_pad = size // 4
        draw.ellipse([inner_pad, inner_pad, size - inner_pad, size - inner_pad], fill=color)
        mic_w = size // 6
        cx = size // 2
        draw.rectangle([cx - mic_w, pad + 4, cx + mic_w, size // 2], fill='white')
        draw.arc([cx - size // 4, size // 4, cx + size // 4, size // 2 + size // 6],
                 0, 180, fill='white', width=2)
        draw.line([cx, size // 2 + size // 6, cx, size - pad - 2], fill='white', width=2)
        return img

    def _update_tray(self):
        if not self._tray_icon:
            return
        if self.engine.is_recording:
            clr = self.ORANGE if self.engine.is_paused else self.PINK
            state = "Paused" if self.engine.is_paused else "Recording"
        else:
            clr = '#71717a'
            state = "Ready"
        self._tray_icon.icon = self._draw_tray_icon(clr)
        self._tray_icon.title = f"Audio Recorder - {state}"

    def _on_window_unmap(self, event):
        if event.widget != self:
            return
        if self.state() == 'iconic' and HAS_TRAY and self._tray_icon:
            self.after(50, self.withdraw)

    def _tray_show(self):
        self.after(0, self._restore_window)

    def _restore_window(self):
        self.deiconify()
        self.state('normal')
        self.lift()
        self.focus_force()

    def _tray_toggle_rec(self):
        self.after(0, self._on_record)

    def _tray_toggle_pause(self):
        self.after(0, self._on_pause)

    def _tray_exit(self):
        self.after(0, self._on_close)

    def _check_for_updates(self):
        def _progress(stage, pct, text):
            self.after(0, lambda: self._update_progress(stage, pct, text))

        def _run():
            updater = AutoUpdater(on_progress=_progress)
            result = updater.run()
            if result is None:
                self.after(0, lambda: self._show_status("Ready", self.TEXT_DIM))

        threading.Thread(target=_run, daemon=True).start()

    def _update_progress(self, stage, pct, text):
        self._show_status(text, self.NEON)

        if stage == "downloading" and not hasattr(self, '_update_bar'):
            self._update_overlay = ctk.CTkFrame(
                self, fg_color=self.BG, corner_radius=0
            )
            self._update_overlay.place(relx=0, rely=1.0, relwidth=1, anchor="sw", y=-2)

            self._update_bar = ctk.CTkProgressBar(
                self._update_overlay, width=400, height=6,
                fg_color=self.SURFACE, progress_color=self.NEON,
                corner_radius=3,
            )
            self._update_bar.set(0)
            self._update_bar.pack(fill="x", padx=16, pady=4)

        if hasattr(self, '_update_bar'):
            self._update_bar.set(pct)

        if stage == "installing" and hasattr(self, '_update_bar'):
            self._update_bar.configure(progress_color=self.GREEN, mode="indeterminate")
            self._update_bar.start()

    def _force_icon(self):
        try:
            self.iconbitmap(self._icon_path)
            import ctypes
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            icon_flags = 0x00000010 | 0x00000040 | 0x00000001
            handle = ctypes.windll.user32.LoadImageW(
                0, self._icon_path, 1, 0, 0, icon_flags
            )
            if handle:
                ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 0, handle)
                ctypes.windll.user32.SendMessageW(hwnd, 0x0080, 1, handle)
        except Exception:
            pass

    def _on_close(self):
        if self._tray_icon:
            self._tray_icon.stop()
        self.engine.shutdown()
        self.destroy()

    def _card(self, parent, glow=None, **kw):
        border_clr = glow if glow else self.BORDER
        return ctk.CTkFrame(
            parent, fg_color=self.SURFACE, corner_radius=12,
            border_width=1, border_color=border_clr, **kw
        )

    def _build_interface(self):
        scroll = ctk.CTkScrollableFrame(
            self, fg_color="transparent",
            scrollbar_button_color=self.BORDER,
            scrollbar_button_hover_color=self.NEON
        )
        scroll.pack(fill="both", expand=True)

        self._build_header(scroll)
        self._build_timer(scroll)
        self._build_meters(scroll)
        self._build_devices(scroll)
        self._build_volumes(scroll)
        self._build_controls(scroll)

    def _build_header(self, parent):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=16, pady=(14, 0))

        ctk.CTkLabel(
            row, text="AUDIO RECORDER",
            font=ctk.CTkFont(size=20, weight="bold"), text_color=self.NEON
        ).pack(side="left")

        box = ctk.CTkFrame(row, fg_color="transparent")
        box.pack(side="right")

        self._dot = ctk.CTkLabel(
            box, text="\u25cf", font=ctk.CTkFont(size=11), text_color=self.TEXT_DIM
        )
        self._dot.pack(side="left", padx=(0, 5))

        self._status = ctk.CTkLabel(
            box, text="Ready", font=ctk.CTkFont(size=12), text_color=self.TEXT_DIM
        )
        self._status.pack(side="left")

        credit_row = ctk.CTkFrame(parent, fg_color="transparent")
        credit_row.pack(fill="x", padx=16, pady=(0, 6))
        ctk.CTkLabel(
            credit_row, text="by YALOKGAR",
            font=ctk.CTkFont(size=10), text_color=self.PURPLE
        ).pack(side="left")

    def _build_timer(self, parent):
        card = self._card(parent, glow='#0a3545')
        card.pack(fill="x", padx=16, pady=6)

        self._timer = ctk.CTkLabel(
            card, text="00:00:00",
            font=ctk.CTkFont(family="Consolas", size=42, weight="bold"),
            text_color=self.NEON
        )
        self._timer.pack(pady=(16, 14))

    def _build_meters(self, parent):
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(fill="x", padx=16, pady=(0, 4))

        r1 = ctk.CTkFrame(wrap, fg_color="transparent")
        r1.pack(fill="x", pady=2)
        ctk.CTkLabel(
            r1, text="MIC", width=36, anchor="w",
            font=ctk.CTkFont(size=10, weight="bold"), text_color=self.NEON
        ).pack(side="left")
        self._mic_bar = ctk.CTkProgressBar(
            r1, height=8, corner_radius=4,
            progress_color=self.NEON, fg_color='#0a1520'
        )
        self._mic_bar.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._mic_bar.set(0)

        r2 = ctk.CTkFrame(wrap, fg_color="transparent")
        r2.pack(fill="x", pady=2)
        ctk.CTkLabel(
            r2, text="SYS", width=36, anchor="w",
            font=ctk.CTkFont(size=10, weight="bold"), text_color=self.MAGENTA
        ).pack(side="left")
        self._sys_bar = ctk.CTkProgressBar(
            r2, height=8, corner_radius=4,
            progress_color=self.MAGENTA, fg_color='#1a0a15'
        )
        self._sys_bar.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._sys_bar.set(0)

    def _build_devices(self, parent):
        card = self._card(parent)
        card.pack(fill="x", padx=16, pady=6)

        if not AUDIO_OK:
            ctk.CTkLabel(
                card, text="pyaudiowpatch not installed!\npip install pyaudiowpatch",
                font=ctk.CTkFont(size=13), text_color=self.DANGER, justify="center"
            ).pack(padx=12, pady=16)
            self._mic_combo = None
            self._sys_combo = None
            self._mic_sw = None
            self._sys_sw = None
            return

        ctk.CTkLabel(
            card, text="Microphone",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=self.TEXT
        ).pack(anchor="w", padx=12, pady=(10, 3))

        self._mic_combo = ctk.CTkComboBox(
            card, state="readonly", command=self._pick_mic,
            fg_color=self.BG, border_color=self.BORDER,
            button_color=self.NEON, button_hover_color=self.NEON_DIM,
            dropdown_fg_color=self.SURFACE, dropdown_hover_color='#0a2535',
            font=ctk.CTkFont(size=11)
        )
        self._mic_combo.pack(fill="x", padx=12, pady=(0, 4))

        self._mic_sw = ctk.CTkSwitch(
            card, text="Record microphone", font=ctk.CTkFont(size=11),
            progress_color=self.NEON, button_color=self.NEON,
            button_hover_color='#ffffff', command=self._toggle_mic
        )
        self._mic_sw.pack(anchor="w", padx=12, pady=(0, 8))
        self._mic_sw.select()

        ctk.CTkFrame(card, fg_color=self.BORDER, height=1).pack(fill="x", padx=12)

        ctk.CTkLabel(
            card, text="System Audio (Loopback)",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=self.TEXT
        ).pack(anchor="w", padx=12, pady=(8, 3))

        self._sys_combo = ctk.CTkComboBox(
            card, state="readonly", command=self._pick_loopback,
            fg_color=self.BG, border_color=self.BORDER,
            button_color=self.NEON, button_hover_color=self.NEON_DIM,
            dropdown_fg_color=self.SURFACE, dropdown_hover_color='#0a2535',
            font=ctk.CTkFont(size=11)
        )
        self._sys_combo.pack(fill="x", padx=12, pady=(0, 4))

        self._sys_sw = ctk.CTkSwitch(
            card, text="Record system audio", font=ctk.CTkFont(size=11),
            progress_color=self.NEON, button_color=self.NEON,
            button_hover_color='#ffffff', command=self._toggle_sys
        )
        self._sys_sw.pack(anchor="w", padx=12, pady=(0, 10))
        self._sys_sw.select()

    def _build_volumes(self, parent):
        card = self._card(parent)
        card.pack(fill="x", padx=16, pady=6)

        h1 = ctk.CTkFrame(card, fg_color="transparent")
        h1.pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(h1, text="Mic Volume", font=ctk.CTkFont(size=11),
                     text_color=self.TEXT).pack(side="left")
        self._mic_pct = ctk.CTkLabel(
            h1, text="100%", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self.NEON
        )
        self._mic_pct.pack(side="right")

        self._mic_sl = ctk.CTkSlider(
            card, from_=0, to=200, command=self._vol_mic,
            progress_color=self.NEON, button_color=self.NEON,
            button_hover_color='#ffffff', fg_color='#0a1520'
        )
        self._mic_sl.set(100)
        self._mic_sl.pack(fill="x", padx=12, pady=(0, 6))

        h2 = ctk.CTkFrame(card, fg_color="transparent")
        h2.pack(fill="x", padx=12, pady=(4, 2))
        ctk.CTkLabel(h2, text="System Volume", font=ctk.CTkFont(size=11),
                     text_color=self.TEXT).pack(side="left")
        self._sys_pct = ctk.CTkLabel(
            h2, text="100%", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self.NEON
        )
        self._sys_pct.pack(side="right")

        self._sys_sl = ctk.CTkSlider(
            card, from_=0, to=200, command=self._vol_sys,
            progress_color=self.NEON, button_color=self.NEON,
            button_hover_color='#ffffff', fg_color='#0a1520'
        )
        self._sys_sl.set(100)
        self._sys_sl.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkFrame(card, fg_color=self.BORDER, height=1).pack(fill="x", padx=12)

        ng_hdr = ctk.CTkFrame(card, fg_color="transparent")
        ng_hdr.pack(fill="x", padx=12, pady=(8, 2))
        ctk.CTkLabel(ng_hdr, text="Noise Gate", font=ctk.CTkFont(size=11),
                     text_color=self.TEXT).pack(side="left")
        self._ng_lbl = ctk.CTkLabel(
            ng_hdr, text="OFF", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=self.TEXT_DIM
        )
        self._ng_lbl.pack(side="right")

        self._ng_sl = ctk.CTkSlider(
            card, from_=0, to=100, command=self._set_noise_gate,
            progress_color=self.ORANGE, button_color=self.ORANGE,
            button_hover_color='#ffffff', fg_color='#1a1000'
        )
        self._ng_sl.set(0)
        self._ng_sl.pack(fill="x", padx=12, pady=(0, 10))

    def _build_controls(self, parent):
        name_card = self._card(parent)
        name_card.pack(fill="x", padx=16, pady=6)

        ctk.CTkLabel(
            name_card, text="Recording Name",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=self.TEXT
        ).pack(anchor="w", padx=12, pady=(10, 3))

        self._name_entry = ctk.CTkEntry(
            name_card, placeholder_text="Leave empty for auto-name",
            fg_color=self.BG, border_color=self.BORDER,
            text_color=self.TEXT, placeholder_text_color=self.TEXT_DIM,
            font=ctk.CTkFont(size=12)
        )
        self._name_entry.pack(fill="x", padx=12, pady=(0, 10))

        fmt_card = self._card(parent)
        fmt_card.pack(fill="x", padx=16, pady=6)

        fmt_row = ctk.CTkFrame(fmt_card, fg_color="transparent")
        fmt_row.pack(fill="x", padx=12, pady=8)
        ctk.CTkLabel(
            fmt_row, text="Format",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=self.TEXT
        ).pack(side="left")

        self._fmt = ctk.StringVar(value="wav")
        for tag in ["flac", "wav"]:
            ctk.CTkRadioButton(
                fmt_row, text=tag.upper(), variable=self._fmt, value=tag,
                font=ctk.CTkFont(size=11), fg_color=self.NEON,
                hover_color=self.NEON_DIM, border_color=self.BORDER
            ).pack(side="right", padx=(8, 4))

        bf = ctk.CTkFrame(parent, fg_color="transparent")
        bf.pack(fill="x", padx=16, pady=(8, 4))
        bf.grid_columnconfigure(0, weight=4)
        bf.grid_columnconfigure(1, weight=2)
        bf.grid_columnconfigure(2, weight=2)

        self._rec_btn = ctk.CTkButton(
            bf, text="REC", height=48,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=self.PINK, hover_color=self.PINK_DIM,
            text_color='#ffffff',
            corner_radius=10, command=self._on_record
        )
        self._rec_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        self._pause_btn = ctk.CTkButton(
            bf, text="PAUSE", height=48,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=self.PURPLE, hover_color=self.PURPLE_DIM,
            text_color='#ffffff',
            corner_radius=10, state="disabled", command=self._on_pause
        )
        self._pause_btn.grid(row=0, column=1, sticky="ew", padx=3)

        self._mark_btn = ctk.CTkButton(
            bf, text="\u2691", height=48,
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=self.ORANGE, hover_color='#cc7700',
            text_color='#ffffff',
            corner_radius=10, state="disabled", command=self._on_mark
        )
        self._mark_btn.grid(row=0, column=2, sticky="ew", padx=(3, 0))

        ctk.CTkButton(
            parent, text="Open Recordings", height=34,
            font=ctk.CTkFont(size=11), fg_color="transparent",
            hover_color=self.SURFACE_ALT, border_width=1,
            border_color=self.BORDER, corner_radius=8,
            text_color=self.TEXT_DIM,
            command=self._open_folder
        ).pack(fill="x", padx=16, pady=(4, 6))

        footer = ctk.CTkFrame(parent, fg_color="transparent")
        footer.pack(fill="x", padx=16, pady=(0, 12))
        ctk.CTkLabel(
            footer, text="YALOKGAR",
            font=ctk.CTkFont(family="Consolas", size=9),
            text_color='#252545'
        ).pack(side="right")

    def _load_devices(self):
        if not AUDIO_OK:
            return

        mics = self.engine.scan_microphones()
        self._mic_list = mics
        if mics:
            names = [m[1] for m in mics]
            self._mic_combo.configure(values=names)
            self._mic_combo.set(names[0])
            self.engine.pick_mic(mics[0][0])
        else:
            self._mic_combo.configure(values=["No microphone found"])
            self._mic_combo.set("No microphone found")

        sources = self.engine.scan_loopback()
        self._sys_list = sources
        if sources:
            names = [s[1] for s in sources]
            self._sys_combo.configure(values=names)
            self._sys_combo.set(names[0])
            self.engine.pick_loopback(sources[0][0])
        else:
            self._sys_combo.configure(values=["No loopback device"])
            self._sys_combo.set("No loopback device")

    def _pick_mic(self, val):
        for dev_id, name in self._mic_list:
            if name == val:
                self.engine.pick_mic(dev_id)
                return

    def _pick_loopback(self, val):
        for dev_id, name in self._sys_list:
            if name == val:
                self.engine.pick_loopback(dev_id)
                return

    def _toggle_mic(self):
        self.engine.capture_mic = bool(self._mic_sw.get())

    def _toggle_sys(self):
        self.engine.capture_system = bool(self._sys_sw.get())

    def _vol_mic(self, v):
        p = int(v)
        self._mic_pct.configure(text=f"{p}%")
        self.engine.mic_gain = p / 100.0

    def _vol_sys(self, v):
        p = int(v)
        self._sys_pct.configure(text=f"{p}%")
        self.engine.system_gain = p / 100.0

    def _set_noise_gate(self, v):
        val = int(v)
        if val == 0:
            self._ng_lbl.configure(text="OFF", text_color=self.TEXT_DIM)
            self.engine.noise_threshold = 0.0
        else:
            self._ng_lbl.configure(text=f"{val}%", text_color=self.ORANGE)
            self.engine.noise_threshold = val / 2000.0

    def _show_status(self, text, color):
        self._status.configure(text=text, text_color=color)
        self._dot.configure(text_color=color)

    def _on_record(self):
        if not AUDIO_OK:
            self._show_status("pyaudiowpatch required", self.DANGER)
            return

        if not self.engine.is_recording:
            if not self.engine.capture_mic and not self.engine.capture_system:
                self._show_status("Enable at least one source", self.WARNING)
                return

            self._rec_sources = []
            if self.engine.capture_mic:
                self._rec_sources.append("mic")
            if self.engine.capture_system:
                self._rec_sources.append("sys")

            self.engine.start()
            self._rec_btn.configure(text="STOP", fg_color=self.GREEN,
                                    hover_color=self.GREEN_DIM)
            self._pause_btn.configure(state="normal", text="PAUSE")
            self._mark_btn.configure(text="\u2691", state="normal")
            self._name_entry.configure(state="disabled")
            self._show_status("Recording", self.PINK)
            self._timer.configure(text_color=self.PINK)
            if self._mic_combo:
                self._mic_combo.configure(state="disabled")
            if self._sys_combo:
                self._sys_combo.configure(state="disabled")
            self._update_tray()
        else:
            self._last_duration = self.engine.elapsed()
            self._pending_markers = list(self.engine.markers)
            self._show_status("Processing...", self.WARNING)
            self.update()

            audio = self.engine.stop()

            self._rec_btn.configure(text="REC", fg_color=self.PINK,
                                    hover_color=self.PINK_DIM)
            self._pause_btn.configure(state="disabled", text="PAUSE")
            self._mark_btn.configure(text="\u2691", state="disabled")
            self._name_entry.configure(state="normal")
            self._timer.configure(text="00:00:00", text_color=self.NEON)
            if self._mic_combo:
                self._mic_combo.configure(state="readonly")
            if self._sys_combo:
                self._sys_combo.configure(state="readonly")
            self._mic_bar.set(0)
            self._sys_bar.set(0)
            self._update_tray()

            if audio is None or len(audio) == 0:
                msg = self.engine.last_error or "No audio captured"
                self._show_status(msg, self.DANGER)
                return

            PostSaveDialog(
                self, audio, self.engine.sample_rate,
                self._pending_markers, self._categories,
                self._name_entry.get().strip(), self._rec_sources,
                self._last_duration, self._fmt.get(),
                self.recordings_path, self._handle_post_save
            )

    def _on_pause(self):
        if not self.engine.is_recording:
            return
        if self.engine.is_paused:
            self.engine.resume()
            self._pause_btn.configure(text="PAUSE")
            self._show_status("Recording", self.PINK)
            self._timer.configure(text_color=self.PINK)
        else:
            self.engine.pause()
            self._pause_btn.configure(text="RESUME")
            self._show_status("Paused", self.ORANGE)
            self._timer.configure(text_color=self.ORANGE)
        self._update_tray()

    def _on_mark(self):
        if not self.engine.is_recording:
            return
        ts, idx = self.engine.add_marker()
        self._mark_btn.configure(text=f"\u2691 {idx}")
        m, s = int(ts // 60), int(ts % 60)
        self._show_status(f"Marker #{idx} at {m:02d}:{s:02d}", self.ORANGE)

    def _build_filename(self):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ext = self._fmt.get()
        custom_name = self._name_entry.get().strip()

        if custom_name:
            safe = _sanitize_filename(custom_name)
            if safe:
                return f"{safe}_{ts}.{ext}"

        src_tag = "+".join(self._rec_sources) if self._rec_sources else "rec"
        duration = self._last_duration if self._last_duration > 0 else self.engine.elapsed()
        mins = int(duration // 60)
        secs = int(duration % 60)
        dur_tag = f"{mins}m{secs:02d}s" if mins > 0 else f"{secs}s"

        return f"recording_{ts}_{src_tag}_{dur_tag}.{ext}"

    def _save(self, data, category=None, markers=None):
        if data is None or len(data) == 0:
            msg = self.engine.last_error or "No audio captured"
            self._show_status(msg, self.DANGER)
            return

        filename = self._build_filename()

        save_dir = self.recordings_path
        if category:
            save_dir = os.path.join(self.recordings_path, category)
            os.makedirs(save_dir, exist_ok=True)

        filepath = os.path.join(save_dir, filename)

        try:
            sf.write(filepath, data, self.engine.sample_rate)
            if markers:
                self._write_markers(filepath, markers)
            display = f"{category}/{filename}" if category else filename
            self._show_status(f"Saved: {display}", self.SUCCESS)
        except Exception as e:
            self._show_status(f"Save failed: {e}", self.DANGER)

    def _write_markers(self, audio_path, markers):
        base = audio_path.rsplit('.', 1)[0]
        marker_path = base + '.markers.txt'
        try:
            with open(marker_path, 'w', encoding='utf-8') as f:
                for ts, label in markers:
                    m = int(ts // 60)
                    s = int(ts % 60)
                    ms = int((ts % 1) * 1000)
                    f.write(f"{m:02d}:{s:02d}.{ms:03d} - {label}\n")
        except Exception:
            pass

    def _handle_post_save(self, action, audio, category, updated_cats, markers):
        if updated_cats != self._categories:
            self._categories = updated_cats
            self._save_config()

        if action == "discard":
            self._show_status("Recording discarded", self.TEXT_DIM)
            return

        self._save(audio, category=category, markers=markers)

    def _load_config(self):
        default_cats = ["Химия", "Вброс предлога", "Место есть"]
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())
                cats = data.get('categories', default_cats)
                return cats if isinstance(cats, list) and len(cats) > 0 else default_cats
        except Exception:
            return list(default_cats)

    def _save_config(self):
        try:
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump({'categories': self._categories}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _open_folder(self):
        os.startfile(self.recordings_path)

    def _tick_timer(self):
        if self.engine.is_recording:
            t = self.engine.elapsed()
            h, r = divmod(int(t), 3600)
            m, s = divmod(r, 60)
            self._timer.configure(text=f"{h:02d}:{m:02d}:{s:02d}")
        self.after(200, self._tick_timer)

    def _tick_meters(self):
        if self.engine.is_recording and not self.engine.is_paused:
            self._mic_bar.set(self.engine.mic_level)
            self._sys_bar.set(self.engine.system_level)
            self.engine.mic_level *= 0.85
            self.engine.system_level *= 0.85

        if self.engine.last_error and self.engine.is_recording:
            self._show_status(self.engine.last_error, self.WARNING)
            self.engine.last_error = None

        self.after(50, self._tick_meters)


def main():
    app = RecorderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
