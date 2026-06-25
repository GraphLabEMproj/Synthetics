import sys
import os
import json
import copy
import numpy as np
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot, Property, QThread, QMutex
from PySide6.QtGui import QImage, QColor
from PySide6.QtQuick import QQuickImageProvider

PROJECT_ROOT = str(Path(__file__).parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── parameter metadata ─────────────────────────────────────────────────────────

def _param_meta(key: str, value) -> dict:
    k = key.lower()
    is_int = isinstance(value, int) and not isinstance(value, bool)
    if any(x in k for x in ['color_mean', 'addcolor_mean', 'centerline_color_mean', 'spam_color_mean']):
        return {'min': 0.0, 'max': 255.0, 'step': 1.0 if is_int else 0.5, 'decimals': 0 if is_int else 1}
    if any(x in k for x in ['color_std', 'addcolor_std', 'centerline_color_std', 'spam_color_std']):
        return {'min': 0.0, 'max': 60.0, 'step': 0.5, 'decimals': 1}
    if 'color_diff_mean' in k:
        return {'min': 0.0, 'max': 150.0, 'step': 0.5, 'decimals': 1}
    if 'color_diff_std' in k:
        return {'min': 0.0, 'max': 50.0, 'step': 0.5, 'decimals': 1}
    if 'radius' in k:
        return {'min': 1.0, 'max': 15.0, 'step': 1.0, 'decimals': 0}
    if 'sigma' in k:
        return {'min': 0.05, 'max': 6.0, 'step': 0.05, 'decimals': 2}
    if 'gausse_noise_value' in k:
        return {'min': 0.0, 'max': 100.0, 'step': 1.0, 'decimals': 0}
    if 'noise_w' in k:
        return {'min': 1.0, 'max': 15.0, 'step': 1.0, 'decimals': 0}
    if 'space_thickness_max' in k:
        return {'min': 0.0, 'max': 10.0, 'step': 0.5, 'decimals': 1}
    if 'thickness' in k:
        return {'min': 0.0, 'max': 10.0, 'step': 0.5, 'decimals': 1}
    if 'border_w' in k:
        return {'min': 0.0, 'max': 10.0, 'step': 0.5, 'decimals': 1}
    if 'mit_len' in k:
        return {'min': 0.0, 'max': 500.0, 'step': 1.0, 'decimals': 0}
    if 'poisson_noise' in k:
        return {'min': 0.0, 'max': 200.0, 'step': 1.0, 'decimals': 0}
    if is_int:
        return {'min': 0.0, 'max': float(max(255, value * 3)), 'step': 1.0, 'decimals': 0}
    return {'min': 0.0, 'max': max(255.0, float(value) * 3 if value else 10.0), 'step': 0.5, 'decimals': 1}


_GROUP_ORDER  = ['main','axon','membrane','mitohondrion','mit','psd','vesicles','spam','poisson','other']
_GROUP_LABELS = {
    'main':'Фон', 'axon':'Аксон', 'membrane':'Мембраны',
    'mitohondrion':'Митохондрии', 'mit':'Митохондрии (длина)',
    'psd':'PSD', 'vesicles':'Везикулы',
    'spam':'Спам-компоненты', 'poisson':'Шум', 'other':'Прочее',
}

def _get_group(key: str) -> str:
    k = key.lower()
    for prefix in ['mitohondrion','membrane','vesicles','axon','psd','main','spam']:
        if k.startswith(prefix):
            return prefix
    if k.startswith('mit_'):   return 'mit'
    if 'poisson' in k:         return 'poisson'
    return 'other'

def _fmt_label(key: str) -> str:
    label = key
    for p in ['main_','axon_','membrane_','mitohondrion_','mit_','psd_','vesicles_','spam_']:
        if label.startswith(p):
            label = label[len(p):]
            break
    return label.replace('_',' ').title()

def build_param_groups(params: dict) -> list:
    groups: dict[str, list] = {}
    for key, value in params.items():
        g = _get_group(key)
        if g not in groups:
            groups[g] = []
        meta = _param_meta(key, value)
        groups[g].append({'key': key, 'label': _fmt_label(key), 'value': float(value), **meta})
    return [{'name': _GROUP_LABELS.get(g, g.title()), 'params': groups[g]}
            for g in _GROUP_ORDER if g in groups]


# ── mask constants (order matches DrawsLayerAndMask return tuple) ───────────────

MASK_PROVIDER_KEYS  = ['layer','mask_psd','mask_axon','mask_mem','mask_mito','mask_mito_b','mask_ves']
MASK_DISPLAY_LABELS = ['Оригинал','PSD','Аксон','Мембраны','Митохондрии','Границы мито','Везикулы']


# ── image provider ─────────────────────────────────────────────────────────────

class SyntheticsImageProvider(QQuickImageProvider):
    def __init__(self):
        super().__init__(QQuickImageProvider.ImageType.Pixmap)
        self._images: dict[str, object] = {}
        self._mutex = QMutex()

    def requestPixmap(self, image_id: str, size, requestedSize):
        from PySide6.QtGui import QPixmap
        key = image_id.split('/')[0]
        self._mutex.lock()
        img = self._images.get(key)
        self._mutex.unlock()
        if img is None:
            pix = QPixmap(512, 512)
            pix.fill(QColor(40, 40, 50))
            return pix
        return img

    def set_array(self, key: str, arr):
        if arr is None:
            return
        from PySide6.QtGui import QPixmap
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            h, w = arr.shape[:2]
            qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
        elif arr.ndim == 2:
            h, w = arr.shape
            u8 = arr.astype(np.uint8)
            qimg = QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            return
        pix = QPixmap.fromImage(qimg.copy())
        self._mutex.lock()
        self._images[key] = pix
        self._mutex.unlock()

    def set_png_bytes(self, key: str, data: bytes):
        from PySide6.QtGui import QPixmap
        pix = QPixmap()
        pix.loadFromData(data)
        self._mutex.lock()
        self._images[key] = pix
        self._mutex.unlock()


# ── stdout/stderr → logMessage forwarder ───────────────────────────────────────

class _StreamForwarder:
    """Line-buffered wrapper that forwards print() output to a Qt signal."""
    def __init__(self, emit_fn):
        self._emit = emit_fn
        self._buf  = ""

    def write(self, text: str):
        self._buf += text
        while '\n' in self._buf:
            line, self._buf = self._buf.split('\n', 1)
            if line:          # skip blank lines
                self._emit(line)

    def flush(self):
        if self._buf.strip():
            self._emit(self._buf)
            self._buf = ""

    # make it look like a real file so cv2/numpy don't complain
    fileno  = lambda self: -1
    isatty  = lambda self: False
    readable = lambda self: False
    writable = lambda self: True


# ── worker thread ──────────────────────────────────────────────────────────────

class GenerationWorker(QThread):
    # all 7 outputs: layer + 6 masks
    finished   = Signal(object, object, object, object, object, object, object)
    progress   = Signal(int, int)
    logMessage = Signal(str)
    error      = Signal(str)

    def __init__(self, mode: str, params: dict, gen_params: dict):
        super().__init__()
        self._mode = mode
        self._params = params
        self._gen_params = gen_params

    def run(self):
        import contextlib
        log = self.logMessage.emit
        forwarder = _StreamForwarder(log)
        with contextlib.redirect_stdout(forwarder), contextlib.redirect_stderr(forwarder):
            self._run_inner()

    def _run_inner(self):
        log = self.logMessage.emit
        try:
            log("▶ Установка параметров...")
            import settings
            settings.PARAM.update(self._params)

            log("▶ Загрузка генератора...")
            from src.container.main_field import Form

            sz = self._gen_params.get('size', 256)
            gp = self._gen_params
            form = Form((sz, sz))

            if self._mode == 'single':
                log("▶ Генерация органелл...")
                lst = form.createListGeneration(
                    gp.get('count_psd', 3), gp.get('count_axon', 1),
                    gp.get('count_vesicles', 3), gp.get('count_mito', 3),
                    gp.get('count_spam', 5))
                log(f"▶ Отрисовка {sz}×{sz}...")
                results = form.DrawsLayerAndMask(lst)
                log(f"✓ Готово  {sz}×{sz} px")
                self.finished.emit(*results)

            else:
                from src.container.output import SaveGeneration
                n        = gp.get('count', 10)
                dir_save = gp.get('dir_save', 'dataset/ui_dataset')
                start_i  = gp.get('start_index', 0)
                results  = None
                # output.py uses os.mkdir (one level only) — pre-create tree
                os.makedirs(dir_save, exist_ok=True)
                for i in range(n):
                    self.progress.emit(i + 1, n)
                    log(f"▶ Кадр {i + 1}/{n}...")
                    import settings as s
                    color = s.normal_randint(s.PARAM['main_color_mean'], s.PARAM['main_color_std'])
                    form.backgroundСolor = (color, color, color)
                    lst = form.createListGeneration(
                        gp.get('count_psd', 3), gp.get('count_axon', 1),
                        gp.get('count_vesicles', 3), gp.get('count_mito', 3),
                        gp.get('count_spam', 5))
                    results = form.DrawsLayerAndMask(lst)
                    SaveGeneration(*results, i, dir_save, start_i)
                log(f"✓ Датасет ({n} кадров) → {dir_save}")
                self.finished.emit(*results)

        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())


# ── backend QObject ────────────────────────────────────────────────────────────

class Backend(QObject):
    paramsChanged      = Signal(str)
    imageReady         = Signal(str)
    generationProgress = Signal(int, int)
    generationDone     = Signal()
    generationBusy     = Signal(bool)
    errorOccurred      = Signal(str)
    configPathChanged  = Signal(str)
    modeChanged        = Signal(str)
    logMessage         = Signal(str)
    histListChanged    = Signal(str)   # JSON list of provider keys, newest first

    def __init__(self, provider: SyntheticsImageProvider, parent=None):
        super().__init__(parent)
        self._provider    = provider
        self._params: dict = {}
        self._config_path = ''
        self._mode        = 'single'
        self._gen_params  = {
            'count_psd': 3, 'count_axon': 1, 'count_vesicles': 3,
            'count_mito': 3, 'count_spam': 5,
            'size': 256, 'count': 10,
            'dir_save': 'dataset/ui_dataset', 'start_index': 0,
        }
        self._worker      = None
        self._counter     = 0
        self._last_layer  = None
        self._hist_keys: list[str] = []
        self._max_hists   = 5

    # ── config ─────────────────────────────────────────────────────────────────

    @Slot(str)
    def loadConfig(self, path: str):
        path = _fix_path(path)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            # strip comment keys (non-numeric values)
            self._params = {k: v for k, v in raw.items() if isinstance(v, (int, float))}
            self._config_path = path
            self.configPathChanged.emit(os.path.basename(path))
            self.paramsChanged.emit(json.dumps(build_param_groups(self._params), ensure_ascii=False))
        except Exception as e:
            self.errorOccurred.emit(f"Ошибка загрузки конфига: {e}")

    @Slot(str)
    def saveConfig(self, path: str):
        path = _fix_path(path)
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._params, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.errorOccurred.emit(f"Ошибка сохранения: {e}")

    @Slot()
    def saveConfigInPlace(self):
        if self._config_path:
            self.saveConfig(self._config_path)

    # ── params ─────────────────────────────────────────────────────────────────

    @Slot(str, float)
    def updateParam(self, key: str, value: float):
        if key in self._params:
            orig = self._params[key]
            self._params[key] = int(round(value)) if isinstance(orig, int) and not isinstance(orig, bool) else value

    # ── generation ─────────────────────────────────────────────────────────────

    @Slot()
    def generateStructure(self):
        """Stub for generator_3d. On main branch runs normal generation."""
        self.logMessage.emit("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.logMessage.emit("ℹ  «Сгенерировать основу» — функция ветки generator_3d.")
        self.logMessage.emit("   В ветке main выполняется полная генерация слоя.")
        self.logMessage.emit("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self.redraw()

    @Slot()
    def redraw(self):
        if self._worker and self._worker.isRunning():
            return
        if not self._params:
            self.errorOccurred.emit("Сначала загрузите конфиг-файл")
            return
        self.generationBusy.emit(True)
        self._worker = GenerationWorker(
            self._mode, copy.deepcopy(self._params), self._gen_params.copy())
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self.generationProgress)
        self._worker.logMessage.connect(self.logMessage)
        self._worker.start()

    def _on_done(self, layer, mask_psd, mask_axon, mask_mem, mask_mito, mask_mito_b, mask_ves):
        self._last_layer = layer
        self._counter   += 1
        def _g(m):   # 3-ch mask → 2D grayscale
            return np.ascontiguousarray(m[:, :, 0]) if m is not None and m.ndim == 3 else m
        self._provider.set_array('layer',      layer)
        self._provider.set_array('mask_psd',   _g(mask_psd))
        self._provider.set_array('mask_axon',  _g(mask_axon))
        self._provider.set_array('mask_mem',   _g(mask_mem))
        self._provider.set_array('mask_mito',  _g(mask_mito))
        self._provider.set_array('mask_mito_b',_g(mask_mito_b))
        self._provider.set_array('mask_ves',   _g(mask_ves))
        self.imageReady.emit(str(self._counter))
        self.generationDone.emit()
        self.generationBusy.emit(False)

    def _on_error(self, msg: str):
        self.errorOccurred.emit(msg)
        self.generationBusy.emit(False)

    # ── histograms ─────────────────────────────────────────────────────────────

    @Slot()
    def buildHistograms(self):
        if self._last_layer is None:
            self.errorOccurred.emit("Сначала сгенерируйте изображение")
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io

            gray = self._last_layer[:,:,0] if self._last_layer.ndim == 3 else self._last_layer
            n    = len(self._hist_keys) + 1

            fig, ax = plt.subplots(figsize=(5.8, 2.6), facecolor='#1e1e2e')
            ax.set_facecolor('#2a2a3e')
            ax.hist(gray.flatten(), bins=128, color='#7aa2f7', alpha=0.88, edgecolor='none')
            ax.set_xlabel('Интенсивность', color='#c0caf5', fontsize=7)
            ax.set_ylabel('Пиксели',       color='#c0caf5', fontsize=7)
            ax.tick_params(colors='#a9b1d6', labelsize=6)
            for sp in ax.spines.values(): sp.set_color('#3b3b5e')
            ax.set_title(f'#{n}', color='#9aa5ce', fontsize=8, loc='right')
            plt.tight_layout(pad=0.4)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)

            self._counter += 1
            key = f'hist_{self._counter}'
            self._provider.set_png_bytes(key, buf.read())
            self._hist_keys.insert(0, key)
            self._hist_keys = self._hist_keys[:self._max_hists]
            self.histListChanged.emit(json.dumps(self._hist_keys))

        except ImportError:
            self.errorOccurred.emit("Установите matplotlib: pip install matplotlib")
        except Exception:
            import traceback
            self.errorOccurred.emit(traceback.format_exc())

    @Slot(int)
    def setMaxHistograms(self, n: int):
        self._max_hists  = max(1, n)
        self._hist_keys  = self._hist_keys[:self._max_hists]
        self.histListChanged.emit(json.dumps(self._hist_keys))

    # ── generation settings ────────────────────────────────────────────────────

    @Slot(str)
    def setMode(self, mode: str):
        self._mode = mode
        self.modeChanged.emit(mode)

    @Slot(str, 'QVariant')
    def setGenParam(self, key: str, value):
        self._gen_params[key] = value

    @Property(str, notify=configPathChanged)
    def configPath(self):
        return self._config_path

    @Property(str, notify=modeChanged)
    def mode(self):
        return self._mode


def _fix_path(path: str) -> str:
    path = path.replace('file:///', '').replace('file://', '')
    if os.name == 'nt' and path.startswith('/') and len(path) > 2 and path[2] == ':':
        path = path[1:]
    return path
