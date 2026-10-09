"""Teaser vertical (9:16) estilo tráiler para el evento "Café Luca".

Muestra solo detalles, desenfoques y pistas de la decoración: nunca el montaje completo.
Uso: python3 render_teaser.py  ->  genera teaser_cafe_luca.mp4 y portada.jpg
"""
import functools
import math
import multiprocessing as mp
import os
import subprocess
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 30
DUR = 24.0
SR = 44100

NAVY = (24, 42, 72)
CREAM = (246, 238, 224)
GOLD = (214, 176, 96)

SERIF = "/usr/share/fonts/truetype/crosextra/Caladea-Regular.ttf"
SERIF_IT = "/usr/share/fonts/truetype/crosextra/Caladea-Italic.ttf"
SANS = "/usr/share/fonts/opentype/inter/Inter-Light.otf"

lugar = ImageOps.exif_transpose(Image.open(os.path.join(HERE, "assets/lugar.jpg"))).convert("RGB")
deco = ImageOps.exif_transpose(Image.open(os.path.join(HERE, "assets/decoracion.jpg"))).convert("RGB")
# La foto de la decoración se trabajó a 1402x1122; las coordenadas abajo usan esa escala.
DS = deco.width / 1402.0
LS = lugar.width / 1500.0


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def crop_view(img, cx, cy, cw, scale):
    """Recorta un rectángulo 9:16 centrado en (cx, cy) de ancho cw y lo lleva a WxH."""
    cw = cw * scale
    ch = cw * H / W
    box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
    return img.transform((W, H), Image.Transform.EXTENT, box, Image.Resampling.BICUBIC)


# ---------- capas fijas ----------
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
_r = np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.62)) ** 2)
VIGNETTE = np.clip(1.0 - 0.75 * _r ** 2.2, 0.15, 1.0)[..., None]
rng = np.random.default_rng(7)
GRAIN = [rng.normal(0, 1, (H // 2, W // 2)).astype(np.float32) for _ in range(6)]


_pr = np.random.default_rng(11)
SPARKS = [dict(x=_pr.uniform(0, W), y=_pr.uniform(0, H), r=_pr.uniform(6, 22),
               vy=_pr.uniform(-60, -15), vx=_pr.uniform(-12, 12), ph=_pr.uniform(0, 6.3),
               b=_pr.uniform(0.25, 0.8)) for _ in range(38)]
SPARK_RGB = np.array([1.0, 0.82, 0.48], np.float32)


def sparkles(a, t, amount):
    if amount <= 0:
        return a
    for s in SPARKS:
        x = (s["x"] + s["vx"] * t) % W
        y = (s["y"] + s["vy"] * t) % H
        r = s["r"]
        tw = 0.55 + 0.45 * math.sin(t * 2.2 + s["ph"])
        x0, x1 = int(max(x - 3 * r, 0)), int(min(x + 3 * r, W))
        y0, y1 = int(max(y - 3 * r, 0)), int(min(y + 3 * r, H))
        if x1 <= x0 or y1 <= y0:
            continue
        gx = np.exp(-((np.arange(x0, x1) - x) / r) ** 2)
        gy = np.exp(-((np.arange(y0, y1) - y) / r) ** 2)
        a[y0:y1, x0:x1] += (gy[:, None] * gx[None, :])[..., None] * SPARK_RGB * (s["b"] * tw * amount)
    return a


def grade(arr, warm=0.0, lift=0.0):
    a = arr.astype(np.float32) / 255.0
    a = np.clip((a - 0.5) * 1.08 + 0.5, 0, 1)  # contraste
    a[..., 0] *= 1 + 0.05 * warm
    a[..., 2] *= 1 - 0.04 * warm + 0.03
    a = a * VIGNETTE + lift
    return a


def finish(a, fi):
    g = GRAIN[fi % len(GRAIN)]
    g = np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None]
    a = a + g * 0.022
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def text_layer(lines, t_in, t, size=64, font=SERIF, color=CREAM, y=None, spacing=1.5,
               track=0, glow=True, rise=40):
    """Devuelve (RGBA layer, alpha) con aparición suave y leve subida."""
    p = ease_out((t - t_in) / 0.7)
    y0 = (H - len(lines) * size * spacing) / 2 if y is None else y
    y0 = int(round(y0 + (1 - p) * rise))
    return _text_img(tuple(lines), size, font, color, y0, spacing, track, glow), p


@functools.lru_cache(maxsize=48)
def _text_img(lines, size, font, color, y0, spacing, track, glow):
    f = ImageFont.truetype(font, size)
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for i, ln in enumerate(lines):
        widths = [d.textlength(ch, font=f) + track for ch in ln]
        tw = sum(widths) - track
        x = (W - tw) / 2
        for ch, cw in zip(ln, widths):
            d.text((x, y0 + i * size * spacing), ch, font=f, fill=color + (255,))
            x += cw
    if glow:
        # sombra oscura para leer sobre fondos claros + brillo suave
        sh = Image.new("RGBA", (W, H), (8, 14, 28, 0))
        sh.putalpha(layer.getchannel("A").filter(ImageFilter.GaussianBlur(18)).point(lambda v: min(255, v * 3)))
        g = layer.filter(ImageFilter.GaussianBlur(10))
        layer = Image.alpha_composite(Image.alpha_composite(sh, g), layer)
    return layer


def over(base_arr, layer, alpha):
    if alpha <= 0:
        return base_arr
    la = np.asarray(layer, dtype=np.float32) / 255.0
    a = la[..., 3:4] * alpha
    return base_arr * (1 - a) + la[..., :3] * a


def fade_out(t, t_end, d=0.6):
    return 1.0 - ease((t - (t_end - d)) / d)


# ---------- planos de detalle (nunca el conjunto completo) ----------
DETAILS = [  # (cx, cy, ancho, zoom_ini, zoom_fin, texto)
    (95, 840, 170, 1.15, 0.95, ["Cada detalle..."]),
    (335, 600, 150, 0.9, 1.12, ["...se está diseñando"]),
    (225, 860, 120, 1.1, 0.92, ["con amor"]),
    (425, 155, 170, 0.95, 1.15, ["pieza por pieza"]),
    (775, 385, 90, 1.2, 0.98, ["y con un toque de oro"]),
]


def render_frame(fi):
    t = fi / FPS
    a = np.zeros((H, W, 3), np.float32)

    if t < 2.8:  # S0: gancho: detalle muy desenfocado en movimiento + frase
        img = crop_view(deco, (330 + 40 * t) * DS, 620 * DS, 260 * DS, 1.1 - 0.04 * t)
        img = img.resize((W // 4, H // 4)).filter(ImageFilter.GaussianBlur(9)).resize((W, H))
        a = grade(np.asarray(img), warm=0.5) * 0.42 * ease(t / 0.25)
        a = sparkles(a, t, 0.9)
        lay, p = text_layer(["Algo muy especial", "se está preparando..."], 0.15, t, size=88,
                            font=SERIF_IT)
        a = over(a, lay, p)
        a = a * fade_out(t, 2.8, 0.35)

    elif t < 6.6:  # S1: el lugar, empuje lento hacia el letrero
        u = ease((t - 2.8) / 3.8)
        cx = 760 * LS + (585 - 760) * LS * u
        cy = 1000 * LS + (1340 - 1000) * LS * u
        cw = 1120 * LS + (560 - 1120) * LS * u
        shake = math.sin(t * 23) * 3 * (1 - u)
        img = crop_view(lugar, cx + shake, cy, cw, 1.0)
        a = grade(np.asarray(img), warm=0.6) * ease((t - 2.8) / 0.4)
        lay, p = text_layer(["En un lugar", "muy cerca de ti..."], 3.3, t, size=76, font=SERIF_IT,
                            y=H * 0.69)
        a = over(a, lay, p * fade_out(t, 6.6, 0.4))

    elif t < 12.6:  # S2: detalles rápidos con enfoque/desenfoque
        k = int((t - 6.6) // 1.2)
        lt = (t - 6.6) - k * 1.2
        cx, cy, cw, z0, z1, txt = DETAILS[k]
        z = z0 + (z1 - z0) * ease(lt / 1.2)
        img = crop_view(deco, cx * DS, cy * DS, cw * DS, z)
        focus = abs(lt - 0.55) / 0.65  # 0 = nítido
        img = img.filter(ImageFilter.GaussianBlur(2 + 22 * focus ** 2))
        a = grade(np.asarray(img), warm=0.4)
        a = sparkles(a, t, 0.5)
        lay, p = text_layer(txt, 0.12, lt, size=86, font=SERIF_IT, y=H * 0.44, rise=25)
        a = over(a, lay, p * fade_out(lt, 1.2, 0.25))

    elif t < 16.2:  # S3: el escudo, casi irreconocible + barrido de luz + latido
        lt = t - 12.6
        img = crop_view(deco, 780 * DS, 668 * DS, 150 * DS, 1.05 - 0.08 * ease(lt / 3.6))
        img = img.filter(ImageFilter.GaussianBlur(38 - 14 * ease(lt / 3.6)))
        a = grade(np.asarray(img), warm=0.2) * 0.6
        sweep = np.exp(-((xx + yy * 0.35 - (lt / 3.6) * (W + H * 0.6) * 1.3 + 200) / 160) ** 2)
        a = a + sweep[..., None] * np.array([0.35, 0.3, 0.2]) * 0.6
        beat = max(0.0, math.sin(lt * math.pi * 1.6)) ** 8
        a = a * (1 + 0.15 * beat)
        lay, p = text_layer(["Un pequeño", "caballero..."], 0.3, lt, size=92, font=SERIF_IT,
                            y=H * 0.30)
        a = over(a, lay, p)
        lay2, p2 = text_layer(["está en camino"], 1.7, lt, size=92, font=SERIF_IT, y=H * 0.30 + 330,
                                color=GOLD)
        a = over(a, lay2, p2)
        a = a * fade_out(t, 16.2, 0.25)

    elif t < 17.0:  # S4: ráfaga tipo tráiler (cortes de 3 cuadros)
        k = int((t - 16.2) * FPS // 3) % len(DETAILS)
        cx, cy, cw, *_ = DETAILS[(k * 2) % len(DETAILS)]
        img = crop_view(deco, cx * DS, cy * DS, cw * DS * 0.8, 1.0).filter(ImageFilter.GaussianBlur(9))
        a = grade(np.asarray(img), warm=0.5)
        if (fi // 3) % 2 == 0:
            a = a * 0.35 + 0.65

    elif t < 20.5:  # S5: el nombre
        lt = t - 17.0
        a[:] = np.array(NAVY, np.float32) / 255.0 * 0.55
        a = a * VIGNETTE
        track = int(4 + 26 * ease_out(lt / 2.5))
        lay, p = text_layer(["CAFÉ LUCA"], 0.05, lt, size=128, font=SERIF, color=CREAM,
                            y=H * 0.43, track=track, rise=0)
        a = over(a, lay, p)
        band = np.exp(-((xx[:, :, None] - (lt - 0.6) / 1.6 * (W + 600) + 300 + yy[:, :, None] * 0.25 - 250) / 90) ** 2)
        mask = np.asarray(lay.getchannel("A"), np.float32)[..., None] / 255.0
        a = a + band * mask * np.array([0.55, 0.42, 0.12], np.float32)
        a = sparkles(a, t, 0.8 * ease(lt / 0.6))
        lw = int(360 * ease_out((lt - 0.4) / 1.2))
        line = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dl = ImageDraw.Draw(line)
        if lw > 0:
            dl.line([(W / 2 - lw, H * 0.43 + 190), (W / 2 + lw, H * 0.43 + 190)], fill=GOLD + (255,), width=3)
            dl.ellipse([W / 2 - 7, H * 0.43 + 183, W / 2 + 7, H * 0.43 + 197], fill=GOLD + (255,))
        a = over(a, line, 1.0)
        if lt < 0.12:  # destello del golpe
            a = a + (0.12 - lt) / 0.12 * 0.8
        a = a * fade_out(t, 20.5, 0.35)

    else:  # S6: cierre
        lt = t - 20.5
        bg = crop_view(deco, 700 * DS, 560 * DS, 640 * DS, 1.0 + 0.04 * lt).filter(ImageFilter.GaussianBlur(40))
        a = grade(np.asarray(bg)) * 0.35 + np.array(NAVY, np.float32) / 255.0 * 0.35
        a = a * ease(lt / 0.5)
        a = sparkles(a, t, 0.8)
        lay, p = text_layer(["MUY PRONTO"], 0.3, lt, size=118, font=SERIF, track=14, y=H * 0.38)
        a = over(a, lay, p)
        lay2, p2 = text_layer(["BABY SHOWER  ·  SAVE THE DATE"], 1.0, lt, size=42, font=SANS,
                              color=GOLD, track=6, y=H * 0.38 + 190, glow=False)
        a = over(a, lay2, p2)
        lay3, p3 = text_layer(["#CaféLuca"], 1.6, lt, size=62, font=SERIF_IT, y=H * 0.38 + 300)
        a = over(a, lay3, p3)
        a = a * fade_out(t, DUR, 0.7)

    # destellos blancos en los cortes clave
    for tc in (6.6, 7.8, 9.0, 10.2, 11.4, 12.6):
        if 0 <= t - tc < 0.1:
            a = a + (1 - (t - tc) / 0.1) * 0.55
    return finish(a, fi)


# ---------- audio sintetizado (sin derechos de autor) ----------
def make_audio(path):
    n = int(DUR * SR)
    tt = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    rng = np.random.default_rng(1)

    def env(start, attack, decay):
        e = np.zeros(n, np.float32)
        i0 = int(start * SR)
        seg = tt[i0:] - start
        e[i0:] = np.minimum(seg / attack, 1.0) * np.exp(-np.maximum(seg - attack, 0) / decay)
        return e

    # dron grave que crece
    swell = np.clip(tt / 16.0, 0, 1) * (tt < 17.0) + (tt >= 20.5) * np.clip((DUR - tt) / 3.5, 0, 1) * 0.6
    out += 0.10 * swell * (np.sin(2 * np.pi * 55 * tt) + 0.6 * np.sin(2 * np.pi * 82.4 * tt + 0.3)
                           + 0.3 * np.sin(2 * np.pi * 110.2 * tt))

    def boom(t0, amp=0.8):
        seg_env = env(t0, 0.005, 0.55)
        f = 38 + 90 * np.exp(-np.maximum(tt - t0, 0) * 9)
        ph = 2 * np.pi * np.cumsum(f) / SR
        return amp * seg_env * np.sin(ph)

    for t0 in (0.3, 2.8, 6.6, 12.6):
        out += boom(t0)
    for t0 in (7.8, 9.0, 10.2, 11.4):
        out += boom(t0, 0.45)
    # gran golpe con ruido
    noise = rng.normal(0, 1, n).astype(np.float32)
    out += boom(17.0, 1.0) + 0.25 * env(17.0, 0.003, 0.25) * noise

    # whooshes antes de cada corte
    def whoosh(t_end, dur, amp):
        e = np.clip((tt - (t_end - dur)) / dur, 0, 1) ** 2 * (tt < t_end)
        k = np.ones(40) / 40
        return amp * e * np.convolve(noise, k, "same")

    for tc in (2.8, 6.6, 7.8, 9.0, 10.2, 11.4, 12.6):
        out += whoosh(tc, 0.35, 0.5)
    out += whoosh(17.0, 0.8, 1.0)  # subida antes del nombre

    # latido en la escena del escudo
    for i in range(6):
        b = 12.9 + i * 0.625
        for d, amp in ((0.0, 0.7), (0.17, 0.45)):
            e = env(b + d, 0.004, 0.09)
            out += amp * e * np.sin(2 * np.pi * 50 * tt)

    # cajita musical (arpegio pentatónico) en detalles y cierre
    notes = [659.3, 784.0, 987.8, 880.0, 784.0, 659.3, 587.3, 659.3]
    starts = list(np.arange(6.75, 12.6, 0.6)) + list(np.arange(20.8, 23.2, 0.4))
    for i, s in enumerate(starts):
        f = notes[i % len(notes)]
        e = env(s, 0.003, 0.5)
        out += 0.09 * e * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt))
    # acorde final
    for f in (329.6, 415.3, 493.9, 659.3):
        out += 0.06 * env(20.5, 0.4, 2.2) * np.sin(2 * np.pi * f * tt)

    # reverb simple
    for dly, g in ((0.071, 0.35), (0.113, 0.25), (0.197, 0.18)):
        dd = int(dly * SR)
        out[dd:] += g * out[:-dd]
    out = np.tanh(out * 1.1) / np.tanh(1.1)
    out *= 0.9 / np.max(np.abs(out))
    pcm = (out * 32767).astype(np.int16)
    stereo = np.stack([pcm, pcm], 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())


def main():
    audio = os.path.join(HERE, "_audio.wav")
    out = os.path.join(HERE, "teaser_cafe_luca.mp4")
    make_audio(audio)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", audio,
           "-c:v", "libx264", "-preset", "medium", "-crf", "24", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    total = int(DUR * FPS)
    with mp.Pool(os.cpu_count()) as pool:
        for fi, frame in enumerate(pool.imap(render_frame, range(total), chunksize=4)):
            proc.stdin.write(frame.tobytes())
            if fi == int(18.5 * FPS):
                Image.fromarray(frame).save(os.path.join(HERE, "portada.jpg"), quality=92)
    proc.stdin.close()
    proc.wait()
    os.remove(audio)
    print("listo:", out)


if __name__ == "__main__":
    main()
