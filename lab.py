"""lab.py - نداء النماذج (مع إعادة محاولة وتبديل تلقائي) + محاكي التجارب الافتراضية"""
import ast
import base64
import json
import re
import time

import fitz
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
import anthropic

MODELS = []                      # app.py بيملاها بقايمة النماذج
LAST_USED = {"model": None, "asked": None}
MAX_CHARS = 30000
SRC_TYPES = ["pdf", "png", "jpg", "jpeg"]


# ================================================================ نداء النماذج
def _ask_gemini(model_name, items, system, key, json_mode):
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name)
    content = [system]
    for kind, data in items:
        content.append({"mime_type": "image/png", "data": data} if kind == "image" else data)
    cfg = {"temperature": 0.2}
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    opts = {"timeout": 120, "retry": None}
    try:
        r = model.generate_content(content, generation_config=cfg, request_options=opts)
    except Exception as e:
        low = str(e).lower()
        if json_mode and ("mime" in low or "json" in low):      # نموذج مايدعمش JSON mode
            cfg.pop("response_mime_type")
            r = model.generate_content(content, generation_config=cfg, request_options=opts)
        else:
            raise
    return r.text


def _ask_claude(model_name, items, system, key, json_mode):
    client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=120)
    content = []
    for kind, data in items:
        if kind == "image":
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": base64.b64encode(data).decode()}})
        else:
            content.append({"type": "text", "text": data})
    kw = dict(model=model_name, max_tokens=8000, system=system,
              messages=[{"role": "user", "content": content}])
    try:
        r = client.messages.create(temperature=0.2, **kw)
    except Exception as e:
        low = str(e).lower()
        if "temperature" in low or "top_p" in low or "sampling" in low:   # نماذج جديدة بترفض temperature
            r = client.messages.create(**kw)
        else:
            raise
    return "".join(b.text for b in r.content if b.type == "text")


def _parse_delay(msg):
    m = re.search(r'(?:retry|try again) in\s+((?:\d+h)?(?:\d+m)?(?:\d+(?:\.\d+)?s)?)', msg)
    if not m or not m.group(1):
        return None
    total = 0.0
    for pat, mult in ((r'(\d+)h', 3600), (r'(\d+)m', 60), (r'(\d+(?:\.\d+)?)s', 1)):
        x = re.search(pat, m.group(1))
        if x:
            total += float(x.group(1)) * mult
    return total


def _classify(msg):
    low = msg.lower()
    if "credit balance" in low:
        return "credit"
    if ("401" in msg or "authentication" in low or "invalid api key" in low
            or "api key not valid" in low or "api_key_invalid" in low):
        return "auth"
    if "limit: 0" in msg:
        return "nofree"
    if "429" in msg or "rate_limit" in low or "rate limit" in low or "quota" in low \
            or "resource_exhausted" in low:
        return "rate"
    if any(w in low for w in ("503", "500", "502", "504", "529", "unavailable", "high demand",
                              "overloaded", "deadline", "timed out", "timeout", "internal",
                              "connection")):
        return "busy"
    return "other"


def call_model(provider, model_name, items, system, key, json_mode=False, fallback=True):
    """بيجرّب النموذج المختار، ولو مشغول/مش شغال بيحاول تاني وبعدين يبدّل لنموذج تاني."""
    chain = [model_name]
    if fallback:
        chain += [m for m in MODELS if m != model_name][:3]
    ask = _ask_claude if provider == "Claude" else _ask_gemini
    last = ""
    for m in chain:
        for attempt in range(3):
            try:
                out = ask(m, items, system, key, json_mode)
                LAST_USED["model"], LAST_USED["asked"] = m, model_name
                return out
            except Exception as e:
                msg = str(e)
                last = msg
                kind = _classify(msg)
                if kind == "credit":
                    raise RuntimeError("رصيد Claude خلص، اشحن الحساب من console.anthropic.com.")
                if kind == "auth":
                    raise RuntimeError("المفتاح غلط أو منتهي.")
                if kind == "busy" and attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                if kind == "rate" and attempt < 2:
                    d = _parse_delay(msg)
                    if d is not None and d <= 15:
                        time.sleep(d + 1)
                        continue
                break                                   # النموذج ده مش نافع، جرّب اللي بعده
    raise RuntimeError("كل النماذج المجرّبة مشغولة أو وصلت للحد. جرّب بعد شوية. (" + last[:140] + ")")


def parse_json_block(raw):
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?', '', raw).strip()
    raw = re.sub(r'```$', '', raw).strip()
    a, b = raw.find('{'), raw.rfind('}')
    if a == -1 or b == -1:
        raise ValueError("الرد مش JSON")
    return json.loads(raw[a:b + 1])


# ================================================================ قراءة المصدر (بدون OCR منفصل)
def _png(pg, side=1280):
    r = pg.rect
    z = side / max(r.width, r.height)
    return pg.get_pixmap(matrix=fitz.Matrix(z, z)).tobytes("png")


def gather(files):
    """نص من الـ PDF اللي فيه نص + صور (الصور والصفحات الممسوحة بتروح للنموذج مباشرة)"""
    texts, imgs = [], []
    for f in files:
        name = f.name.lower()
        if name.endswith(".pdf"):
            doc = fitz.open(stream=f.getvalue(), filetype="pdf")
            for pg in doc:
                t = pg.get_text().strip()
                if len(t) >= 40:
                    texts.append(t)
                elif len(imgs) < 4:
                    imgs.append(_png(pg))
        elif len(imgs) < 4:
            ext = name.rsplit(".", 1)[-1]
            ext = "jpeg" if ext == "jpg" else ext
            imgs.append(_png(fitz.open(stream=f.getvalue(), filetype=ext)[0]))
    return "\n\n".join(texts)[:MAX_CHARS], imgs


# ================================================================ البرومبت
SIM_PROMPT = """
You build interactive VIRTUAL-LAB simulations for university science students (chemistry, biology, physics, sports science).
INPUT: lecture material (text and/or images) and/or a topic typed by the student.
TASK: create @N@ simulation(s). @FOCUS@ @KIND@

Choose the TYPE from the topic itself, not from habit:
- "experiment": the student ADDS substances and changes conditions and WATCHES what happens over time. Use it for any lab experiment, reaction, process or apparatus (yeast + hydrogen peroxide, titration, enzyme vs temperature, dissolving, electrolysis, heating, fermentation, osmosis...).
- "graph": ONLY a pure equation/relationship the student wants to explore on axes (Arrhenius, Henderson-Hasselbalch...).
If the material is an experiment, a reaction or a process, it MUST be "experiment".

EXPERIMENT format (example for yeast + hydrogen peroxide; adapt it, do not copy it for other topics):
{"type":"experiment","title":"Yeast decomposing hydrogen peroxide",
 "description":"Yeast contains catalase, which splits H₂O₂ into water and oxygen. Add H₂O₂ and yeast, change the temperature, and watch the O₂ released.",
 "steps":["Add hydrogen peroxide to the flask","Add yeast and watch the foam","Change the temperature and compare"],
 "vessel":"flask",
 "reagents":[{"id":"h2o2","name":"Hydrogen peroxide 3%","unit":"mL","state":"liquid","color":"#dbe9f4","step":10,"max":100,"ml_per_unit":1},
             {"id":"yeast","name":"Yeast (catalase)","unit":"g","state":"solid","color":"#c8a165","step":1,"max":5,"ml_per_unit":0}],
 "trigger":["h2o2","yeast"],
 "controls":[{"name":"T","label":"Temperature","unit":"°C","min":0,"max":70,"default":25,"step":1}],
 "constants":{"kc":0.01,"Topt":37,"width":18},
 "derived":[{"name":"k","expr":"kc*yeast*exp(-((T-Topt)/width)^2)"}],
 "outputs":[{"id":"o2","label":"O₂ released","unit":"mL","expr":"10.8*h2o2*(1-exp(-k*t))","min":0,"max":1100},
            {"id":"rate","label":"Rate","unit":"mL/s","expr":"10.8*h2o2*k*exp(-k*t)","min":0,"max":60}],
 "visuals":{"bubbles":"min(1,40*k*exp(-k*t))*min(1,h2o2/5)","foam":"min(1,0.0012*10.8*h2o2*(1-exp(-k*t)))","heat":"min(1,20*k*exp(-k*t))*min(1,h2o2/5)"},
 "formula":"2 H₂O₂ → 2 H₂O + O₂",
 "observe":["More yeast gives faster O₂ release","The rate peaks near 37 °C and drops at high temperature (enzyme denatures)"],
 "rtl":false}
vessel is one of: flask, beaker, test_tube, cylinder.
visuals (all optional, each an expression giving 0..1): bubbles, foam, precipitate (solid settling), heat (exothermic glow/steam), color_mix (blend toward "color_to": "#hex").

GRAPH format:
{"type":"graph","title":"...","description":"...","formula":"...","variables":[{"name":"T","label":"Temperature","unit":"K","min":250,"max":500,"default":300,"step":1}],
 "x":"T","x_label":"Temperature (K)","y_label":"k","log_y":false,"constants":{"R":8.314},
 "curves":[{"label":"...","expr":"..."}],"readouts":[{"label":"...","expr":"..."}],"observe":["..."],"rtl":false}

RULES
1. Expressions: one math expression. Operators + - * / ** (or ^) and parentheses; functions exp, ln, log (natural), log10, log2, sqrt, abs, sin, cos, tan, sinh, cosh, tanh, min, max, pow; constants pi and e. No comparisons, no if. Put negative numbers in parentheses when raised to a power.
2. In experiments an expression may use: reagent ids (amount added so far, in that reagent's unit), control names, constants, derived names, t (seconds since the reaction started) and V (total liquid volume in mL). In graphs: variables and constants only.
3. Names: letters, digits, underscore, start with a letter, no spaces; never a function name, e, pi, t or V (t is allowed as a variable name only in graphs).
4. Reagents are things the student adds (liquid or solid, with a realistic colour). Conditions (temperature, pH, light, current...) are controls.
5. The vessel starts EMPTY: every output and visual must be 0 or neutral until the needed reagents are added (multiply by min(1, amount/small)). Stay consistent with stoichiometry (limiting reagent) and real units.
6. Time is in seconds. At default settings the change should be clearly visible within 20-120 s.
7. Use real textbook chemistry/biology/physics and the numbers in the material; never invent formulas.
8. Limits: 5 reagents, 3 controls, 6 derived, 4 outputs, 3 curves.
9. Write all text in @LANG@; keep symbols and units standard, Unicode subscripts (H₂O). Set "rtl": true only if the text is Arabic.
10. No Markdown. Output valid JSON only, no text outside it: {"simulations":[ ... ]}
"""

# ================================================================ تقييم آمن للمعادلات
SAFE_FUNCS = {
    "exp": np.exp, "ln": np.log, "log": np.log, "log10": np.log10, "log2": np.log2,
    "sqrt": np.sqrt, "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh, "abs": np.abs,
    "min": np.minimum, "max": np.maximum, "pow": np.power,
}
JS_FUNCS = {
    "exp": "Math.exp", "ln": "Math.log", "log": "Math.log", "log10": "Math.log10",
    "log2": "Math.log2", "sqrt": "Math.sqrt", "sin": "Math.sin", "cos": "Math.cos",
    "tan": "Math.tan", "sinh": "Math.sinh", "cosh": "Math.cosh", "tanh": "Math.tanh",
    "abs": "Math.abs", "min": "Math.min", "max": "Math.max", "pow": "Math.pow",
}
TWO_ARGS = {"min", "max", "pow"}
NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
HEX = re.compile(r'#[0-9a-fA-F]{6}')


def _prep(expr):
    expr = str(expr).strip().replace("^", "**").replace("×", "*").replace("·", "*").replace("−", "-")
    if not expr or len(expr) > 300:
        raise ValueError("bad expression")
    return ast.parse(expr, mode="eval")


def safe_eval(expr, env):
    tree = _prep(expr)

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) \
                and not isinstance(n.value, bool):
            return np.float64(n.value)
        if isinstance(n, ast.Name) and n.id in env:
            return env[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                   ast.Div: lambda: a / b, ast.Pow: lambda: np.power(a, b)}
            if type(n.op) in ops:
                return ops[type(n.op)]()
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in SAFE_FUNCS \
                and not n.keywords and len(n.args) == (2 if n.func.id in TWO_ARGS else 1):
            return SAFE_FUNCS[n.func.id](*[ev(x) for x in n.args])
        raise ValueError("expression not allowed")

    with np.errstate(all="ignore"):
        return ev(tree)


def to_js(expr, names):
    """يحوّل المعادلة المتحقَّق منها لكود JS مقفول بالكامل (كل الأقواس صريحة)"""
    tree = _prep(expr)

    def g(n):
        if isinstance(n, ast.Expression):
            return g(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) \
                and not isinstance(n.value, bool):
            if not np.isfinite(float(n.value)):
                raise ValueError("bad number")
            return repr(float(n.value))
        if isinstance(n, ast.Name):
            if n.id == "pi":
                return "Math.PI"
            if n.id == "e":
                return "Math.E"
            if n.id in names:
                return '$["%s"]' % n.id
            raise ValueError("unknown name " + n.id)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return "(%s%s)" % ("-" if isinstance(n.op, ast.USub) else "+", g(n.operand))
        if isinstance(n, ast.BinOp):
            a, b = g(n.left), g(n.right)
            sym = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}.get(type(n.op))
            if sym:
                return "(%s %s %s)" % (a, sym, b)
            if isinstance(n.op, ast.Pow):
                return "Math.pow(%s, %s)" % (a, b)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in JS_FUNCS \
                and not n.keywords and len(n.args) == (2 if n.func.id in TWO_ARGS else 1):
            return "%s(%s)" % (JS_FUNCS[n.func.id], ", ".join(g(x) for x in n.args))
        raise ValueError("expression not allowed")

    return g(tree)


def _compile(expr, names, env):
    """JS صالح لو المعادلة سليمة وبتطلّع رقم عند القيم التجريبية، وإلا None"""
    try:
        js = to_js(expr, names)
        v = float(np.asarray(safe_eval(expr, env), dtype=float).reshape(-1)[0])
        return js if np.isfinite(v) else None
    except Exception:
        return None


def _f(v, default=None):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _vars(items, reserved, limit, taken):
    out = []
    for v in (items or [])[:limit]:
        if not isinstance(v, dict):
            continue
        name = str(v.get("name", "")).strip()
        lo, hi = _f(v.get("min")), _f(v.get("max"))
        if (not NAME.fullmatch(name) or name in reserved or name in taken
                or lo is None or hi is None or not lo < hi):
            continue
        de = min(max(_f(v.get("default"), (lo + hi) / 2), lo), hi)
        step = _f(v.get("step"), (hi - lo) / 100)
        if not step or step <= 0 or step > hi - lo:
            step = (hi - lo) / 100
        taken.add(name)
        out.append({"name": name, "label": str(v.get("label") or name),
                    "unit": str(v.get("unit") or ""), "min": lo, "max": hi,
                    "default": de, "step": step})
    return out


def clean_sim(spec):
    """بيتحقق من اللي النموذج طلّعه ويشيل أي حاجة بايظة"""
    if not isinstance(spec, dict):
        return None
    typ = spec.get("type")
    if typ not in ("experiment", "graph"):
        typ = "experiment" if spec.get("reagents") else "graph"
    base = {
        "type": typ,
        "title": str(spec.get("title") or "Simulation"),
        "description": str(spec.get("description") or ""),
        "formula": str(spec.get("formula") or ""),
        "observe": [str(o) for o in (spec.get("observe") or [])][:5],
        "steps": [str(o) for o in (spec.get("steps") or [])][:6],
        "rtl": bool(spec.get("rtl")),
    }
    reserved = set(SAFE_FUNCS) | {"e", "pi"}
    if typ == "experiment":
        reserved |= {"t", "V"}
    taken, consts = set(), {}
    for k, v in (spec.get("constants") or {}).items():
        val = _f(v)
        if val is not None and NAME.fullmatch(str(k)) and str(k) not in reserved:
            consts[str(k)] = val
            taken.add(str(k))
    env = {k: np.float64(v) for k, v in consts.items()}
    env.update({"pi": np.pi, "e": np.e})

    # ------------------------------------------------------------ رسم معادلة
    if typ == "graph":
        variables = _vars(spec.get("variables"), reserved, 5, taken)
        if not variables:
            return None
        names = set(taken)
        for v in variables:
            env[v["name"]] = np.float64(v["default"])
        xname = str(spec.get("x", ""))
        if xname not in names:
            xname = variables[0]["name"]
        xv = next(v for v in variables if v["name"] == xname)
        arr = {**env, xname: np.linspace(xv["min"], xv["max"], 20)}
        curves = []
        for c in (spec.get("curves") or [])[:3]:
            try:
                js = _compile(c.get("expr", ""), names, arr)
            except Exception:
                js = None
            if js:
                curves.append({"label": str(c.get("label") or "y"), "js": js})
        if not curves:
            return None
        readouts = []
        for r in (spec.get("readouts") or [])[:4]:
            try:
                js = _compile(r.get("expr", ""), names, env)
            except Exception:
                js = None
            if js:
                readouts.append({"label": str(r.get("label") or ""), "js": js})
        return {**base, "variables": variables, "x": xname,
                "x_label": str(spec.get("x_label") or xv["label"]),
                "y_label": str(spec.get("y_label") or ""),
                "log_y": bool(spec.get("log_y")), "constants": consts,
                "curves": curves, "readouts": readouts}

    # ------------------------------------------------------------ تجربة افتراضية
    reagents = []
    for r in (spec.get("reagents") or [])[:5]:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id", "")).strip()
        mx = _f(r.get("max"))
        if not NAME.fullmatch(rid) or rid in reserved or rid in taken or not mx or mx <= 0:
            continue
        step = _f(r.get("step"), mx / 10)
        if not step or step <= 0 or step > mx:
            step = mx / 10
        state = "solid" if str(r.get("state")).lower() == "solid" else "liquid"
        unit = str(r.get("unit") or "")
        mlu = _f(r.get("ml_per_unit"), 1.0 if (state == "liquid" and unit.lower() == "ml") else 0.0)
        color = str(r.get("color") or "")
        taken.add(rid)
        reagents.append({"id": rid, "name": str(r.get("name") or rid), "unit": unit,
                         "state": state, "color": color if HEX.fullmatch(color) else "#9ec5e8",
                         "step": step, "max": mx, "ml_per_unit": max(0.0, mlu or 0.0)})
    controls = _vars(spec.get("controls"), reserved, 3, taken)
    if not reagents and not controls:
        return None
    for r in reagents:
        env[r["id"]] = np.float64(r["max"] / 2)
    for c in controls:
        env[c["name"]] = np.float64(c["default"])
    env["t"] = np.float64(30)
    env["V"] = np.float64(sum(r["max"] / 2 * r["ml_per_unit"] for r in reagents) or 50)

    names = set(taken) | {"t", "V"}
    derived = []
    for d in (spec.get("derived") or [])[:6]:
        if not isinstance(d, dict):
            continue
        dn = str(d.get("name", "")).strip()
        if not NAME.fullmatch(dn) or dn in reserved or dn in names:
            continue
        js = _compile(d.get("expr", ""), names, env)
        if js:
            derived.append({"name": dn, "js": js})
            names.add(dn)
            env[dn] = np.float64(safe_eval(d["expr"], env))

    outputs = []
    for o in (spec.get("outputs") or [])[:4]:
        if not isinstance(o, dict):
            continue
        js = _compile(o.get("expr", ""), names, env)
        if not js:
            continue
        lo, hi = _f(o.get("min"), 0.0), _f(o.get("max"), 100.0)
        if not lo < hi:
            hi = lo + 1
        outputs.append({"id": str(o.get("id") or ""), "label": str(o.get("label") or ""),
                        "unit": str(o.get("unit") or ""), "min": lo, "max": hi, "js": js})
    if not outputs:
        return None

    visuals = {}
    vis_in = spec.get("visuals") or {}
    if isinstance(vis_in, dict):
        for k in ("bubbles", "foam", "precipitate", "heat", "color_mix"):
            if k in vis_in:
                js = _compile(vis_in[k], names, env)
                if js:
                    visuals[k] = js
        if HEX.fullmatch(str(vis_in.get("color_to", ""))):
            visuals["color_to"] = vis_in["color_to"]

    ids = [r["id"] for r in reagents]
    trig = [x for x in (spec.get("trigger") or []) if x in ids] or ids
    vessel = spec.get("vessel") if spec.get("vessel") in ("flask", "beaker", "test_tube", "cylinder") \
        else "beaker"
    return {**base, "vessel": vessel, "reagents": reagents, "trigger": trig,
            "controls": controls, "constants": consts, "derived": derived,
            "outputs": outputs, "visuals": visuals}


# ================================================================ صفحة المحاكي (JS شغال عند المتصفح = فوري)
LAB_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;background:#0e1117;color:#e8eaf0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,Tahoma,sans-serif;line-height:1.65}
body{padding:4px 2px 12px}
.card{background:#161a24;border:1px solid #283044;border-radius:12px;padding:12px;margin:10px 0}
h3{margin:2px 0 6px;font-size:1.1rem} h4{margin:0 0 8px;font-size:.95rem;color:#aab4c8;font-weight:600}
p,li{font-size:.95rem;margin:4px 0} ol,ul{margin:4px 0;padding-inline-start:22px}
canvas{width:100%;display:block;border-radius:10px;background:#0f131c}
button{background:#222a3b;color:#e8eaf0;border:1px solid #38425c;border-radius:10px;padding:10px 14px;font-size:1rem;cursor:pointer;touch-action:manipulation;min-width:44px}
button:active{background:#303a54} button.pri{background:#e0483f;border-color:#e0483f;color:#fff}
select{background:#222a3b;color:#e8eaf0;border:1px solid #38425c;border-radius:8px;padding:6px}
.row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.rg{display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #222839;flex-wrap:wrap}
.rg:last-child{border:0}
.dot{width:18px;height:18px;border-radius:50%;border:1px solid #fff6;flex:none}
.nm{flex:1;min-width:130px} .am{font-variant-numeric:tabular-nums;min-width:78px;text-align:center;color:#ffd166}
input[type=range]{width:100%;accent-color:#e0483f}
.bar{height:12px;background:#222839;border-radius:8px;overflow:hidden}
.bar>i{display:block;height:100%;width:0;background:linear-gradient(90deg,#2fb4a0,#ffd166)}
.val{font-variant-numeric:tabular-nums;color:#ffd166}
.f{font-family:ui-monospace,Menlo,Consolas,monospace;direction:ltr;text-align:left;background:#0f131c;border-radius:8px;padding:8px 10px;overflow-x:auto;margin-top:6px}
.mut{color:#97a1b6;font-size:.85rem}
.legend span{display:inline-flex;align-items:center;gap:5px;margin-inline-end:12px;font-size:.85rem}
.legend i{width:12px;height:4px;border-radius:2px;display:inline-block}
.met{flex:1;min-width:130px;background:#0f131c;border-radius:10px;padding:8px 10px}
.met b{display:block;font-size:1.3rem;color:#ffd166;font-weight:600}
</style></head><body><div id="app"></div>
<script>
(function(){
const S=__SPEC__;
const AR=!!S.rtl;
const T=AR?{start:"▶ كمّل",pause:"⏸ إيقاف",reset:"↺ إعادة",speed:"السرعة",time:"الزمن",s:"ث",steps:"الخطوات",obs:"🔎 لاحظ",ctl:"الظروف",reag:"المواد (اضغط + لإضافتها)",out:"القياسات",plot:"المنحنى المباشر",wait:"أضف المواد المطلوبة ليبدأ التفاعل",run:"التفاعل شغّال",paused:"متوقف مؤقتًا",norm:"كل منحنى مقيّس على مداه"}
 :{start:"▶ Resume",pause:"⏸ Pause",reset:"↺ Reset",speed:"Speed",time:"Time",s:"s",steps:"Steps",obs:"🔎 Notice",ctl:"Conditions",reag:"Materials (tap + to add)",out:"Measurements",plot:"Live curve",wait:"Add the required materials to start the reaction",run:"Reaction running",paused:"Paused",norm:"each curve is scaled to its own range"};
document.documentElement.dir=AR?"rtl":"ltr";
const app=document.getElementById("app");
const COLS=["#4cc9f0","#ffd166","#ef476f","#06d6a0"];
const resizers=[];
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
const fmt=v=>{if(!isFinite(v))return"—";const a=Math.abs(v);if(a!==0&&(a>=1e5||a<1e-3))return v.toExponential(2);return String(+v.toPrecision(4));};
const mk=f=>{try{return new Function("$","return ("+f+");")}catch(e){return function(){return NaN}}};
const num=(f,E)=>{try{const v=f(E);return isFinite(v)?v:NaN}catch(e){return NaN}};
const esc=s=>String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const $id=id=>document.getElementById(id);
function size(c,h){const d=window.devicePixelRatio||1,w=c.clientWidth||300;c.width=Math.round(w*d);c.height=Math.round(h*d);c.style.height=h+"px";c.getContext("2d").setTransform(d,0,0,d,0,0);return w}
function fit(){try{window.frameElement.style.height=(document.documentElement.scrollHeight+8)+"px"}catch(e){}}
function obs(){return S.observe&&S.observe.length?`<div class="card"><h4>${T.obs}</h4><ul>${S.observe.map(o=>`<li>${esc(o)}</li>`).join("")}</ul></div>`:""}
let head=`<div class="card"><h3>${esc(S.title)}</h3><p>${esc(S.description)}</p>`;
if(S.steps&&S.steps.length)head+=`<h4 style="margin-top:10px">${T.steps}</h4><ol>${S.steps.map(x=>`<li>${esc(x)}</li>`).join("")}</ol>`;
if(S.formula)head+=`<div class="f">${esc(S.formula)}</div>`;
head+=`</div>`;

function initLab(){
 const R=S.reagents||[],C=S.controls||[],O=S.outputs||[],D=S.derived||[],VS=S.visuals||{};
 let h=head;
 h+=`<div class="card"><canvas id="lab"></canvas><div class="row" style="margin-top:8px"><button class="pri" id="play"></button><button id="reset">${T.reset}</button><label class="mut">${T.speed} <select id="spd"><option value="1">1×</option><option value="5">5×</option><option value="20">20×</option><option value="60">60×</option></select></label><span class="val" id="clk"></span></div><div class="mut" id="msg" style="margin-top:6px"></div></div>`;
 if(R.length)h+=`<div class="card"><h4>${T.reag}</h4>`+R.map((r,i)=>`<div class="rg"><span class="dot" style="background:${r.color}"></span><span class="nm">${esc(r.name)}</span><button data-a="-" data-i="${i}">−</button><span class="am" id="am${i}"></span><button class="pri" data-a="+" data-i="${i}">+ ${r.step} ${esc(r.unit)}</button></div>`).join("")+`</div>`;
 if(C.length)h+=`<div class="card"><h4>${T.ctl}</h4>`+C.map((c,i)=>`<div style="margin:8px 0"><div class="row" style="justify-content:space-between"><span>${esc(c.label)}</span><span class="val" id="cv${i}"></span></div><input type="range" id="cr${i}" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.default}"></div>`).join("")+`</div>`;
 h+=`<div class="card"><h4>${T.out}</h4>`+O.map((o,i)=>`<div style="margin:8px 0"><div class="row" style="justify-content:space-between"><span>${esc(o.label)}</span><span class="val" id="ov${i}"></span></div><div class="bar"><i id="ob${i}"></i></div></div>`).join("")+`</div>`;
 h+=`<div class="card"><h4>${T.plot}</h4><canvas id="plot"></canvas><div class="legend" style="margin-top:6px">`+O.map((o,j)=>`<span><i style="background:${COLS[j%4]}"></i>${esc(o.label)}</span>`).join("")+`</div><div class="mut">${T.norm}</div></div>`;
 h+=obs();
 app.innerHTML=h;

 const lab=$id("lab"),lx=lab.getContext("2d"),pl=$id("plot"),px=pl.getContext("2d"),playB=$id("play");
 let LW=300;const LH=340;
 const amt={},ctl={};R.forEach(r=>amt[r.id]=0);C.forEach(c=>ctl[c.name]=c.default);
 const fO=O.map(o=>mk(o.js)),fD=D.map(d=>mk(d.js)),fV={};
 ["bubbles","foam","precipitate","heat","color_mix"].forEach(k=>{if(VS[k])fV[k]=mk(VS[k])});
 const trig=(S.trigger&&S.trigger.length)?S.trigger:R.map(r=>r.id);
 const go=()=>trig.every(id=>amt[id]>0);
 const volume=()=>R.reduce((s,r)=>s+amt[r.id]*(r.ml_per_unit||0),0);
 const cap=Math.max(20,R.reduce((s,r)=>s+r.max*(r.ml_per_unit||0),0))*1.15;
 let t=0,running=true,last=performance.now(),spd=1,hist=[],lastS=-1,bub=[],acc=0;
 const rnd=i=>{const x=Math.sin(i*127.1+311.7)*43758.5453;return x-Math.floor(x)};
 const hex=c=>{const n=parseInt(c.slice(1),16);return[n>>16&255,n>>8&255,n&255]};

 function env(){const E=Object.assign({},S.constants||{});R.forEach(r=>E[r.id]=amt[r.id]);C.forEach(c=>E[c.name]=ctl[c.name]);E.t=t;E.V=volume();D.forEach((d,i)=>{const v=num(fD[i],E);E[d.name]=isNaN(v)?0:v});return E}
 function geom(){const cx=LW/2,v=S.vessel;
  if(v==="flask")return{cx,top:26,bot:306,w:Math.min(210,LW-40),neck:56,neckY:120,type:v};
  if(v==="test_tube")return{cx,top:20,bot:308,w:64,type:v};
  if(v==="cylinder")return{cx,top:16,bot:312,w:84,type:v};
  return{cx,top:60,bot:308,w:Math.min(180,LW-40),type:"beaker"}}
 function path(g){lx.beginPath();const h=g.w/2;
  if(g.type==="flask"){const n=g.neck/2;lx.moveTo(g.cx-n,g.top);lx.lineTo(g.cx-n,g.neckY);lx.lineTo(g.cx-h,g.bot-14);lx.quadraticCurveTo(g.cx-h,g.bot,g.cx-h+14,g.bot);lx.lineTo(g.cx+h-14,g.bot);lx.quadraticCurveTo(g.cx+h,g.bot,g.cx+h,g.bot-14);lx.lineTo(g.cx+n,g.neckY);lx.lineTo(g.cx+n,g.top)}
  else if(g.type==="test_tube"||g.type==="cylinder"){const r=g.type==="test_tube"?h:10;lx.moveTo(g.cx-h,g.top);lx.lineTo(g.cx-h,g.bot-r);lx.quadraticCurveTo(g.cx-h,g.bot,g.cx-h+r,g.bot);lx.lineTo(g.cx+h-r,g.bot);lx.quadraticCurveTo(g.cx+h,g.bot,g.cx+h,g.bot-r);lx.lineTo(g.cx+h,g.top)}
  else{lx.moveTo(g.cx-h,g.top);lx.lineTo(g.cx-h+4,g.bot-8);lx.quadraticCurveTo(g.cx-h+4,g.bot,g.cx-h+12,g.bot);lx.lineTo(g.cx+h-12,g.bot);lx.quadraticCurveTo(g.cx+h-4,g.bot,g.cx+h-4,g.bot-8);lx.lineTo(g.cx+h,g.top)}}
 function liqColor(vis){let tv=0,c=[0,0,0];R.forEach(r=>{const m=amt[r.id]*(r.ml_per_unit||0);if(m>0){const k=hex(r.color);c=c.map((x,i)=>x+k[i]*m);tv+=m}});
  if(!tv)return null;c=c.map(x=>x/tv);
  if(VS.color_to&&vis.color_mix){const k=hex(VS.color_to);c=c.map((x,i)=>x+(k[i]-x)*vis.color_mix)}
  return"rgba("+c.map(Math.round).join(",")+",0.8)"}

 function drawLab(vis,dt){
  const g=geom(),vol=volume(),frac=clamp(vol/cap,0,1);
  const yFull=g.type==="flask"?g.top+(g.neckY-g.top)*0.6:g.top+(g.bot-g.top)*0.1;
  const yL=g.bot-frac*(g.bot-yFull);
  lx.clearRect(0,0,LW,LH);
  lx.fillStyle="#1d2434";lx.fillRect(g.cx-g.w/2-18,g.bot+4,g.w+36,6);
  const col=liqColor(vis),h2=g.w/2;
  lx.save();path(g);lx.clip();
  if(col&&frac>0){lx.fillStyle=col;lx.fillRect(0,yL,LW,LH-yL)}
  if(vis.precipitate){lx.fillStyle="rgba(240,240,240,"+(0.25+0.6*vis.precipitate)+")";lx.fillRect(0,g.bot-8-vis.precipitate*46,LW,60)}
  R.forEach((r,ri)=>{if(r.state!=="solid"||!(amt[r.id]>0))return;const n=Math.max(3,Math.round(amt[r.id]/r.max*70));lx.fillStyle=r.color;
   for(let j=0;j<n;j++){const x=g.cx-h2*0.8+rnd(j*7+ri*131)*h2*1.6,y=g.bot-5-rnd(j*13+ri*57)*20,rr=2.5+rnd(j*3+ri)*2.5;lx.beginPath();lx.arc(x,y,rr,0,6.283);lx.fill()}});
  if(vis.bubbles>0.01&&frac>0.02){acc+=vis.bubbles*dt*45;while(acc>=1&&bub.length<130){bub.push({x:g.cx+(Math.random()-.5)*g.w*0.7,y:g.bot-10,r:2+Math.random()*4,v:30+Math.random()*50});acc--}if(acc>3)acc=0}
  bub.forEach(b=>{b.y-=b.v*dt*(0.6+vis.bubbles);b.x+=Math.sin(b.y*0.1)*0.3});
  bub=bub.filter(b=>b.y>yL-2);
  lx.strokeStyle="rgba(235,245,255,.75)";lx.lineWidth=1.2;bub.forEach(b=>{lx.beginPath();lx.arc(b.x,b.y,b.r,0,6.283);lx.stroke()});
  if(vis.foam>0.02&&frac>0.02){const hF=vis.foam*(g.bot-g.top)*0.55,yt=Math.max(yL-hF,g.top);
   lx.fillStyle="rgba(235,240,250,.55)";lx.fillRect(0,yt,LW,yL-yt);
   const cnt=Math.min(170,Math.round((yL-yt)*g.w/240));
   for(let i=0;i<cnt;i++){const x=g.cx-h2+rnd(i*3.1)*g.w,y=yt+rnd(i*5.7)*(yL-yt),r=4+rnd(i*9.3)*7;lx.fillStyle="rgba(244,248,255,.85)";lx.beginPath();lx.arc(x,y,r,0,6.283);lx.fill();lx.strokeStyle="rgba(150,170,200,.5)";lx.lineWidth=.8;lx.stroke()}}
  lx.restore();
  if(vis.foam>0.97){const ow=(g.type==="flask"?g.neck:g.w)/2;for(let i=0;i<14;i++){const x=g.cx-ow+rnd(i*4.3)*ow*2,y=g.top-rnd(i*2.9)*30,r=5+rnd(i*8.1)*7;lx.fillStyle="rgba(244,248,255,.9)";lx.beginPath();lx.arc(x,y,r,0,6.283);lx.fill()}}
  if(vis.heat>0.05){lx.shadowColor="rgba(255,120,40,"+vis.heat*0.9+")";lx.shadowBlur=vis.heat*26}
  lx.lineWidth=3;lx.lineJoin="round";lx.strokeStyle="rgba(215,232,255,.7)";path(g);lx.stroke();
  const ow2=(g.type==="flask"?g.neck:g.w)/2;lx.beginPath();lx.moveTo(g.cx-ow2-6,g.top);lx.lineTo(g.cx-ow2,g.top);lx.moveTo(g.cx+ow2,g.top);lx.lineTo(g.cx+ow2+6,g.top);lx.stroke();
  lx.shadowBlur=0;
  if(vis.heat>0.05){lx.lineWidth=2;const now=performance.now()/300;for(let k=0;k<3;k++){lx.strokeStyle="rgba(255,255,255,"+0.3*vis.heat+")";lx.beginPath();for(let y=g.top-6;y>g.top-70;y-=4){const x=g.cx+(k-1)*16+Math.sin(y*0.15+now+k)*6;if(y===g.top-6)lx.moveTo(x,y);else lx.lineTo(x,y)}lx.stroke()}}
  lx.fillStyle="#97a1b6";lx.font="13px sans-serif";lx.textAlign="left";lx.fillText(Math.round(vol)+" mL",10,22)}

 function drawPlot(){const w=pl.clientWidth||300,h=220,m={l:12,r:12,t:10,b:28};
  px.clearRect(0,0,w,h);px.strokeStyle="#283044";px.fillStyle="#97a1b6";px.font="11px sans-serif";px.lineWidth=1;px.textAlign="center";
  const tmax=Math.max(60,hist.length?hist[hist.length-1][0]*1.05:60);
  for(let i=0;i<=4;i++){const y=m.t+(h-m.t-m.b)*i/4;px.beginPath();px.moveTo(m.l,y);px.lineTo(w-m.r,y);px.stroke()}
  for(let i=0;i<=4;i++){const x=m.l+(w-m.l-m.r)*i/4;px.fillText(fmt(tmax*i/4)+T.s,x,h-10)}
  O.forEach((o,j)=>{px.strokeStyle=COLS[j%4];px.lineWidth=2;px.beginPath();
   hist.forEach((p,k)=>{const x=m.l+(w-m.l-m.r)*p[0]/tmax,y=h-m.b-(h-m.t-m.b)*clamp((p[j+1]-o.min)/(o.max-o.min||1),0,1);if(k)px.lineTo(x,y);else px.moveTo(x,y)});px.stroke()})}

 function frame(now){
  const dt=Math.min(0.1,Math.max(0,(now-last)/1000));last=now;
  const ready=go(),on=running&&ready;
  if(on)t+=dt*spd;
  const E=env();
  O.forEach((o,i)=>{const v=num(fO[i],E),x=isNaN(v)?0:v;o.v=x;$id("ov"+i).textContent=fmt(x)+" "+o.unit;$id("ob"+i).style.width=clamp((x-o.min)/(o.max-o.min||1),0,1)*100+"%"});
  const vis={};for(const k in fV){const v=num(fV[k],E);vis[k]=isNaN(v)?0:clamp(v,0,1)}
  if(on&&t-lastS>=0.5){hist.push([t].concat(O.map(o=>o.v)));lastS=t;if(hist.length>4000)hist.shift()}
  R.forEach((r,i)=>{$id("am"+i).textContent=fmt(amt[r.id])+" "+r.unit});
  C.forEach((c,i)=>{$id("cv"+i).textContent=fmt(ctl[c.name])+" "+c.unit});
  $id("clk").textContent=T.time+": "+t.toFixed(0)+" "+T.s;
  $id("msg").textContent=!ready?T.wait:(running?T.run:T.paused);
  drawLab(vis,dt);drawPlot();
  requestAnimationFrame(frame)}

 function upd(){playB.textContent=running?T.pause:T.start}
 app.addEventListener("click",e=>{const b=e.target.closest("button");if(!b)return;
  if(b.id==="play"){running=!running;upd();return}
  if(b.id==="reset"){R.forEach(r=>amt[r.id]=0);t=0;hist=[];lastS=-1;bub=[];running=true;upd();return}
  if(b.dataset.a){const r=R[+b.dataset.i];amt[r.id]=clamp(Math.round((amt[r.id]+(b.dataset.a==="+"?r.step:-r.step))*1e6)/1e6,0,r.max)}});
 $id("spd").onchange=e=>{spd=+e.target.value};
 C.forEach((c,i)=>{$id("cr"+i).oninput=e=>{ctl[c.name]=+e.target.value}});
 resizers.push(()=>{LW=size(lab,LH);size(pl,220)});
 resizers.forEach(f=>f());upd();requestAnimationFrame(frame)}

function initGraph(){
 const Vs=S.variables,Cs=S.curves,Rs=S.readouts||[],K=S.constants||{};
 const fC=Cs.map(c=>mk(c.js)),fR=Rs.map(r=>mk(r.js));
 const vals={};Vs.forEach(v=>vals[v.name]=v.default);
 const xv=Vs.find(v=>v.name===S.x)||Vs[0];
 let h=head;
 h+=`<div class="card"><canvas id="plot"></canvas><div class="legend" style="margin-top:6px">`+Cs.map((c,j)=>`<span><i style="background:${COLS[j%4]}"></i>${esc(c.label)}</span>`).join("")+`</div></div>`;
 h+=`<div class="card">`+Vs.map((v,i)=>`<div style="margin:8px 0"><div class="row" style="justify-content:space-between"><span>${esc(v.label)}</span><span class="val" id="gv${i}"></span></div><input type="range" id="gr${i}" min="${v.min}" max="${v.max}" step="${v.step}" value="${v.default}"></div>`).join("")+`</div>`;
 if(Rs.length)h+=`<div class="row">`+Rs.map((r,i)=>`<div class="met"><span class="mut">${esc(r.label)}</span><b id="rv${i}"></b></div>`).join("")+`</div>`;
 h+=obs();
 app.innerHTML=h;
 const pl=$id("plot"),px=pl.getContext("2d");
 const E=()=>Object.assign({},K,vals);
 function draw(){
  const w=size(pl,300),hh=300,m={l:56,r:14,t:14,b:44},lg=!!S.log_y;
  px.clearRect(0,0,w,hh);px.font="11px sans-serif";px.lineWidth=1;
  const N=200,xs=[];for(let i=0;i<=N;i++)xs.push(xv.min+(xv.max-xv.min)*i/N);
  const series=Cs.map((c,j)=>xs.map(x=>{const e=E();e[xv.name]=x;const y=num(fC[j],e);return(isFinite(y)&&(!lg||y>0))?y:NaN}));
  const ys=[].concat.apply([],series).filter(isFinite);
  Vs.forEach((v,i)=>{$id("gv"+i).textContent=fmt(vals[v.name])+" "+v.unit});
  Rs.forEach((r,i)=>{$id("rv"+i).textContent=fmt(num(fR[i],E()))});
  if(!ys.length){px.fillStyle="#97a1b6";px.fillText("—",w/2,hh/2);return}
  let lo=Math.min.apply(null,ys),hi=Math.max.apply(null,ys);
  if(lg){lo=Math.log10(lo);hi=Math.log10(hi)}
  if(lo===hi){lo-=1;hi+=1}
  const pad=(hi-lo)*0.06;lo-=pad;hi+=pad;
  const X=x=>m.l+(w-m.l-m.r)*(x-xv.min)/(xv.max-xv.min);
  const Y=y=>{const v=lg?Math.log10(y):y;return hh-m.b-(hh-m.t-m.b)*(v-lo)/(hi-lo)};
  px.strokeStyle="#283044";px.fillStyle="#97a1b6";
  for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,y=hh-m.b-(hh-m.t-m.b)*i/4;px.beginPath();px.moveTo(m.l,y);px.lineTo(w-m.r,y);px.stroke();px.textAlign="right";px.fillText(fmt(lg?Math.pow(10,v):v),m.l-6,y+4)}
  for(let i=0;i<=4;i++){const x=m.l+(w-m.l-m.r)*i/4;px.textAlign="center";px.fillText(fmt(xv.min+(xv.max-xv.min)*i/4),x,hh-m.b+16)}
  px.textAlign="center";px.fillText(S.x_label||"",(m.l+w-m.r)/2,hh-6);
  px.textAlign="left";px.fillText(S.y_label||"",m.l,10);
  const cx=X(vals[xv.name]);px.setLineDash([4,4]);px.strokeStyle="#4a5470";px.beginPath();px.moveTo(cx,m.t);px.lineTo(cx,hh-m.b);px.stroke();px.setLineDash([]);
  series.forEach((s,j)=>{px.strokeStyle=COLS[j%4];px.lineWidth=2;px.beginPath();let pen=false;
   s.forEach((y,i)=>{if(!isFinite(y)){pen=false;return}const x=X(xs[i]),yy=Y(y);if(pen)px.lineTo(x,yy);else{px.moveTo(x,yy);pen=true}});px.stroke();
   const yp=num(fC[j],E());if(isFinite(yp)&&(!lg||yp>0)){px.fillStyle=COLS[j%4];px.beginPath();px.arc(cx,Y(yp),5,0,6.283);px.fill()}})}
 Vs.forEach((v,i)=>{$id("gr"+i).oninput=e=>{vals[v.name]=+e.target.value;draw()}});
 resizers.push(draw);draw()}

if(S.type==="graph")initGraph();else initLab();
window.addEventListener("resize",()=>{resizers.forEach(f=>f());fit()});
setInterval(fit,600);fit();
})();
</script></body></html>"""


def build_page(sim):
    data = json.dumps(sim, ensure_ascii=False).replace("</", "<\\/")
    return LAB_HTML.replace("__SPEC__", data)


# ================================================================ التاب
def sim_tab(provider, model_name, key):
    st.write("اكتب اسم التجربة أو الموضوع، أو ارفع المحاضرة/البوستر. "
             "هتطلعلك تجربة افتراضية تضيف فيها المواد وتغيّر الظروف وتشوف النتيجة لحظة بلحظة "
             "(والرسم بالمعادلات بيتعمل بس لو الموضوع معادلة فعلاً).")
    topic = st.text_input("التجربة أو الموضوع (اختياري لو رفعت ملف)",
                          placeholder="مثال: Yeast and hydrogen peroxide، معايرة حمض وقاعدة، معادلة أرينيوس",
                          key="sim_topic")
    files = st.file_uploader("المحاضرة (PDF أو صور)", type=SRC_TYPES,
                             accept_multiple_files=True, key="sim_files")
    c1, c2 = st.columns(2)
    n = c1.slider("عدد المحاكيات", 1, 3, 1, key="sim_n")
    lang_opt = c2.selectbox("لغة الشرح", ["نفس لغة المصدر", "العربية", "English"], key="sim_lang")
    kind = st.radio("نوع المحاكي", ["تلقائي", "تجربة معملية", "رسم معادلة"],
                    horizontal=True, key="sim_kind")

    if (files or topic.strip()) and st.button("ابنِ المحاكي 🎛️"):
        with st.spinner("بجهّز المحاكي..."):
            try:
                text, imgs = gather(files or [])
                items = []
                if text:
                    items.append(("text", "LECTURE TEXT:\n" + text))
                for i, p in enumerate(imgs):
                    items.append(("text", f"LECTURE IMAGE {i + 1}:"))
                    items.append(("image", p))
                if topic.strip():
                    items.append(("text", "STUDENT REQUEST: " + topic.strip()))
                if not items:
                    st.error("معرفتش أقرأ أي محتوى.")
                    return
                if lang_opt == "العربية":
                    lang = "Arabic"
                elif lang_opt == "English":
                    lang = "English"
                else:
                    lang = "the same language as the material or the student's request"
                focus = (f"Focus on: {topic.strip()}." if topic.strip()
                         else "Pick the most important experiment or relationship in the material.")
                kind_txt = {"تلقائي": "",
                            "تجربة معملية": 'Every simulation MUST be type "experiment".',
                            "رسم معادلة": 'Every simulation MUST be type "graph".'}[kind]
                prompt = (SIM_PROMPT.replace("@N@", str(n)).replace("@FOCUS@", focus)
                          .replace("@KIND@", kind_txt).replace("@LANG@", lang))
                raw = call_model(provider, model_name, items, system=prompt, key=key,
                                 json_mode=True)
                data = parse_json_block(raw)
                sims = [s for s in (clean_sim(x) for x in data.get("simulations", [])) if s]
            except Exception as e:
                st.error(f"معرفتش أجهّز المحاكي: {str(e)[:250]}")
                return
        if not sims:
            st.error("معرفتش أطلّع محاكي من المحتوى ده. جرّب تكتب اسم التجربة بنفسك أو تغيّر النموذج.")
            return
        st.session_state.sims_v2 = sims
        st.session_state.sims_model = LAST_USED["model"]

    sims = st.session_state.get("sims_v2")
    if sims:
        used = st.session_state.get("sims_model")
        if used and used != model_name:
            st.info(f"النموذج المختار كان مشغول فاستخدمت {used}.")
        st.divider()
        st.warning("قارن القيم والمعادلات بالمحاضرة، لأن النموذج ممكن يغلط في الأرقام. "
                   "المحاكي تقريب تعليمي مش قياس حقيقي.")
        for i, sim in enumerate(sims):
            with st.expander(f"🎛️ {sim['title']}", expanded=(i == 0)):
                components.html(build_page(sim), height=1500, scrolling=True)
        if st.button("🗑️ امسح المحاكيات"):
            st.session_state.pop("sims_v2", None)
            st.rerun()
