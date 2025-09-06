# app.py – HU UI + képből-prompt + automatikus EN fordítás + FLUX generálás
import base64
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # majd jelezni fogunk, ha hiányzik a csomag

import os, io, base64, tempfile, requests, streamlit as st
from PIL import Image, ImageOps
import replicate
from datetime import datetime

# ------------------ ALAP ------------------
st.set_page_config(page_title="Kiskakas képgenerátor", layout="centered")

# --- Piros gombok (st.button + st.download_button) ---
st.markdown("""
<style>
.stButton > button, .stDownloadButton > button {
  background-color:#e11d48; /* rose-600 */
  color:white; border:0; border-radius:10px;
  padding:0.6rem 1.1rem; font-weight:600;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background-color:#be123c; /* rose-700 */
}
</style>
""", unsafe_allow_html=True)

def render_help_sidebar():
    with st.sidebar:
        st.title("❓ Súgó / Útmutató")

        with st.expander("Mi ez az alkalmazás?", expanded=False):
            st.markdown("""
**Kiskakas Képgenerátor** – új, fotórealisztikus képet készít egy feltöltött **arckép** alapján.
Az arc megmarad (ugyanaz a személy), a jelenetet a **prompt** (leírás) határozza meg.
A generálás a Replicate szolgáltatáson futó **FLUX Kontext Pro** modellel történik (arc-hű img2img).
            """)

        with st.expander("Mit tud?", expanded=False):
            st.markdown("""
- **Arckép feltöltése** – erre épül az azonosság megőrzése.  
- **Képből prompt** (opcionális második kép): automatikus **angol leírás**, szerkeszthető.  
- **Prompt mező** – bármikor átírhatod/kiegészítheted.  
- **Angolra fordítás** – ha bepipálod, magyarból automatikusan angolra vált a modellnek.  
- **Arc-hasonlóság csúszka** – mennyire ragaszkodjon az eredeti archoz.  
- **Haladó beállítások** – lépések száma, guidance, negatív prompt.  
- **Eredmény** – megjelenítés és letöltés.
            """)

        with st.expander("Lépésről lépésre", expanded=False):
            st.markdown("""
1. **Tölts fel egy arcképet.** Éles, jól megvilágított fotó a legjobb.  
2. *(Opcionális)* **Kép a prompthoz** → katt **„Képből promptot kérek”** (angol leírást kapunk).  
3. **Szerkeszd a promptot.** Írd le a helyszínt, ruhát, fényt, nézőszöget stb.  
   - Ha feliratot szeretnél táblára: kérj **BLANK/ÜRES** kartont; a feliratot később kóddal rá lehet írni.  
4. *(Opcionális)* **„Küldés előtt fordítsd angolra”** – jelöld be, ha magyarul írtál.  
5. **Arc-hasonlóság**: 0.5–0.7 jó kezdés; 0.8–0.9 nagyon hű arc, de kevesebb változás.  
6. *(Haladó)* **Lépések száma** (kevesebb = gyorsabb), **Guidance** (magasabb = jobban követi a promptot), **Negatív prompt**.  
7. **Katt a „Kép generálása” gombra.** Első futás lassabb lehet (1–3 perc). A kész képet letöltheted.
            """)

        with st.expander("Tippek a szép eredményhez", expanded=False):
            st.markdown("""
- **Jó arckép**: szemből, éles, jó fényben, az arc ne legyen takarva.  
- **Pontosság**: helyszín, napszak/fény (pl. *golden hour*), nézőszög (pl. *full-body*, *medium shot*), ruházat.  
- **Üres tábla**: feliratot később programból adj hozzá (a modellek gyakran félreírják).
- **Lassú?** Csökkentsd a lépések számát (pl. 18–22).
            """)

        with st.expander("Hibaelhárítás", expanded=False):
            st.markdown("""
- **„Nem sikerült promptot készíteni a képből…”**  
  – Nincs OpenAI kulcs, vagy a választott Replicate képleíró nem érhető el. Írj promptot kézzel, vagy adj **OPENAI_API_KEY**-t.  
- **Replicate 401/404/422**  
  – Ellenőrizd a **REPLICATE_API_TOKEN**-t és a modellhez való hozzáférést.  
- **Nagyon lassú generálás**  
  – Első futás hideg indítás. Kevesebb lépés gyorsít.
            """)


# --- Lábléc / Copyright ---
from datetime import datetime

def render_footer(owner: str = "Viktor Huszár", year: int | None = None, sticky: bool = False):
    y = year or datetime.now().year
    if not sticky:
        st.markdown(
            f"""
            <hr style="margin-top:3rem;border:none;border-top:1px solid rgba(255,255,255,0.15);" />
            <div style="text-align:center;opacity:.6;font-size:0.9rem;">
                © {y} {owner} · Minden jog fenntartva
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <style>
              .kk-footer {{
                position: fixed; left: 0; right: 0; bottom: 0;
                text-align: center; opacity:.65; font-size:.9rem;
                padding:.35rem 0; pointer-events:none;
              }}
              .kk-footer span {{ pointer-events:auto; padding:.1rem .5rem; }}
            </style>
            <div class="kk-footer"><span>© {y} {owner} · Minden jog fenntartva</span></div>
            """,
            unsafe_allow_html=True
        )


REPLICATE_API_TOKEN = st.secrets.get("REPLICATE_API_TOKEN", os.getenv("REPLICATE_API_TOKEN", "")).strip()
if not REPLICATE_API_TOKEN:
    st.error("Hiányzik a REPLICATE_API_TOKEN. Add meg a .streamlit/secrets.toml fájlban vagy környezeti változóként.")
    st.stop()
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")).strip()

MODEL_FLUX = "black-forest-labs/flux-kontext-pro"
PROMPT_MODELS = [
    "pharmapsychotic/clip-interrogator",
    "tstramer/clip-interrogator",
    "methexis-inc/img2prompt",
]

# ------------------ SEGÉDFÜGGVÉNYEK ------------------
def save_temp_png(upload) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        Image.open(upload).convert("RGB").save(tmp.name, "PNG")
        return tmp.name

def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=180); r.raise_for_status(); return r.content

# a fájl tetején ezek kellenek:
# import base64
# from openai import OpenAI

def try_openai_img2prompt(img_path: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Nincs OPENAI_API_KEY")
    if OpenAI is None:
        raise RuntimeError("Hiányzik az openai csomag. Telepítsd: pip install --upgrade openai")

    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [{
        "role": "user",
        "content": [
            {"type": "text",
             "text": ("Describe this image as a concise English prompt for a photorealistic diffusion model (SDXL/FLUX). "
                      "Include subject, clothing, setting, lighting, composition. One sentence.")},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}}
        ],
    }]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=200,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()



# model-lista és bemenet-építők
PROMPT_MODELS = [
    ("pharmapsychotic/clip-interrogator", lambda p: {"image": open(p, "rb")}),
    ("tstramer/clip-interrogator",        lambda p: {"image": open(p, "rb")}),
    ("cjwbw/blip-image-captioning",       lambda p: {"image": open(p, "rb")}),
    ("yorickvp/llava-13b",                lambda p: {
        "image": open(p, "rb"),
        "prompt": ("Describe this image as a concise English prompt for a photorealistic diffusion model. "
                   "Include subject, clothing, setting, lighting, composition. One sentence.")
    }),
]

def try_replicate_img2prompt(img_path: str) -> str:
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    last_err = None
    for ref, build in PROMPT_MODELS:
        try:
            out = client.run(ref, input=build(img_path))
            if isinstance(out, dict):
                for k in ("prompt", "caption", "best_caption", "best_prompt", "description", "text"):
                    if k in out:
                        return str(out[k]).strip()
                return str(out).strip()
            if isinstance(out, (list, tuple)) and out:
                return str(out[0]).strip()
            return str(out).strip()
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Replicate img2prompt hiba: {last_err}")


def translate_to_english(text: str) -> str:
    """Gyors HU→EN fordítás (OpenAI, ha van; különben visszaadjuk az eredetit)."""
    if not text.strip():
        return text
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            r = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":"Translate to concise, natural English."},
                          {"role":"user","content":text}]
            )
            return r.choices[0].message.content.strip()
        except Exception:
            return text
    else:
        # nincs OpenAI → nem fordítunk (a modellek az angolt szeretik, javasolt a kulcs megadása)
        return text

# ------------------ UI ------------------
st.title("🎨 Kiskakas Képgenerátor")
st.caption("1) Arckép • 2) (opcionális) Képből prompt • 3) Prompt szerkesztése • 4) Generálás")
render_help_sidebar()
render_footer(sticky=True)

# 1) Arckép
face_file = st.file_uploader("📸 1) Arckép feltöltése (JPG/PNG)", type=["jpg","jpeg","png"])
if face_file:
    img = Image.open(face_file).convert("RGB")
    img = ImageOps.exif_transpose(img)  # helyes tájolás
    st.image(img, caption="Feltöltött arckép (kicsinyített)", width=260) 
    with st.expander("Nagyobb nézet"):
        st.image(img, use_container_width=True)

# 2) Kép a prompthoz
prompt_img = st.file_uploader("🖼️ 2) Kép a prompthoz (opcionális)", type=["jpg","jpeg","png"])
col1, col2 = st.columns([1,2])
with col1:
    want_img2prompt = st.button("✨ Képből promptot kérek")
with col2:
    st.caption("A képből automatikus angol prompt készül; lent szabadon szerkesztheted.")

if want_img2prompt and prompt_img is not None:
    ppath = save_temp_png(prompt_img)
    try:
        with st.spinner("OpenAI képleírás…"):
            auto_prompt = try_openai_img2prompt(ppath)
    except Exception as e1:
        try:
            with st.spinner("Replicate képleírás (tartalék)…"):
                auto_prompt = try_replicate_img2prompt(ppath)
        except Exception as e2:
            st.error(f"Nem sikerült promptot készíteni a képből.\nOpenAI: {e1}\nReplicate: {e2}")
            auto_prompt = ""
    if auto_prompt:
        st.session_state["prompt_text"] = auto_prompt
        st.success("✅ Prompt elkészült. Lent szerkeszthető.")

# 3) Prompt mező (szerkeszthető)
default_prompt = ("This man is hitchhiking to Kötcse on a rural Hungarian road, holding a BLANK cardboard sign, "
                  "photorealistic, natural daylight, realistic skin tones, sharp details")
prompt_text = st.session_state.get("prompt_text", default_prompt)
prompt_box = st.text_area("📝 3) Jelenet leírása (szerkeszthető)", value=prompt_text, key="prompt_text", height=140)

translate_checkbox = st.checkbox("Küldés előtt fordítsd angolra (ajánlott)", value=False)

# 4) Arc-hasonlóság + haladó
face_strength = st.slider("Arc-hasonlóság", 0.30, 0.90, 0.60, 0.05,
                          help="Magasabb értéknél erősebben őrzi az arcot, de kevesebb kreatív szabadság.")
with st.expander("Haladó beállítások (opcionális)", expanded=False):
    steps = st.slider("Lépések száma", 10, 60, 24, 1, help="Kevesebb = gyorsabb, több = részletesebb (lassabb).")
    guidance = st.slider("Irányítás (guidance scale)", 2.0, 12.0, 7.5, 0.5)
    negative = st.text_input("Negatív prompt", "blurry, deformed, low quality, watermark, extra fingers, bad anatomy, text")

# 5) Generálás
go = st.button("🚀 4) Kép generálása")

if go:
    if not face_file:
        st.warning("Először tölts fel egy arcképet.")
        st.stop()

    face_path = save_temp_png(face_file)

    # végső prompt (opcionális HU→EN)
    orig_prompt = st.session_state.get("prompt_text", default_prompt).strip()
    final_prompt = translate_to_english(orig_prompt) if translate_checkbox else orig_prompt

    with st.expander("Angol prompt (amit a modell kap)", expanded=False):
        st.write(final_prompt)

    st.info("Generálás elkezdődött..")
    with st.spinner("Generálás a Replicate-en…"):
        try:
            client = replicate.Client(api_token=REPLICATE_API_TOKEN)
            inputs = {
                "input_image": open(face_path, "rb"),
                "prompt": final_prompt,
                "negative_prompt": (negative or None),
                "prompt_strength": float(face_strength),
                "guidance_scale": float(guidance),
                "num_inference_steps": int(steps),
                "aspect_ratio": "match_input_image",
                "output_format": "jpg",
                "safety_tolerance": 2,
                "prompt_upsampling": True,
                "seed": -1,
            }
            output = client.run(MODEL_FLUX, input=inputs)
            url = str(output[0]) if isinstance(output, (list, tuple)) and output else str(output)
            st.image(url, caption="🎉 Elkészült kép", use_container_width=True)
            st.markdown(f"[Kép megnyitása új lapon]({url})")
            try:
                data = download_bytes(url)
                st.download_button("📥 Kép letöltése", data, file_name="kiskakas_generated.png", mime="image/png")
            except Exception as e:
                st.warning(f"Letöltés sikertelen: {e}")
        except replicate.exceptions.ReplicateError as e:
            st.error(f"Replicate hiba: {e}")
        except Exception as e:
            st.error(f"Váratlan hiba: {e}")
