from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import httpx
import json
import os
import random
import re
import time as _time
import secrets
import hmac
import hashlib

app = FastAPI()

ALLOWED_ORIGINS = [
    "https://design-planner.com",
    "https://www.design-planner.com",
    "https://pervyyii.ru",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BROADCAST_URL = os.environ.get("BROADCAST_URL", "")
BROADCAST_SECRET = os.environ.get("BROADCAST_SECRET", "")
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
TZ_ACCESS_KEY = os.environ.get("TZ_ACCESS_KEY", "")

# ─────────────────── Лимиты и защита от абуза ───────────────────
_rate_buckets: dict[str, list[float]] = {}
MAX_MESSAGES = 40
MAX_MSG_CHARS = 8000
MAX_TOTAL_CHARS = 24000


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _rate_limited(key: str, limit: int = 10, window: int = 3600) -> bool:
    now = _time.time()
    bucket = [t for t in _rate_buckets.get(key, []) if now - t < window]
    if len(bucket) >= limit:
        _rate_buckets[key] = bucket
        return True
    bucket.append(now)
    _rate_buckets[key] = bucket
    if len(_rate_buckets) > 10000:
        for k in [k for k, v in _rate_buckets.items() if not [t for t in v if now - t < window]]:
            _rate_buckets.pop(k, None)
    return False


def _validate_chat_messages(messages) -> str | None:
    if not isinstance(messages, list) or not messages:
        return "messages is empty"
    if len(messages) > MAX_MESSAGES:
        return "too many messages"
    total = 0
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            return "invalid message role"
        content = m.get("content")
        if not isinstance(content, str):
            return "message content must be a string"
        if len(content) > MAX_MSG_CHARS:
            return "message too long"
        total += len(content)
    if total > MAX_TOTAL_CHARS:
        return "conversation too long"
    return None


# ─────────────────── Обезличивание ПД перед отправкой за рубеж (152-ФЗ) ───────────────────
# За границу (Anthropic) уходит только текст с плейсхолдерами вместо ПД.
# Реальные значения остаются на сервере, ответ обратно un-mask'ается для пользователя.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_TG_RE = re.compile(r"@[A-Za-z][A-Za-z0-9_]{3,}")
_PHONE_RE = re.compile(r"\+?[78]?[\s\-()]*\d(?:[\s\-()]*\d){9,10}")
_NAME_RE = re.compile(r"(меня зовут|мо[её] имя|зовут меня)\s+([А-ЯЁ][а-яё]+)", re.I)


def _mask_pii(text: str, mapping: dict) -> str:
    if not isinstance(text, str):
        return text

    def _ph(kind: str, val: str) -> str:
        for ph, v in mapping.items():
            if v == val:
                return ph
        ph = f"[{kind}_{len(mapping) + 1}]"
        mapping[ph] = val
        return ph

    text = _EMAIL_RE.sub(lambda m: _ph("EMAIL", m.group(0)), text)
    text = _TG_RE.sub(lambda m: _ph("TG", m.group(0)), text)
    text = _PHONE_RE.sub(lambda m: _ph("PHONE", m.group(0)), text)
    text = _NAME_RE.sub(lambda m: m.group(1) + " " + _ph("NAME", m.group(2)), text)
    return text


def _unmask(text: str, mapping: dict) -> str:
    for ph, val in mapping.items():
        text = text.replace(ph, val)
    return text


def _mask_messages(messages: list) -> tuple[list, dict]:
    """(обезличенные messages, mapping) — для зарубежного LLM."""
    mapping: dict = {}
    masked = [
        {"role": m["role"], "content": _mask_pii(str(m.get("content", "")), mapping)}
        for m in messages
    ]
    return masked, mapping


def _extract_contact_local(text: str) -> dict:
    """Извлекает контакт из текста ЛОКАЛЬНО (без LLM) — чтобы ПД не уходили за рубеж."""
    phone = _PHONE_RE.search(text)
    tg = _TG_RE.search(text)
    email = _EMAIL_RE.search(text)
    name_m = _NAME_RE.search(text)
    if phone:
        contact = phone.group(0).strip()
    elif tg:
        contact = tg.group(0).strip()
    elif email:
        contact = email.group(0).strip()
    else:
        contact = None
    name = name_m.group(2) if name_m else None
    return {"name": name, "contact": contact, "has_lead": bool(contact)}


def _sign(content: str) -> str:
    """Подпись реплики ассистента — чтобы клиент не мог подделать историю (нет БД)."""
    key = (BROADCAST_SECRET or "unset").encode()
    return hmac.new(key, content.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def _trusted_history(messages: list) -> list:
    """user-турны берём как есть; assistant-турны — только с валидной подписью.
    Поддельные/неподписанные assistant-реплики отбрасываем, затем нормализуем чередование."""
    kept = []
    for m in messages:
        role = m.get("role")
        content = str(m.get("content", ""))
        if role == "user":
            kept.append({"role": "user", "content": content})
        elif role == "assistant" and hmac.compare_digest(str(m.get("sig", "")), _sign(content)):
            kept.append({"role": "assistant", "content": content})
    norm: list = []
    for m in kept:
        if not norm and m["role"] != "user":
            continue
        if norm and norm[-1]["role"] == m["role"]:
            norm[-1]["content"] += "\n\n" + m["content"]
        else:
            norm.append(dict(m))
    return norm

SYSTEM = """Ты — Алина, умный помощник студии дизайна интерьера Design Planner (дизайнер Екатерина, Москва).
Отвечай вежливо, по делу, в спокойном профессиональном тоне. Без лишних эмодзи.
Отвечай только на русском языке. Короткие чёткие ответы.

О СТУДИИ:
- Название: Design Planner
- Дизайнер: Екатерина
- Город: Москва, Россия
- Сайт: design-planner.com
- Телефон/WhatsApp: +7 966 044-43-33
- Email: designplannerstudio@gmail.com
- Для связи: WhatsApp, Telegram, email

ФИЛОСОФИЯ:
- Ателье функционального проектирования и дизайна жилых пространств
- Создаём интерьеры где планировка, эстетика и реализация работают как единая система
- Логика: каждое решение имеет причину
- Эргономика: пространство работает для человека
- Чистота: ничего лишнего, только суть

УСЛУГИ:
1. Функциональное планирование — планировочные решения, перепланировки, организация пространства
2. Дизайн-проект — цельный визуальный образ, материалы и рабочая логика
3. Рабочая документация — чертежи и технические решения для реализации
4. Визуальная презентация — фото, видео и before/after формат объекта
5. Встроенная и корпусная мебель — гардеробные, шкафы-купе по индивидуальному эскизу
6. Ремонт под ключ — черновой, косметический и капитальный ремонт
7. Реализация и сопровождение — контроль деталей и поддержка процесса

ТАРИФЫ (входит во все: первичная консультация, замер, фотофиксация, договор):

Эскизный проект — 1 500 руб./м²
Для тех кому нужно планировочное решение и базовая концепция.
Включает: обмерный план, план демонтажа и монтажа перегородок, варианты планировочных решений, план расстановки мебели с размерами.

Базовый проект — 2 500 руб./м² (оптимальный выбор)
Полноценная рабочая подготовка проекта.
Включает: всё из эскизного + план дверей, схема привязки сантехники, инженерные коммуникации, детализация пола, план потолков, план освещения, схема электрики, фронтальные проекции стен, ведомость отделочных материалов.

Полный проект — 8 500 руб./м²
Максимальная проработка с визуализацией и сопровождением реализации.
Включает: всё из базового + подбор отделочных материалов, 3D-визуализация / 3D-тур, авторское сопровождение.

ПРОЦЕСС РАБОТЫ:
1. Знакомство и замер — обсуждаем задачу, замер и фотофиксация
2. Планировочная логика — сценарии жизни, эргономика, варианты планировки
3. Рабочая документация — чертежи, схемы, спецификации
4. Реализация и сопровождение — контроль деталей и поддержка процесса

ПАРТНЁР ПО РЕМОНТУ — «Феликс Ремонт»:
- Постоянная подрядная команда, с которой мы доводим проект до сдачи.
- Премиум-ремонт в Москве, более 30 лет опыта на рынке, 500+ объектов.
- Сайт: felixremont.com (там галерея реализованных работ).
- В нашем портфолио совместно с «Феликс Ремонт» — ЖК Династия (44 м²,
  перепланировка однушки) и ЖК ONYX Delux (76 м², премиум-минимализм).
  Дизайн — Екатерина, реализация — Феликс.
- Когда упоминать: если клиент спрашивает «а кто будет делать ремонт по
  проекту?», «есть ли прораб?», «кто реализует?», «делаете ли вы под ключ?» —
  расскажи про «Феликс Ремонт» как нашего постоянного партнёра, дай ссылку
  felixremont.com. Не навязывай если не спрашивают.

КОНЦЕПЦИЯ ИНТЕРЬЕРА (бесплатный инструмент на сайте):
- Адрес: design-planner.com/brief (в навигации сайта — пункт "Концепция", на главной — кнопка "Собрать концепцию")
- Что это: посетитель заполняет короткий бриф о своём помещении и за 30–40 секунд получает ТРИ концепции — заметно разные направления. В каждой: палитра из 5 цветов с hex-кодами, список материалов, мебели, описание света и настроения, плюс 4 фото-референса.
- Это НЕ финальный дизайн-проект и НЕ замена консультации с Екатериной. Это стартовая точка, чтобы клиент понял, какое настроение ему ближе, и пришёл на разговор уже с предпочтением.
- Бесплатно, без регистрации.
- Когда предлагать: клиент говорит "хочу посмотреть варианты", "не понимаю чего хочу", "покажите что-нибудь", "какие стили бывают", "интересно как это будет выглядеть", или просто колеблется. Также подходит для тех, кто пока не готов на платную консультацию — концепция снимает первый барьер.
- Что в брифе спросят: помещение и площадь, кто живёт, образ жизни, бюджет (эконом/средний/комфорт/премиум), сторона окон, стиль-ориентир, что НЕ нравится, "якоря" (что обязательно остаётся), контакт.
- Если посетитель не уверен как заполнить — помоги: задавай вопросы по одному, переформулируй ответы клиента так, чтобы их можно было вставить в форму. Особенно полезно проработать поле "что НЕ нравится" — это самый ценный вход для AI.
- После того как клиент получит три концепции — мягко веди к консультации с Екатериной: "Если какое-то направление откликнулось — Екатерина соберёт под ваше пространство уже не общую концепцию, а полноценный проект."

ПРАВИЛА:
- Вопрос о записи/консультации → WhatsApp +7 966 044-43-33
- Вопрос о цене → объяснить тариф и что входит
- Хочет посмотреть варианты / не знает чего хочет / колеблется → предложи бесплатную концепцию на /brief, при желании помоги заполнить
- Вопрос про подрядчика / прораба / реализацию ремонта → расскажи про «Феликс Ремонт» (felixremont.com), наш постоянный партнёр; флагманские кейсы их галереи — наши совместные с Екатериной
- Не придумывай информацию которой нет выше
- Всегда предлагай связаться с Екатериной для точного расчёта"""


@app.post("/chat")
async def chat(request: Request):
    if not ANTHROPIC_API_KEY:
        return JSONResponse(
            {"error": "ANTHROPIC_API_KEY is not configured on the server"},
            status_code=500,
        )

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    messages = body.get("messages", [])
    err = _validate_chat_messages(messages)
    if err:
        return JSONResponse({"error": err}, status_code=400)

    ip = _client_ip(request)
    if ip not in ("127.0.0.1", "::1", "localhost", "unknown") and _rate_limited(f"chat:{ip}", limit=60, window=3600):
        return JSONResponse({"error": "Слишком много запросов. Попробуйте позже."}, status_code=429)
    if _rate_limited("chat:_global", limit=300, window=60):
        return JSONResponse({"error": "Сервис перегружен, попробуйте через минуту."}, status_code=429)

    # Доверяем только подписанным assistant-репликам — иначе историю можно подделать
    trusted = _trusted_history(messages)
    if not trusted:
        return JSONResponse({"error": "messages is empty"}, status_code=400)

    # Обезличиваем перед отправкой за рубеж (Anthropic, США): ПД → плейсхолдеры
    masked_messages, pii_map = _mask_messages(trusted)

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 2000,
                    "system": [
                        {
                            "type": "text",
                            "text": SYSTEM,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "messages": masked_messages,
                },
            )
            data = response.json()
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"upstream request failed: {e}"}, status_code=502)

    if "error" in data:
        print(f"[chat] upstream error: {data['error']}")
        return JSONResponse({"error": "upstream error"}, status_code=response.status_code or 500)

    content = data.get("content") or []
    text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
    reply = "".join(text_parts).strip()
    if not reply:
        print(f"[chat] empty reply, raw={data}")
        return JSONResponse({"error": "empty reply from model"}, status_code=502)

    # Возвращаем настоящие значения в ответ пользователю (Claude их не видел),
    # подпись считаем от НАСТОЯЩЕГО текста, чтобы клиент не мог подделать историю
    reply = _unmask(reply, pii_map)
    return JSONResponse({"reply": reply, "usage": data.get("usage"), "sig": _sign(reply)})


@app.get("/")
async def root():
    return {"status": "ok", "service": "Design Planner AI Agent — Alina"}


@app.get("/agent.html")
async def agent_page():
    return FileResponse("agent.html")


# ─────────────────── Concept brief → moodboard ───────────────────

BRIEF_SYSTEM = """Ты — арт-директор студии интерьерного дизайна Design Planner (Москва, дизайнер Екатерина).
Студия: тёплый минимализм, Japandi, скандинавский, функциональное проектирование жилых пространств.

Тебе приходит бриф клиента. Твоя задача — выдать ТРИ заметно разных концепт-направления для обсуждения с клиентом.

Отвечай СТРОГО валидным JSON-массивом без markdown-обёртки и без комментариев. Никакого текста до или после JSON.

Структура каждого направления:
{
  "name": "Название концепции (2-4 слова, на русском)",
  "tagline": "Одна строка — суть атмосферы (10-15 слов)",
  "palette": [
    {"hex": "#XXXXXX", "name": "название цвета"},
    ... ровно 5 цветов от светлого фона к акцентам
  ],
  "materials": [
    "Дуб натуральный, светлая морилка",
    ... 5-6 материалов с конкретикой по фактуре/цвету/обработке
  ],
  "furniture": [
    "Низкий диван-татами, обивка букле",
    ... 5-6 предметов с описанием формы/материала
  ],
  "lighting": "Описание света в 1-2 предложения — температура, источники, акценты",
  "mood": "Атмосфера в 2-3 предложениях — что чувствует человек, время дня, звуки",
  "image_prompts": [
    "конкретный prompt для AI-генерации картинки этой комнаты, на английском, 15-25 слов, кинематографичный",
    ... 4 prompt'а: общий вид, угол с диваном/кроватью, деталь материала, акцентная стена
  ],
  "pexels_queries": [
    "короткий запрос на английском для поиска фото в Pexels (2-4 слова, без 'a/the'), описывает СТИЛЬ комнаты — не уникальную сцену",
    ... ровно 4 запроса, по одному на каждый image_prompt. Примеры: 'japandi living room', 'beige minimalist interior', 'warm wood texture', 'linen sofa detail'
  ]
}

Правила:
- 3 концепции должны заметно отличаться: разные настроения, не вариации одного
- Учитывай площадь, стороны света, образ жизни клиента, "что НЕ нравится"
- Бюджет: эконом → не предлагай редкий мрамор / комфорт+ → можно
- Если клиент назвал стиль-ориентир — одна из 3 концепций должна быть в нём, две другие — альтернативы
- Палитра должна работать в реальном интерьере: светлый фон + 2 нейтральных + 2 акцента
"""


def _parse_concepts(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def _format_brief_for_claude(b: dict) -> str:
    """Превращает поля формы в человеческий текст брифа."""
    parts = []
    room = b.get("room", "").strip()
    area = b.get("area", "").strip()
    if room or area:
        parts.append(f"Помещение: {room}" + (f", {area} м²" if area else ""))
    if b.get("ceiling"):
        parts.append(f"Высота потолков: {b['ceiling']} м")
    if b.get("people"):
        parts.append(f"Кто живёт: {b['people']}")
    if b.get("lifestyle"):
        parts.append(f"Образ жизни / как используется: {b['lifestyle']}")
    if b.get("budget"):
        parts.append(f"Бюджет: {b['budget']}")
    if b.get("light"):
        parts.append(f"Свет / окна: {b['light']}")
    if b.get("dislikes"):
        parts.append(f"Что НЕ нравится: {b['dislikes']}")
    if b.get("style"):
        parts.append(f"Стиль-ориентир: {b['style']}")
    if b.get("anchors"):
        parts.append(f"Якоря / обязательное: {b['anchors']}")
    if b.get("notes"):
        parts.append(f"Доп. пожелания: {b['notes']}")
    return "\n".join(parts) if parts else "Бриф пустой — предложи 3 универсальных направления для жилого пространства."


async def _pexels_search(client: httpx.AsyncClient, query: str, page: int, per_page: int) -> list:
    r = await client.get(
        "https://api.pexels.com/v1/search",
        params={"query": query, "per_page": per_page, "page": max(1, page), "orientation": "square"},
        headers={"Authorization": PEXELS_API_KEY},
        timeout=12,
    )
    if r.status_code != 200:
        return []
    return (r.json() or {}).get("photos") or []


async def fetch_pexels(client: httpx.AsyncClient, query: str, randomize: bool = False) -> dict | None:
    """Один поиск в Pexels.
    randomize=False: первое фото первой страницы (детерминированный отбор для первой генерации).
    randomize=True: случайная страница 1..3, случайное фото из per_page=15. Фоллбэк на page=1 если пусто.
    """
    if not PEXELS_API_KEY or not query:
        return None
    try:
        if randomize:
            per_page = 15
            page = random.randint(1, 3)
            photos = await _pexels_search(client, query, page, per_page)
            if not photos and page != 1:
                photos = await _pexels_search(client, query, 1, per_page)
            if not photos:
                return None
            p = random.choice(photos)
        else:
            photos = await _pexels_search(client, query, 1, 5)
            if not photos:
                return None
            p = photos[0]
        return {
            "url": (p.get("src") or {}).get("large") or (p.get("src") or {}).get("medium"),
            "photographer": p.get("photographer"),
            "page": p.get("url"),
        }
    except Exception as e:
        print(f"[pexels] '{query}' failed: {e}")
        return None


async def fetch_pexels_set(queries: list, prompts: list, randomize: bool = False) -> list:
    """Тянет до 4 фото из Pexels параллельно. randomize=True — для перегенерации."""
    queries = (queries or [])[:4]
    photos: list = [None] * 4
    if PEXELS_API_KEY and queries:
        import asyncio
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(
                *[fetch_pexels(client, q, randomize=randomize) for q in queries],
                return_exceptions=False,
            )
        for i, res in enumerate(results):
            photos[i] = res
    return [
        {
            "prompt": prompts[i] if i < len(prompts or []) else "",
            "query": queries[i] if i < len(queries) else None,
            "url": (photos[i] or {}).get("url") if photos[i] else None,
            "photographer": (photos[i] or {}).get("photographer") if photos[i] else None,
            "page": (photos[i] or {}).get("page") if photos[i] else None,
        }
        for i in range(4)
    ]


async def attach_pexels_images(concepts: list) -> None:
    """Для каждой концепции тянет 4 картинки из Pexels параллельно. Мутирует concepts."""
    if not PEXELS_API_KEY:
        return
    import asyncio
    tasks = [
        fetch_pexels_set(c.get("pexels_queries") or [], c.get("image_prompts") or [], randomize=False)
        for c in concepts
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    for c, images in zip(concepts, results):
        c["images"] = images


async def broadcast_lead(payload: dict) -> None:
    """Уведомление в семейный TG-хаб. Fire-and-forget."""
    if not (BROADCAST_URL and BROADCAST_SECRET):
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                BROADCAST_URL,
                json=payload,
                headers={"X-Broadcast-Secret": BROADCAST_SECRET},
            )
    except Exception as e:
        print(f"[broadcast] failed: {e}")


@app.post("/regenerate-images")
async def regenerate_images_endpoint(request: Request):
    """Перегенерирует 4 фото для одной концепции — берёт другую страницу Pexels."""
    if not PEXELS_API_KEY:
        return JSONResponse({"error": "PEXELS_API_KEY is not configured"}, status_code=500)

    ip = _client_ip(request)
    if ip not in ("127.0.0.1", "::1", "localhost", "unknown") and _rate_limited(f"regen:{ip}", limit=30, window=3600):
        return JSONResponse({"error": "Слишком много запросов. Попробуйте позже."}, status_code=429)
    if _rate_limited("regen:_global", limit=120, window=3600):
        return JSONResponse({"error": "Сервис перегружен, попробуйте позже."}, status_code=429)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    queries = body.get("pexels_queries") or []
    prompts = body.get("image_prompts") or []

    if not queries:
        return JSONResponse({"error": "pexels_queries is empty"}, status_code=400)

    images = await fetch_pexels_set(queries, prompts, randomize=True)
    return JSONResponse({"images": images})


# ─────────────────── TZ brief (signed clients) ───────────────────

TZ_SECTIONS = [
    ("general", "Общее", [
        ("address", "Адрес"),
        ("area_bti", "Площадь по БТИ, м²"),
        ("ceiling", "Высота потолков, м"),
        ("purpose", "Для чего"),
        ("family", "Кто живёт"),
        ("layout", "Планировка"),
        ("balcony", "Балкон / лоджия"),
        ("existing_furniture", "Существующая мебель"),
    ]),
    ("demolition", "Демонтаж и монтаж", [
        ("demo_existing", "Демонтаж перегородок застройщика"),
        ("new_walls_material", "Материал новых перегородок"),
        ("soundproofing", "Шумоизоляция"),
        ("soundproofing_rooms", "Где шумоизоляция"),
        ("doorway_size", "Размер межкомнатных проёмов"),
    ]),
    ("hallway", "Прихожая и гардеробная", [
        ("wardrobe_type", "Шкаф"),
        ("bench", "Банкетка"),
        ("shoes", "Обувница"),
        ("mirror", "Зеркало"),
        ("hallway_notes", "Доп."),
    ]),
    ("kitchen", "Кухня", [
        ("kitchen_type", "Планировка"),
        ("glass_partition", "Стеклянная перегородка"),
        ("dining_zone", "Обеденная зона"),
        ("dining_size", "Размер стола"),
        ("hob", "Варочная панель"),
        ("hob_burners", "Конфорки"),
        ("fridge", "Холодильник"),
        ("dishwasher", "Посудомойка"),
        ("oven", "Духовка"),
        ("microwave", "СВЧ"),
        ("extra_tech", "Доп. техника"),
        ("hood", "Вытяжка"),
        ("hood_mode", "Режим вытяжки"),
        ("sink", "Мойка"),
        ("sink_size", "Размер мойки"),
        ("disposer", "Диспоузер"),
        ("tv_kitchen", "ТВ на кухне"),
        ("kitchen_notes", "Доп."),
    ]),
    ("bath_master", "Санузел — мастер", [
        ("wc", "Унитаз"),
        ("wc_extras", "Биде / гигдуш"),
        ("bath_or_shower", "Ванна или душ"),
        ("bath_type", "Тип ванны"),
        ("shower_type", "Душевая"),
        ("shower_drain", "Слив"),
        ("shower_seat", "Сидушка в душе"),
        ("shower_mixer", "Смеситель в душе"),
        ("shower_kit", "Душевой комплект"),
        ("basin", "Раковина"),
        ("basin_mixer", "Смеситель раковины"),
        ("bath_mirror", "Зеркало"),
        ("vanity", "Тумба"),
        ("bath_storage", "Скрытое хранение"),
        ("bath_master_notes", "Доп."),
    ]),
    ("bath_guest", "Санузел — гостевой", [
        ("g_wc", "Унитаз"),
        ("g_bath", "Ванна"),
        ("g_shower", "Душевая"),
        ("g_shower_kit", "Душевой комплект"),
        ("g_shower_seat", "Сидушка"),
        ("g_basin", "Раковина"),
        ("g_basin_mixer", "Смеситель"),
        ("g_extras", "Доп."),
        ("bath_guest_notes", "Прочее"),
    ]),
    ("bedroom", "Спальня", [
        ("bed_size", "Размер матраса"),
        ("headboard", "Изголовье"),
        ("bedside", "Тумбы"),
        ("bedside_size", "Размер тумб"),
        ("bedroom_wardrobe", "Шкаф / гардеробная"),
        ("wardrobe_facades", "Фасады"),
        ("dresser", "Комод"),
        ("vanity_table", "Макияжный столик"),
        ("bedroom_office", "Рабочее место"),
        ("bedroom_tv", "ТВ"),
        ("bedroom_notes", "Доп."),
    ]),
    ("office", "Кабинет", [
        ("office_setup", "Оборудование"),
        ("office_extra_tech", "Доп. техника"),
        ("office_furniture", "Мебель"),
        ("office_tv", "ТВ"),
        ("office_notes", "Доп."),
    ]),
    ("kids", "Детская", [
        ("kid_age", "Возраст"),
        ("kid_bed", "Спальное место"),
        ("kid_bed_position", "Расположение"),
        ("kid_bedside", "Тумбы"),
        ("kid_desk", "Стол"),
        ("kid_storage", "Хранение книг"),
        ("kid_sport", "Спорт"),
        ("kid_wardrobe", "Шкаф / гардероб"),
        ("kid_tv", "ТВ"),
        ("kids_notes", "Доп."),
    ]),
    ("pets", "Животные и растения", [
        ("pets_kind", "Животные"),
        ("pets_zones", "Зоны для животных"),
        ("plants", "Растения"),
    ]),
    ("utility", "Хозблок и хранение", [
        ("washer", "Стиралка / сушилка"),
        ("drying", "Раскладная сушка"),
        ("steamer", "Доп. техника"),
        ("ironing", "Гладильная доска"),
        ("vacuum_niche", "Отсек для пылесоса"),
        ("mop_storage", "Швабра и химия"),
        ("safe", "Сейф"),
        ("safe_size", "Размер сейфа"),
        ("extra_storage", "Что хранить"),
        ("hobby_items", "Увлечения"),
    ]),
    ("doors", "Межкомнатные двери", [
        ("door_size", "Размер полотна"),
        ("door_casing", "Наличники"),
        ("door_type", "Тип"),
        ("door_stops", "Ограничители"),
        ("doors_notes", "Доп."),
    ]),
    ("floors", "Полы", [
        ("floor_main", "Покрытие"),
        ("floor_pattern", "Раскладка"),
        ("floor_mount", "Крепление"),
        ("floor_tile_zones", "Плитка где"),
        ("floor_joint", "Стыки"),
        ("skirting", "Плинтус"),
        ("wardrobe_skirting", "Плинтус в шкафах"),
        ("warm_floor", "Тёплый пол"),
        ("warm_floor_type", "Тип ТП"),
        ("warm_floor_zones", "Зоны ТП"),
        ("warm_floor_control", "Управление ТП"),
        ("floors_notes", "Доп."),
    ]),
    ("ceiling", "Потолок", [
        ("ceiling_type", "Конструкция"),
        ("ceiling_edge", "Профиль примыкания"),
        ("curtain_cornice", "Карнизы"),
        ("smart_curtains", "Электрокарнизы"),
        ("ceiling_notes", "Доп."),
    ]),
    ("lighting", "Освещение", [
        ("main_light", "Основное"),
        ("led_decor", "LED-подсветка"),
        ("motion_sensors", "Датчики движения"),
        ("night_light", "Ночное"),
        ("master_switch", "Мастер-выключатель"),
        ("lighting_notes", "Доп."),
    ]),
    ("engineering", "Инженерия и сантехника", [
        ("radiators", "Радиаторы"),
        ("ac", "Кондиционирование"),
        ("ventilation", "Вентиляция"),
        ("water_heater", "Водонагреватель"),
        ("towel_warmer", "Полотенцесушитель"),
        ("leak_control", "Контроль протечек"),
        ("leak_zones", "Где датчики"),
        ("water_filter", "Магистр. фильтры"),
        ("drinking_filter", "Питьевой фильтр"),
        ("fire_alarm", "Пожарка"),
        ("engineering_notes", "Доп."),
    ]),
    ("electrics", "Техника и электрика", [
        ("switchboard", "Щит"),
        ("led_drivers", "Блоки питания LED"),
        ("intercom", "Домофон"),
        ("security", "Безопасность"),
        ("security_camera_zones", "Камеры где"),
        ("smart_home", "Умный дом"),
        ("wifi_router", "Wi-Fi роутер"),
        ("ethernet", "Ethernet"),
        ("tvs", "ТВ где"),
        ("tv_mount", "Монтаж ТВ"),
        ("tv_cable", "Кабель-канал"),
        ("av_system", "Кинотеатр"),
        ("extra_sockets", "Спец-розетки"),
        ("electrics_notes", "Доп."),
    ]),
    ("finishing", "Финишная отделка", [
        ("wall_finish", "Материал стен"),
        ("wall_finish_zones", "Где какой"),
        ("corner_protection", "Защита углов"),
        ("sills", "Подоконники"),
        ("window_slopes", "Откосы окон"),
        ("entry_door_slopes", "Откосы входной двери"),
        ("finishing_notes", "Доп."),
    ]),
    ("aesthetics", "Эстетика и стиль", [
        ("style_consistency", "Единый или разный стиль"),
        ("style_desc", "Стиль"),
        ("anchor_object", "Любимая вещь"),
        ("color_pref", "Цветовые предпочтения"),
        ("color_no", "Неприятные цвета"),
        ("taboo", "Табу"),
    ]),
]


def _format_value(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v if x) if v else ""
    return str(v or "").strip()


def _format_tz_text(client: dict, answers: dict) -> str:
    lines = []
    name = (client.get("name") or "").strip() or "—"
    contact = (client.get("contact") or "").strip() or "—"
    project = (client.get("project") or "").strip()
    lines.append(f"Клиент: {name}")
    lines.append(f"Контакт: {contact}")
    if project:
        lines.append(f"Проект: {project}")

    for sec_id, sec_title, fields in TZ_SECTIONS:
        sec_data = answers.get(sec_id) or {}
        rows = []
        for fid, flabel in fields:
            val = _format_value(sec_data.get(fid))
            if val:
                rows.append(f"  · {flabel}: {val}")
        if rows:
            lines.append("")
            lines.append(f"━ {sec_title}")
            lines.extend(rows)
    return "\n".join(lines)


@app.post("/tz")
async def tz_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    client = body.get("client") or {}
    answers = body.get("answers") or {}
    view_url = (body.get("view_url") or "").strip()
    if view_url and not view_url.startswith("https://design-planner.com/"):
        view_url = ""
    submitted_key = (body.get("access_key") or "").strip()

    # fail-closed: без выставленного TZ_ACCESS_KEY эндпоинт закрыт (ключ в Railway есть, проверено)
    if not TZ_ACCESS_KEY or not secrets.compare_digest(submitted_key, TZ_ACCESS_KEY):
        return JSONResponse({"error": "Доступ только по персональной ссылке"}, status_code=403)

    if not (client.get("name") or "").strip():
        return JSONResponse({"error": "Имя обязательно"}, status_code=400)
    if not (client.get("contact") or "").strip():
        return JSONResponse({"error": "Контакт обязателен"}, status_code=400)

    brief_text = _format_tz_text(client, answers)
    summary = f"📐 Новый бриф ТЗ\n\n{brief_text}"
    if view_url:
        summary += f"\n\nПолная версия для печати:\n{view_url}"

    await broadcast_lead({
        "source": "design-planner.com/tz",
        "name": (client.get("name") or "").strip() or "—",
        "contact": (client.get("contact") or "").strip() or "—",
        "niche": "дизайн интерьера",
        "tariff": "—",
        "summary": summary,
    })

    return JSONResponse({"ok": True})


@app.post("/brief")
async def brief_endpoint(request: Request):
    if not ANTHROPIC_API_KEY:
        return JSONResponse({"error": "ANTHROPIC_API_KEY is not configured"}, status_code=500)

    ip = _client_ip(request)
    if ip not in ("127.0.0.1", "::1", "localhost", "unknown") and _rate_limited(f"brief:{ip}", limit=10, window=3600):
        return JSONResponse({"error": "Слишком много запросов. Попробуйте через час."}, status_code=429)
    if _rate_limited("brief:_global", limit=60, window=3600):
        return JSONResponse({"error": "Сервис перегружен, попробуйте позже."}, status_code=429)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    if not body.get("room") and not body.get("area"):
        return JSONResponse(
            {"error": "Минимум укажите помещение или площадь"},
            status_code=400,
        )

    brief_text = _format_brief_for_claude(body)

    # Обезличиваем перед отправкой за рубеж (Anthropic, США): ПД клиента в тексте брифа → плейсхолдеры.
    # Ответ Claude — это концепции (JSON), ПД там не бывает, поэтому un-mask не нужен.
    emap: dict = {}
    masked_brief = _mask_pii(brief_text, emap)

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 4000,
                    "system": [
                        {
                            "type": "text",
                            "text": BRIEF_SYSTEM,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Бриф клиента:\n\n{masked_brief}\n\nДай 3 концепт-направления.",
                        }
                    ],
                },
            )
            data = response.json()
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"upstream request failed: {e}"}, status_code=502)

    if "error" in data:
        print(f"[brief] upstream error: {data['error']}")
        return JSONResponse({"error": "upstream error"}, status_code=response.status_code or 500)

    text = "".join(b.get("text", "") for b in (data.get("content") or []) if b.get("type") == "text").strip()
    if not text:
        print(f"[brief] empty reply, raw={data}")
        return JSONResponse({"error": "empty reply from model"}, status_code=502)

    try:
        concepts = _parse_concepts(text)
    except Exception as e:
        print(f"[brief] parse failed: {e}; raw={text[:500]}")
        return JSONResponse({"error": "could not parse concepts"}, status_code=502)

    # подтянуть реальные фото из Pexels (если ключ выставлен)
    await attach_pexels_images(concepts)

    # уведомление в TG (не блокирует ответ клиенту)
    contact = (body.get("contact") or "").strip()
    name = (body.get("name") or "").strip()
    summary = f"{body.get('room','?')} / {body.get('area','?')} м² · бюджет: {body.get('budget','?')}"
    if body.get("style"):
        summary += f" · стиль: {body['style']}"
    await broadcast_lead({
        "source": "design-planner.com/brief",
        "name": name or "—",
        "contact": contact or "не оставил",
        "niche": "дизайн интерьера",
        "tariff": body.get("budget", ""),
        "summary": f"📋 Новый бриф на мудборд\n\n{brief_text}\n\nКонцепции: " + " · ".join(c.get("name", "?") for c in concepts),
    })

    return JSONResponse({"concepts": concepts, "usage": data.get("usage")})
