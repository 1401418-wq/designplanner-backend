from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import httpx
import json
import os
import random
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BROADCAST_URL = os.environ.get("BROADCAST_URL", "")
BROADCAST_SECRET = os.environ.get("BROADCAST_SECRET", "")
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")

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
- Не придумывай информацию которой нет выше
- Всегда предлагай связаться с Екатериной для точного расчёта"""


@app.post("/chat")
async def chat(request: Request):
    if not ANTHROPIC_API_KEY:
        return JSONResponse(
            {"error": "ANTHROPIC_API_KEY is not configured on the server"},
            status_code=500,
        )

    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        return JSONResponse({"error": "messages is empty"}, status_code=400)

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
                    "messages": messages,
                },
            )
            data = response.json()
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"upstream request failed: {e}"}, status_code=502)

    if "error" in data:
        return JSONResponse({"error": data["error"]}, status_code=response.status_code or 500)

    content = data.get("content") or []
    text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
    reply = "".join(text_parts).strip()
    if not reply:
        return JSONResponse(
            {"error": "empty reply from model", "raw": data},
            status_code=502,
        )
    return JSONResponse({"reply": reply, "usage": data.get("usage")})


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

    body = await request.json()
    queries = body.get("pexels_queries") or []
    prompts = body.get("image_prompts") or []

    if not queries:
        return JSONResponse({"error": "pexels_queries is empty"}, status_code=400)

    images = await fetch_pexels_set(queries, prompts, randomize=True)
    return JSONResponse({"images": images})


@app.post("/brief")
async def brief_endpoint(request: Request):
    if not ANTHROPIC_API_KEY:
        return JSONResponse({"error": "ANTHROPIC_API_KEY is not configured"}, status_code=500)

    body = await request.json()
    if not body.get("room") and not body.get("area"):
        return JSONResponse(
            {"error": "Минимум укажите помещение или площадь"},
            status_code=400,
        )

    brief_text = _format_brief_for_claude(body)

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
                            "content": f"Бриф клиента:\n\n{brief_text}\n\nДай 3 концепт-направления.",
                        }
                    ],
                },
            )
            data = response.json()
    except httpx.HTTPError as e:
        return JSONResponse({"error": f"upstream request failed: {e}"}, status_code=502)

    if "error" in data:
        return JSONResponse({"error": data["error"]}, status_code=response.status_code or 500)

    text = "".join(b.get("text", "") for b in (data.get("content") or []) if b.get("type") == "text").strip()
    if not text:
        return JSONResponse({"error": "empty reply from model", "raw": data}, status_code=502)

    try:
        concepts = _parse_concepts(text)
    except Exception as e:
        return JSONResponse({"error": f"could not parse concepts: {e}", "raw": text[:500]}, status_code=502)

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
