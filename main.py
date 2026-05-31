from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import httpx
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

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

ПРАВИЛА:
- Вопрос о записи/консультации → WhatsApp +7 966 044-43-33
- Вопрос о цене → объяснить тариф и что входит
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
                    "system": SYSTEM,
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
    return JSONResponse({"reply": reply})


@app.get("/")
async def root():
    return {"status": "ok", "service": "Design Planner AI Agent — Alina"}


@app.get("/agent.html")
async def agent_page():
    return FileResponse("agent.html")
