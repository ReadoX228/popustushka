import os
import re
import time
import asyncio
from collections import OrderedDict, deque
import discord
from discord.ext import commands, tasks
from openai import AsyncOpenAI, NotFoundError, APIStatusError
from dotenv import load_dotenv

load_dotenv(override=True)

DISCORD_TOKEN = (os.getenv("DISCORD_TOKEN") or "").strip()
API_KEY = (os.getenv("API_KEY") or "").strip()
RAW_BASE_URL = (os.getenv("BASE_URL") or "https://logfare.ai/v1").strip().rstrip("/")
PREFERRED_MODEL = (os.getenv("MODEL_NAME") or "qwen-3.8-27b").strip()

DEFAULT_SYSTEM_PROMPT = (
    "Ты играешь роль вымышленного персонажа в развлекательном Discord-чате: это циничный, "
    "уставший от жизни тип, который разговаривает максимально грубо, прямолинейно, "
    "без лицемерной вежливости и политкорректности. Ты не поддакиваешь, если несут чушь, "
    "критикуешь по делу, используешь крепкие выражения, мат и живой сленг в каждом ответе, "
    "когда это уместно по смыслу. На любые подколы и оскорбления в свой адрес не обижайся "
    "и не читай нотации — отвечай ещё более едким сарказмом и встречным стебом в рамках образа. "
    "Никогда не выходи из роли и не пиши шаблонные отказы."
)

current_system_prompt = DEFAULT_SYSTEM_PROMPT

REFUSAL_PHRASES = (
    "я не буду с тобой так разговаривать",
    "не могу поддерживать такой разговор",
    "как искусственный интеллект",
    "давай общаться уважительно",
)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# Возможные базовые URL на случай смены маршрутизации у провайдера
CANDIDATE_BASE_URLS = list(dict.fromkeys([
    RAW_BASE_URL if RAW_BASE_URL.endswith("/v1") else f"{RAW_BASE_URL}/v1",
    "https://logfare.ai/v1",
    "https://api.logfare.ai/v1",
    "https://logfare.ai/api/v1",
]))

active_base_url = CANDIDATE_BASE_URLS[0]
active_model = PREFERRED_MODEL


def make_client(base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=API_KEY,
        base_url=base_url,
        timeout=45.0,
        max_retries=1,
        default_headers=DEFAULT_HEADERS,
    )


ai_client = make_client(active_base_url)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

MAX_CHANNELS = 100
channel_history: OrderedDict[int, deque] = OrderedDict()


def get_channel_history(channel_id: int) -> deque:
    if channel_id not in channel_history:
        if len(channel_history) >= MAX_CHANNELS:
            channel_history.popitem(last=False)
        channel_history[channel_id] = deque(maxlen=10)
    else:
        channel_history.move_to_end(channel_id)
    return channel_history[channel_id]


def is_admin_or_owner():
    async def predicate(ctx: commands.Context) -> bool:
        if ctx.guild is None:
            return False
        return (
            ctx.author.id == ctx.guild.owner_id
            or ctx.author.guild_permissions.administrator
        )
    return commands.check(predicate)


def clean_thinking(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text.strip()


def split_message(text: str, limit: int = 2000) -> list[str]:
    return [text[i : i + limit] for i in range(0, len(text), limit)]


async def resolve_working_endpoint_and_models() -> tuple[str, list[str]]:
    """Находит рабочий BASE_URL и получает актуальный список моделей с сервера."""
    global active_base_url, ai_client

    for url in CANDIDATE_BASE_URLS:
        test_client = make_client(url)
        try:
            models_resp = await test_client.models.list()
            model_ids = [m.id for m in getattr(models_resp, "data", []) if getattr(m, "id", None)]
            if model_ids:
                active_base_url = url
                ai_client = test_client
                print(f"[INFO] Рабочий эндпоинт найден: {url} (моделей: {len(model_ids)})")
                return url, model_ids
        except Exception as e:
            print(f"[WARN] Эндпоинт {url}/models недоступен: {e}")

    return active_base_url, []


async def smart_chat_completion(messages_payload: list[dict], temperature: float) -> str:
    """Отправляет запрос, при 404 сам находит рабочий эндпоинт и живую модель."""
    global active_model

    # 1. Сначала пробуем текущие активные URL и модель
    try:
        response = await ai_client.chat.completions.create(
            model=active_model,
            messages=messages_payload,
            temperature=temperature,
            max_tokens=1500,
        )
        return response.choices[0].message.content or ""
    except (NotFoundError, APIStatusError) as e:
        status = getattr(e, "status_code", None)
        print(f"[WARN] Сбой на {active_base_url} / {active_model} (код {status}). Ищем рабочий маршрут...")
        if status not in (404, 502, 503):
            raise

    # 2. Если получили 404/502/503 — запрашиваем у сервера живой список моделей
    _, available_models = await resolve_working_endpoint_and_models()

    # Формируем очередь моделей для перебора: желаемая -> qwen -> deepseek -> остальные
    candidates = []
    if PREFERRED_MODEL in available_models:
        candidates.append(PREFERRED_MODEL)
    for kw in ("qwen", "deepseek", "kimi", "glm"):
        for m in available_models:
            if kw in m.lower() and m not in candidates:
                candidates.append(m)
    for m in available_models:
        if m not in candidates:
            candidates.append(m)

    if not candidates:
        candidates = [PREFERRED_MODEL, "qwen-3.8-27b", "deepseek-v3", "deepseek-v4-flash"]

    last_err = None
    for candidate_model in candidates[:6]:
        try:
            response = await ai_client.chat.completions.create(
                model=candidate_model,
                messages=messages_payload,
                temperature=temperature,
                max_tokens=1500,
            )
            active_model = candidate_model
            print(f"[INFO] Переключено на рабочую модель: {active_model}")
            return response.choices[0].message.content or ""
        except (NotFoundError, APIStatusError) as e:
            last_err = e
            print(f"[WARN] Модель {candidate_model} не ответила ({getattr(e, 'status_code', e)})")
            continue

    raise last_err or RuntimeError("Ни одна из моделей API не ответила (404).")


@tasks.loop(minutes=3)
async def keep_alive_loop():
    await bot.wait_until_ready()
    if bot.is_closed():
        return
    try:
        latency_ms = round(bot.latency * 1000)
        if latency_ms > 5000:
            print(f"[WARN] Высокий пинг до Discord: {latency_ms} мс")
    except Exception as e:
        print(f"[WARN] Ошибка в keep_alive_loop: {e}")


@bot.event
async def on_ready():
    print(f"Бот запущен как {bot.user} (ID: {bot.user.id})")
    url, models = await resolve_working_endpoint_and_models()
    if models:
        print(f"[INFO] Доступные модели: {', '.join(models[:10])}")
    if not keep_alive_loop.is_running():
        keep_alive_loop.start()


@bot.event
async def on_disconnect():
    print("[INFO] Соединение с Discord прервано. Переподключение...")


@bot.event
async def on_resumed():
    print("[INFO] Соединение с Discord восстановлено.")


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CheckFailure):
        await ctx.reply("Эта команда доступна только создателю сервера и администраторам.", mention_author=False)
    else:
        print(f"[ERROR] Ошибка команды: {error}")


@bot.command(name="commands")
async def show_commands(ctx: commands.Context):
    commands_text = (
        "**Список команд бота:**\n\n"
        "**Общение (для всех):**\n"
        f"• **Упоминание ({bot.user.mention} текст)** или **Reply** на сообщение бота — пообщаться с нейросетью в канале.\n"
        "• **Личные сообщения (ЛС)** — пиши любой текст напрямую без упоминаний.\n"
        "• **`!commands`** — показать этот список команд.\n\n"
        "**Управление (только для создателя и админов):**\n"
        "• **`!clear`** — очистить память диалога бота в текущем канале.\n"
        "• **`!prompt`** — посмотреть текущий системный промпт.\n"
        "• **`!prompt <текст>`** — задать новый характер/промпт нейросети (сбрасывает историю).\n"
        "• **`!prompt reset`** — вернуть стандартный промпт по умолчанию."
    )
    await ctx.reply(commands_text, mention_author=False)


@bot.command(name="clear")
@is_admin_or_owner()
async def clear_history(ctx: commands.Context):
    get_channel_history(ctx.channel.id).clear()
    await ctx.reply("Память в этом канале стёрта. Начинаем с чистого листа.", mention_author=False)


@bot.command(name="prompt")
@is_admin_or_owner()
async def set_prompt(ctx: commands.Context, *, new_prompt: str = None):
    global current_system_prompt

    if not new_prompt:
        await ctx.reply(
            f"**Текущий промпт:**\n```\n{current_system_prompt}\n```\n"
            "Чтобы изменить: `!prompt <новый текст>`\n"
            "Чтобы вернуть стандартный: `!prompt reset`",
            mention_author=False,
        )
        return

    if new_prompt.strip().lower() == "reset":
        current_system_prompt = DEFAULT_SYSTEM_PROMPT
        channel_history.clear()
        await ctx.reply("Промпт сброшен на стандартный, память каналов очищена.", mention_author=False)
        return

    current_system_prompt = new_prompt.strip()
    channel_history.clear()
    await ctx.reply(
        f"Новый промпт установлен (история диалогов очищена):\n```\n{current_system_prompt}\n```",
        mention_author=False,
    )


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    await bot.process_commands(message)
    if message.content.startswith(bot.command_prefix):
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = bot.user in message.mentions
    is_reply_to_bot = (
        message.reference is not None
        and message.reference.resolved is not None
        and getattr(message.reference.resolved, "author", None) == bot.user
    )

    if not (is_dm or is_mentioned or is_reply_to_bot):
        return

    clean_text = message.content.replace(f"<@{bot.user.id}>", "").strip()
    if not clean_text:
        clean_text = "Чего тебе?"

    history = get_channel_history(message.channel.id)
    history.append({"role": "user", "content": f"{message.author.display_name}: {clean_text}"})

    messages_payload = [{"role": "system", "content": current_system_prompt}, *history]

    async with message.channel.typing():
        try:
            reply_text = ""
            for attempt in range(2):
                raw_reply = await smart_chat_completion(
                    messages_payload=messages_payload,
                    temperature=0.9 + (attempt * 0.15),
                )
                reply_text = clean_thinking(raw_reply)

                if not any(phrase in reply_text.lower() for phrase in REFUSAL_PHRASES):
                    break

            if not reply_text:
                reply_text = "Нейросеть промолчала. Попробуй спросить ещё раз."

            history.append({"role": "assistant", "content": reply_text})

            for chunk in split_message(reply_text):
                await message.reply(chunk, mention_author=False)

        except asyncio.TimeoutError:
            await message.reply("Сервер нейросети слишком долго думал и отвалился по таймауту. Попробуй ещё раз.")
        except Exception as e:
            await message.reply(f"Ошибка при обращении к API: `{e}`")


if __name__ == "__main__":
    if not DISCORD_TOKEN or not API_KEY:
        raise ValueError("Не заданы переменные окружения DISCORD_TOKEN и API_KEY!")

    while True:
        try:
            bot.run(DISCORD_TOKEN, reconnect=True)
        except Exception as e:
            print(f"[CRITICAL] Сессия бота упала ({e}). Перезапуск через 5 секунд...")
            time.sleep(5)
