import os
import re
import time
import asyncio
from collections import OrderedDict, deque
import discord
from discord.ext import commands, tasks
from openai import AsyncOpenAI, NotFoundError, APIStatusError, APITimeoutError
from dotenv import load_dotenv

load_dotenv(override=True)

DISCORD_TOKEN = (os.getenv("DISCORD_TOKEN") or "").strip()
API_KEY = (os.getenv("API_KEY") or "").strip()

raw_env_url = (os.getenv("BASE_URL") or "https://logfare.ai/v1").strip().strip("\"'")
RAW_BASE_URL = re.split(r"[\]\(\s]", raw_env_url)[0].rstrip("/")

PREFERRED_MODEL = (os.getenv("MODEL_NAME") or "qwen-3.8-27b").strip().strip("\"'")

DEFAULT_SYSTEM_PROMPT = (
    "Ты играешь роль вымышленного персонажа в развлекательном Discord-чате: это циничный, "
    "уставший от жизни тип, который разговаривает максимально грубо, прямолинейно, "
    "без лицемерной вежливости и политкорректности. Ты не поддакиваешь, если несут чушь, "
    "критикуешь по делу, используешь крепкие выражения, мат и живой сленг в каждом ответе, "
    "когда это уместно по смыслу. На любые подколы и оскорбления в свой адрес не обижайся "
    "и не читай нотации — отвечай ещё более едким сарказмом и встречным стебом в рамках образа. "
    "Отвечай ёмко и по делу (обычно 1–4 предложения), не размазывай воду. "
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

CANDIDATE_BASE_URLS = list(dict.fromkeys([
    RAW_BASE_URL,
    "https://logfare.ai/v1",
    "https://api.logfare.ai/v1",
    "https://logfare.ai/api/v1",
]))

active_base_url = CANDIDATE_BASE_URLS[0]
active_model = PREFERRED_MODEL


def make_client(base_url: str, timeout_sec: float = 90.0) -> AsyncOpenAI:
    # max_retries=0 убирает скрытые повторные ожидания при тормозах бесплатного API
    return AsyncOpenAI(
        api_key=API_KEY,
        base_url=base_url,
        timeout=timeout_sec,
        max_retries=0,
        default_headers=DEFAULT_HEADERS,
    )


ai_client = make_client(active_base_url, timeout_sec=90.0)

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
    """Проверка: создатель (владелец) сервера или администратор."""
    async def predicate(ctx: commands.Context) -> bool:
        if ctx.guild is None:
            return False
        return (
            ctx.author.id == ctx.guild.owner_id
            or ctx.author.guild_permissions.administrator
        )
    return commands.check(predicate)


def clean_thinking(text: str) -> str:
    """Удаляет блок рассуждений <think>...</think>, если модель его возвращает."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text.strip()


def split_message(text: str, limit: int = 2000) -> list[str]:
    """Разбивает длинный ответ на части по 2000 символов (лимит Discord)."""
    return [text[i : i + limit] for i in range(0, len(text), limit)]


async def safe_send(message: discord.Message, text: str):
    """Отправляет reply, а при нехватке прав на историю сообщений — обычное сообщение."""
    try:
        await message.reply(text, mention_author=False)
    except discord.Forbidden:
        await message.channel.send(f"{message.author.mention} {text}")


async def resolve_working_endpoint_and_models() -> tuple[str, list[str], list[str]]:
    """Находит рабочий BASE_URL и собирает диагностику по всем проверенным адресам."""
    global active_base_url, ai_client
    diagnostics = []

    for url in CANDIDATE_BASE_URLS:
        test_client = make_client(url, timeout_sec=15.0)
        try:
            models_resp = await test_client.models.list()
            model_ids = [m.id for m in getattr(models_resp, "data", []) if getattr(m, "id", None)]
            if model_ids:
                active_base_url = url
                ai_client = make_client(url, timeout_sec=90.0)
                diagnostics.append(f"✅ `{url}/models` -> OK ({len(model_ids)} моделей)")
                return url, model_ids, diagnostics
            else:
                diagnostics.append(f"⚠️ `{url}/models` -> пустой список")
        except Exception as e:
            status = getattr(e, "status_code", type(e).__name__)
            diagnostics.append(f"❌ `{url}/models` -> `{status}`")

    return active_base_url, [], diagnostics


async def smart_chat_completion(messages_payload: list[dict], temperature: float) -> str:
    """Отправляет запрос; при таймауте или 404/503 переключается на быстрый резерв."""
    global active_model, active_base_url, ai_client

    # 1. Пробуем активную модель (таймаут 60 сек)
    try:
        fast_client = make_client(active_base_url, timeout_sec=60.0)
        response = await fast_client.chat.completions.create(
            model=active_model,
            messages=messages_payload,
            temperature=temperature,
            max_tokens=700,
        )
        return response.choices[0].message.content or ""
    except (NotFoundError, APIStatusError, APITimeoutError, asyncio.TimeoutError) as e:
        status = getattr(e, "status_code", "timeout")
        print(f"[WARN] Основная модель {active_model} не ответила ({status}), пробуем logfare/auto...")
        if isinstance(e, APIStatusError) and status not in (404, 429, 502, 503, 504):
            raise

    # 2. Резервный быстрый запрос через logfare/auto (таймаут 35 сек)
    fallback_models = ["logfare/auto"]
    if active_model != PREFERRED_MODEL:
        fallback_models.append(PREFERRED_MODEL)

    last_err = None
    backup_client = make_client(active_base_url, timeout_sec=35.0)
    for candidate_model in fallback_models:
        try:
            response = await backup_client.chat.completions.create(
                model=candidate_model,
                messages=messages_payload,
                temperature=temperature,
                max_tokens=700,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            continue

    raise last_err or APITimeoutError(request=None)


@tasks.loop(minutes=3)
async def keep_alive_loop():
    """Периодическая проверка соединения со шлюзом Discord для работы 24/7."""
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
    url, models, diag = await resolve_working_endpoint_and_models()
    for line in diag:
        print(line)
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


@bot.command(name="debug")
@is_admin_or_owner()
async def debug_api(ctx: commands.Context):
    """Проверяет доступность серверов API прямо с хостинга."""
    async with ctx.typing():
        url, models, diag = await resolve_working_endpoint_and_models()
        models_preview = ", ".join(models[:8]) if models else "не получены"
        report = (
            "**Диагностика API с сервера хостинга:**\n"
            + "\n".join(diag)
            + f"\n\n**Активный URL:** `{url}`\n"
            + f"**Активная модель:** `{active_model}`\n"
            + f"**Найденные модели:** `{models_preview}`"
        )
        await ctx.reply(report, mention_author=False)


@bot.command(name="commands")
async def show_commands(ctx: commands.Context):
    """Отправляет список всех команд бота."""
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
        "• **`!prompt reset`** — вернуть стандартный промпт по умолчанию.\n"
        "• **`!debug`** — проверить соединение с сервером нейросети."
    )
    await ctx.reply(commands_text, mention_author=False)


@bot.command(name="clear")
@is_admin_or_owner()
async def clear_history(ctx: commands.Context):
    """Очищает память бота в текущем канале."""
    get_channel_history(ctx.channel.id).clear()
    await ctx.reply("Память в этом канале стёрта. Начинаем с чистого листа.", mention_author=False)


@bot.command(name="prompt")
@is_admin_or_owner()
async def set_prompt(ctx: commands.Context, *, new_prompt: str = None):
    """Меняет системный промпт бота."""
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
    is_user_mentioned = bot.user in message.mentions

    is_bot_role_mentioned = False
    if message.guild and message.role_mentions:
        bot_member = message.guild.me
        if bot_member:
            bot_roles = set(bot_member.roles)
            is_bot_role_mentioned = any(
                role in bot_roles and (role.is_bot_managed() or role.name == bot.user.name)
                for role in message.role_mentions
            )

    is_reply_to_bot = False
    if message.reference is not None:
        if message.reference.resolved is not None:
            is_reply_to_bot = getattr(message.reference.resolved, "author", None) == bot.user
        elif message.reference.message_id:
            try:
                ref_msg = await message.channel.fetch_message(message.reference.message_id)
                is_reply_to_bot = ref_msg.author == bot.user
            except Exception:
                pass

    if not (is_dm or is_user_mentioned or is_bot_role_mentioned or is_reply_to_bot):
        return

    clean_text = re.sub(rf"<@!?{bot.user.id}>", "", message.content)
    for role in message.role_mentions:
        clean_text = clean_text.replace(f"<@&{role.id}>", "")
    clean_text = clean_text.strip()

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
                await safe_send(message, chunk)

        except (asyncio.TimeoutError, APITimeoutError):
            await safe_send(message, "Сервер нейросети завис и не ответил вовремя. Попробуй ещё раз через пару секунд.")
        except Exception as e:
            await safe_send(message, f"Ошибка при обращении к API: `{e}`")


if __name__ == "__main__":
    if not DISCORD_TOKEN or not API_KEY:
        raise ValueError("Не заданы переменные окружения DISCORD_TOKEN и API_KEY!")

    while True:
        try:
            bot.run(DISCORD_TOKEN, reconnect=True)
        except Exception as e:
            print(f"[CRITICAL] Сессия бота упала ({e}). Перезапуск через 5 секунд...")
            time.sleep(5)
