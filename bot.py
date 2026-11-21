import os
import re
import io
import time
import base64
import asyncio
from collections import OrderedDict, deque
import aiohttp
import discord
from discord import app_commands
from discord.ext import commands, tasks
from openai import AsyncOpenAI, NotFoundError, APIStatusError, APITimeoutError
from dotenv import load_dotenv

load_dotenv(override=True)

DISCORD_TOKEN = (os.getenv("DISCORD_TOKEN") or "").strip()
API_KEY = (os.getenv("API_KEY") or "").strip()

raw_env_url = (os.getenv("BASE_URL") or "https://logfare.ai/v1").strip().strip("\"'")
RAW_BASE_URL = re.split(r"[\]\(\s]", raw_env_url)[0].rstrip("/")

PREFERRED_MODEL = (os.getenv("MODEL_NAME") or "qwen-3.8-27b").strip().strip("\"'")
DEFAULT_IMAGE_MODEL = (os.getenv("IMAGE_MODEL") or "flux-2-dev").strip().strip("\"'")

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


async def sync_all_slash_commands() -> tuple[int, int]:
    """Мгновенно синхронизирует слеш-команды для всех серверов и глобально."""
    synced_global = await bot.tree.sync()
    synced_guilds = 0
    for guild in bot.guilds:
        try:
            bot.tree.copy_global_to(guild=guild)
            await bot.tree.sync(guild=guild)
            synced_guilds += 1
        except Exception as e:
            print(f"[WARN] Не удалось синхронизировать слеш-команды на сервере {guild.name} ({guild.id}): {e}")
    return len(synced_global), synced_guilds


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


async def translate_prompt_to_english(prompt: str, ref_count: int) -> str:
    """Переводит русский промпт в точный английский промпт для Flux (если он содержит кириллицу)."""
    if not re.search(r"[а-яА-ЯёЁ]", prompt):
        return prompt

    ref_hint = (
        f" The user attached {ref_count} reference image(s) (image 1, image 2, etc.) — preserve references to them accurately."
        if ref_count > 0
        else ""
    )
    sys_msg = (
        "You are an expert prompt translator for FLUX.2 image generation. "
        "Translate the user's image prompt from Russian into clear, detailed, natural English "
        "preserving every object, style, action, and detail without censoring or lecturing."
        f"{ref_hint} Return ONLY the translated English prompt text, nothing else."
    )

    try:
        fast_client = make_client(active_base_url, timeout_sec=15.0)
        resp = await fast_client.chat.completions.create(
            model="logfare/auto",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        translated = clean_thinking(resp.choices[0].message.content or "").strip("\"'` \n")
        if translated and len(translated) >= 5:
            print(f"[INFO] Промпт переведён для Flux: {translated}")
            return translated
    except Exception as e:
        print(f"[WARN] Автоперевод промпта пропущен ({e}), используем оригинал.")

    return prompt


async def download_image_bytes(url: str) -> bytes:
    """Скачивает сгенерированное изображение по URL."""
    async with aiohttp.ClientSession(headers=DEFAULT_HEADERS) as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            resp.raise_for_status()
            return await resp.read()


async def extract_image_from_api_json(data: dict) -> bytes | None:
    """Извлекает байты изображения из JSON-ответа OpenAI/Flux/Chat форматов."""
    items = data.get("data")
    if isinstance(items, list) and items:
        first = items[0]
        if isinstance(first, dict):
            if first.get("b64_json"):
                return base64.b64decode(first["b64_json"])
            if first.get("url"):
                return await download_image_bytes(first["url"])

    for key in ("b64_json", "image_base64", "base64"):
        if isinstance(data.get(key), str) and data[key]:
            b64_str = data[key].split(",")[-1]
            return base64.b64decode(b64_str)

    for key in ("url", "image_url", "image"):
        val = data.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return await download_image_bytes(val)

    output = data.get("output")
    if isinstance(output, list) and output and isinstance(output[0], str) and output[0].startswith("http"):
        return await download_image_bytes(output[0])
    if isinstance(output, str) and output.startswith("http"):
        return await download_image_bytes(output)

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {})
        images = msg.get("images")
        if isinstance(images, list) and images:
            img_obj = images[0]
            if isinstance(img_obj, dict):
                u = img_obj.get("image_url", {}).get("url") or img_obj.get("url")
                if u:
                    if u.startswith("data:image"):
                        return base64.b64decode(u.split(",", 1)[1])
                    if u.startswith("http"):
                        return await download_image_bytes(u)

        content = msg.get("content")
        if isinstance(content, str):
            data_uri_match = re.search(r"data:image/[a-zA-Z0-9+.-]+;base64,([A-Za-z0-9+/=]+)", content)
            if data_uri_match:
                return base64.b64decode(data_uri_match.group(1))
            url_match = re.search(r"https?://[^\s\)\"']+\.(?:png|jpg|jpeg|webp)(?:\?[^\s\)\"']*)?", content, re.IGNORECASE)
            if not url_match:
                url_match = re.search(r"!\[.*?\]\((https?://[^\s\)]+)\)", content)
            if url_match:
                target_url = url_match.group(1) if url_match.lastindex else url_match.group(0)
                return await download_image_bytes(target_url)

    return None


async def generate_flux2_image(
    prompt: str,
    ref_attachments: list[discord.Attachment],
    image_model: str = DEFAULT_IMAGE_MODEL,
) -> bytes:
    """Генерирует изображение через выбранную модель Flux с передачей промпта и Base64-референсов."""
    auth_headers = {
        **DEFAULT_HEADERS,
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data_uris: list[str] = []
    raw_files: list[tuple[str, bytes, str]] = []
    for idx, att in enumerate(ref_attachments, start=1):
        raw_bytes = await att.read()
        mime = att.content_type if (att.content_type and att.content_type.startswith("image/")) else "image/png"
        filename = att.filename or f"ref_{idx}.png"
        raw_files.append((filename, raw_bytes, mime))
        b64 = base64.b64encode(raw_bytes).decode("ascii")
        data_uris.append(f"data:{mime};base64,{b64}")

    timeout = aiohttp.ClientTimeout(total=120)
    last_error = "Неизвестная ошибка при генерации."

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 1. Если есть референсы — сначала пробуем multipart /images/edits
        if raw_files:
            form = aiohttp.FormData()
            form.add_field("model", image_model)
            form.add_field("prompt", prompt)
            for fname, fbytes, fmime in raw_files:
                field_name = "image" if len(raw_files) == 1 else "image[]"
                form.add_field(field_name, fbytes, filename=fname, content_type=fmime)

            try:
                async with session.post(
                    f"{active_base_url}/images/edits",
                    headers={**DEFAULT_HEADERS, "Authorization": f"Bearer {API_KEY}"},
                    data=form,
                ) as resp:
                    if resp.status == 200:
                        res_json = await resp.json()
                        img_bytes = await extract_image_from_api_json(res_json)
                        if img_bytes:
                            print(f"[INFO] {image_model}: успешно через POST /images/edits (multipart, refs={len(raw_files)})")
                            return img_bytes
                    else:
                        err_txt = await resp.text()
                        last_error = f"/images/edits HTTP {resp.status}: {err_txt[:200]}"
                        print(f"[WARN] {last_error}")
            except Exception as e:
                last_error = f"/images/edits exception: {e}"
                print(f"[WARN] {last_error}")

        # 2. JSON-запрос в /images/generations с передачей prompt и Base64-референсов
        gen_payload: dict = {
            "model": image_model,
            "prompt": prompt,
            "n": 1,
        }
        if data_uris:
            gen_payload["image"] = data_uris[0]
            gen_payload["images"] = data_uris
            gen_payload["image_urls"] = data_uris
            gen_payload["reference_images"] = data_uris

        try:
            async with session.post(
                f"{active_base_url}/images/generations",
                headers=auth_headers,
                json=gen_payload,
            ) as resp:
                if resp.status == 200:
                    res_json = await resp.json()
                    img_bytes = await extract_image_from_api_json(res_json)
                    if img_bytes:
                        print(f"[INFO] {image_model}: успешно через POST /images/generations (JSON, refs={len(data_uris)})")
                        return img_bytes
                    last_error = f"API вернул 200, но картинка не найдена в ответе: {str(res_json)[:200]}"
                else:
                    err_txt = await resp.text()
                    last_error = f"/images/generations HTTP {resp.status}: {err_txt[:200]}"
                    print(f"[WARN] {last_error}")
        except Exception as e:
            last_error = f"/images/generations exception: {e}"
            print(f"[WARN] {last_error}")

        # 3. Мультимодальный резерв через /chat/completions
        if data_uris:
            content_parts: list[dict] = [{"type": "text", "text": prompt}]
            for uri in data_uris:
                content_parts.append({"type": "image_url", "image_url": {"url": uri}})

            chat_payload = {
                "model": image_model,
                "messages": [{"role": "user", "content": content_parts}],
            }
            try:
                async with session.post(
                    f"{active_base_url}/chat/completions",
                    headers=auth_headers,
                    json=chat_payload,
                ) as resp:
                    if resp.status == 200:
                        res_json = await resp.json()
                        img_bytes = await extract_image_from_api_json(res_json)
                        if img_bytes:
                            print(f"[INFO] {image_model}: успешно через POST /chat/completions (multimodal, refs={len(data_uris)})")
                            return img_bytes
            except Exception as e:
                print(f"[WARN] /chat/completions fallback exception: {e}")

    raise RuntimeError(last_error)


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

    try:
        cmd_count, guild_count = await sync_all_slash_commands()
        print(f"[INFO] Синхронизировано слеш-команд: {cmd_count} (мгновенно для {guild_count} серверов)")
    except Exception as e:
        print(f"[ERROR] Не удалось синхронизировать слеш-команды: {e}")

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


# ==================== СЛЕШ-КОМАНДА /flux2 ====================

@bot.tree.command(name="flux2", description="Сгенерировать изображение через FLUX.2 (с автопереводом промпта)")
@app_commands.describe(
    prompt="Описание картинки (минимум 10 символов, можно на русском)",
    image1="Референс-изображение 1 (необязательно)",
    image2="Референс-изображение 2 (необязательно)",
    image3="Референс-изображение 3 (необязательно)",
    model="Выбор модели FLUX (по умолчанию flux-2-dev)",
)
@app_commands.choices(
    model=[
        app_commands.Choice(name="flux-2-dev (Макс. качество и референсы)", value="flux-2-dev"),
        app_commands.Choice(name="flux-2-klein-9b (Быстро + хорошее качество)", value="flux-2-klein-9b"),
        app_commands.Choice(name="flux-2-klein-4b (Максимально быстро)", value="flux-2-klein-4b"),
    ]
)
async def flux2_command(
    interaction: discord.Interaction,
    prompt: str,
    image1: discord.Attachment | None = None,
    image2: discord.Attachment | None = None,
    image3: discord.Attachment | None = None,
    model: app_commands.Choice[str] | None = None,
):
    clean_prompt = prompt.strip()
    if len(clean_prompt) < 10:
        await interaction.response.send_message(
            "❌ Промпт слишком короткий! Напиши минимум **10 символов**.",
            ephemeral=True,
        )
        return

    refs: list[discord.Attachment] = []
    for idx, att in enumerate((image1, image2, image3), start=1):
        if att is not None:
            if att.content_type and not att.content_type.startswith("image/"):
                await interaction.response.send_message(
                    f"❌ Вложение `image{idx}` должно быть изображением (PNG, JPG, WEBP).",
                    ephemeral=True,
                )
                return
            refs.append(att)

    selected_model = model.value if model else DEFAULT_IMAGE_MODEL
    await interaction.response.defer(thinking=True)

    try:
        en_prompt = await translate_prompt_to_english(clean_prompt, len(refs))
        image_bytes = await generate_flux2_image(en_prompt, refs, image_model=selected_model)
        file = discord.File(fp=io.BytesIO(image_bytes), filename="flux2_result.png")

        refs_info = f" | Референсов: {len(refs)}" if refs else ""
        short_orig = clean_prompt if len(clean_prompt) <= 220 else f"{clean_prompt[:217]}..."
        caption = f"🎨 **Модель:** `{selected_model}`{refs_info}\n**Промпт:** `{short_orig}`"
        if en_prompt != clean_prompt:
            short_en = en_prompt if len(en_prompt) <= 220 else f"{en_prompt[:217]}..."
            caption += f"\n**EN-промпт:** `{short_en}`"

        await interaction.followup.send(content=caption, file=file)
    except (asyncio.TimeoutError, APITimeoutError):
        await interaction.followup.send("⏳ Сервер генерации изображений не успел ответить по таймауту. Попробуй модель `flux-2-klein-9b` или повтори запрос.")
    except Exception as e:
        await interaction.followup.send(f"❌ Ошибка при генерации изображения: `{e}`")


# ==================== ТЕКСТОВЫЕ КОМАНДЫ ====================

@bot.command(name="sync")
@is_admin_or_owner()
async def sync_commands_cmd(ctx: commands.Context):
    """Принудительно обновляет слеш-команды на сервере."""
    try:
        cmd_count, guild_count = await sync_all_slash_commands()
        await ctx.reply(
            f"✅ Слеш-команды обновлены (`{cmd_count}` шт. на `{guild_count}` серверах). Если не появились сразу — нажми `Ctrl + R` в Discord.",
            mention_author=False,
        )
    except Exception as e:
        await ctx.reply(f"❌ Ошибка синхронизации слеш-команд: `{e}`", mention_author=False)


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
            + f"**Текстовая модель:** `{active_model}`\n"
            + f"**Модель картинок:** `{DEFAULT_IMAGE_MODEL}`\n"
            + f"**Найденные модели:** `{models_preview}`"
        )
        await ctx.reply(report, mention_author=False)


@bot.command(name="commands")
async def show_commands(ctx: commands.Context):
    """Отправляет список всех команд бота."""
    commands_text = (
        "**Список команд бота:**\n\n"
        "**Общение и генерация (для всех):**\n"
        f"• **Упоминание ({bot.user.mention} текст)** или **Reply** на сообщение бота — пообщаться с нейросетью (`{PREFERRED_MODEL}`).\n"
        "• **Личные сообщения (ЛС)** — пиши любой текст напрямую без упоминаний.\n"
        "• **`/flux2 prompt:<текст> [image1] [image2] [image3] [model]`** — сгенерировать картинку через FLUX.2 (минимум 10 символов, до 3 референсов, автоперевод на EN).\n"
        "• **`!commands`** — показать этот список команд.\n\n"
        "**Управление (только для создателя и админов):**\n"
        "• **`!clear`** — очистить память диалога бота в текущем канале.\n"
        "• **`!prompt`** — посмотреть текущий системный промпт.\n"
        "• **`!prompt <текст>`** — задать новый характер/промпт нейросети (сбрасывает историю).\n"
        "• **`!prompt reset`** — вернуть стандартный промпт по умолчанию.\n"
        "• **`!sync`** — принудительно обновить слеш-команды на сервере.\n"
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
