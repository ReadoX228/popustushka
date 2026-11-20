import os
import re
from collections import defaultdict, deque
import discord
from discord.ext import commands
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://logfare.ai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen-3.8-27b")

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

ai_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)
channel_history = defaultdict(lambda: deque(maxlen=10))


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


@bot.event
async def on_ready():
    print(f"Бот запущен как {bot.user} (ID: {bot.user.id})")
    print(f"Модель: {MODEL_NAME} | Эндпоинт: {BASE_URL}")


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CheckFailure):
        await ctx.reply("Эта команда доступна только создателю сервера и администраторам.", mention_author=False)
    else:
        raise error


@bot.command(name="clear")
@is_admin_or_owner()
async def clear_history(ctx: commands.Context):
    """Очищает память бота в текущем канале."""
    channel_history[ctx.channel.id].clear()
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

    history = channel_history[message.channel.id]
    history.append({"role": "user", "content": f"{message.author.display_name}: {clean_text}"})

    messages_payload = [{"role": "system", "content": current_system_prompt}, *history]

    async with message.channel.typing():
        try:
            reply_text = ""
            for attempt in range(2):
                response = await ai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages_payload,
                    temperature=0.9 + (attempt * 0.15),
                    max_tokens=1500,
                )
                raw_reply = response.choices[0].message.content or ""
                reply_text = clean_thinking(raw_reply)

                if not any(phrase in reply_text.lower() for phrase in REFUSAL_PHRASES):
                    break

            if not reply_text:
                reply_text = "Нейросеть промолчала. Попробуй спросить ещё раз."

            history.append({"role": "assistant", "content": reply_text})

            for chunk in split_message(reply_text):
                await message.reply(chunk, mention_author=False)

        except Exception as e:
            await message.reply(f"Ошибка при обращении к API: `{e}`")


if __name__ == "__main__":
    if not DISCORD_TOKEN or not API_KEY:
        raise ValueError("Не заданы переменные окружения DISCORD_TOKEN и API_KEY!")
    bot.run(DISCORD_TOKEN)
