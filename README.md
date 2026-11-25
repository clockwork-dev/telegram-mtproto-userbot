# Autonomous Telegram Group Userbot (`telegram-group-bot`)

Автономный Telegram userbot-агент для групповых чатов на базе официального **Google Antigravity Python SDK** (`google-antigravity`) и CLI-адаптера `agy`, с изоляцией отдельного аккаунта с подпиской, мультимодальным пайплайном, долговременной памятью **SQLite + FTS5**, горячей перезагрузкой конфигов (`bot.yaml`, `friends.yaml`, `system_prompt.md`) и проактивным генератором шуток/вбросов.

---

## Основные возможности

1. **Движок Antigravity SDK + изоляция аккаунта (`src/bot/antigravity/`)**:
   - Прямая интеграция с `google-antigravity` (`Agent`, `LocalAgentConfig`, `CapabilitiesConfig(enabled_tools=[])`) и автоматический fallback на пул процессов `agy --output-format stream-json`.
   - Полная изоляция отдельного аккаунта Antigravity в выделенной директории (`data/antigravity`, права `0700` на папки и `0600` на OAuth-токены).
   - Принудительное файловое хранение токена без перезаписи основного системного keyring через `SSH_CONNECTION="127.0.0.1 0 127.0.0.1 22"`.
   - Атомарная синхронизация обновлённых токенов (`os.replace` + `0600`), подавление глобальной персоны (`SOUL.md`) и **Zero Secret Exfiltration** (секреты и загрузчики `LD_PRELOAD`/`PYTHONPATH` вычищаются из окружения дочерних процессов и никогда не логируются).
   - Перехват и блокировка несанкционированных нативных вызовов инструментов (`AntigravityNativeToolAbort`), классификация ошибок (`429`, переполнение контекста, ошибки авторизации/биллинга) и экспоненциальный backoff с jitter.

2. **MTProto Userbot и мультимодальность (`src/bot/telegram/`, `src/bot/media/`)**:
   - Работает через `Telethon` под выделенным Telegram-аккаунтом пользователя (`0600` на `.session`).
   - Строгий белый список чатов (`allowed_chat_ids`), защита от самоответов (`event.out` и `sender_id == bot_id`), имитация чтения и набора текста (`typing`) с естественной задержкой, ответы цитатами и постановка реакций.
   - Умная эвристика участия (`ReplyHeuristics`): мгновенная реакция на упоминания/реплаи и вероятностное вклинивание в диалог с учётом антиспам-кулдаунов и `quiet_hours`.
   - Понимание мультимодальных вложений с гарантированной очисткой временных файлов (`managed_media`):
     - Фотографии и скриншоты (`JPEG`/`PNG`), конвертация стикеров `WebP -> PNG` через `Pillow`.
     - Документы: `PDF` (`pypdf`), `DOCX` (`python-docx`), `TXT`/`CSV`/`JSON`/исходный код с лимитом размера (`max_doc_size_mb`).
     - Голосовые сообщения (`.ogg`) и видео-кружочки (`.mp4`): транскодинг аудио в `16kHz mono WAV` и извлечение ключевого кадра через `ffmpeg`.

3. **Долговременная память и Hot-Reload конфигураций (`src/bot/memory/`, `src/bot/config/`)**:
   - Краткосрочная буферная история сообщений в SQLite (`messages`).
   - Долговременная память на базе **SQLite FTS5** (`memory_facts` + `memory_fts`) с триггерами синхронизации и русским префиксным стеммером для поиска по склонениям (`шаурму` ↔ `шаурма` ↔ `шаурмами`).
   - Фоновая компактификация (`CompactionEngine`): суммаризация старых сообщений и автоматическое извлечение долгосрочных фактов.
   - Асинхронный `mtime` hot-reload без перезапуска бота для:
     - `config/bot.yaml` — параметры Telegram, поведения, памяти, медиа и таймеров.
     - `config/system_prompt.md` — системный промпт (персона, тон, правила).
     - `config/friends.yaml` — досье друзей (`telegram_user_id -> имя, ники, био, роль, стиль общения, разрешённые темы для подколов и запретные чувствительные темы`).

4. **Проактивное поведение и генератор шуток/вбросов (`src/bot/proactive/`)**:
   - Фоновый цикл с рандомизацией (`jitter`), триггером долгой тишины (`silence_trigger_hours`), антиспам-кулдауном (`cooldown_minutes`) и поддержкой тихих часов через полночь (`23:00` – `08:00`).
   - Генерация дружеских подколов и абсурдных «фейк-ньюс» по профилям друзей и памяти чата со строгой фильтрацией `sensitive_topics` и `forbidden_topics`.

---

## Быстрый старт и установка

### 1. Требования
- Linux (Debian/Ubuntu), Python `>= 3.11`, `ffmpeg`.
- Установленный бинарник Antigravity CLI (`agy >= 1.5.8`).
- `api_id` и `api_hash` приложения Telegram (получить на [my.telegram.org](https://my.telegram.org)).

### 2. Создание виртуального окружения
```bash
cd /home/jeff/telegram-group-bot
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e ".[dev]"
```

### 3. Настройка конфигурационных файлов
Скопируйте шаблоны и укажите свои параметры:
```bash
cp .env.example .env
cp config/bot.example.yaml config/bot.yaml
cp config/friends.example.yaml config/friends.yaml
chmod 600 .env
```
- В `config/bot.yaml` пропишите `api_id`, `api_hash` и список разрешённых групп `allowed_chat_ids` (например, `-1001234567890`).
- В `config/friends.yaml` укажите Telegram User ID ваших друзей, их имена, ники, био, `banter_topics` (разрешённые темы для шуток) и `sensitive_topics` (запретные темы).
- В `config/system_prompt.md` настройте характер и правила поведения бота (изменения применяются на лету без перезапуска).

---

## Первичная авторизация аккаунтов

### Шаг 1. Авторизация изолированного аккаунта Antigravity (`auth-agy`)
Выполните вход в отдельный аккаунт с подпиской Antigravity. Токен сохранится локально в `data/antigravity` с правами `0600`, не затрагивая основной аккаунт системы:
```bash
.venv/bin/telegram-group-bot auth-agy --data-dir data/antigravity
```

### Шаг 2. Авторизация Telegram-аккаунта бота (`auth-telegram`)
Выполните интерактивный вход по номеру телефона, коду из Telegram и (при наличии) облачному паролю 2FA. Сессия сохранится в `data/session/userbot.session` с правами `0600`:
```bash
.venv/bin/telegram-group-bot auth-telegram --config config/bot.yaml
```

---

## Проверка и запуск

### Dry-Run Smoke Test (проверка всего пайплайна без выхода в сеть)
```bash
.venv/bin/telegram-group-bot run --config config/bot.yaml --dry-run
```

### Ручной запуск демона
```bash
.venv/bin/telegram-group-bot run --config config/bot.yaml --log-level INFO
```

### Запуск через `systemd`
```bash
sudo cp systemd/telegram-userbot.service /etc/systemd/system/telegram-userbot.service
sudo systemctl daemon-reload
sudo systemctl enable --now telegram-userbot.service
sudo systemctl status telegram-userbot.service
```

### Автоматический деплой и проверка на удалённом сервере
```bash
chmod +x deploy/deploy_remote.sh
./deploy/deploy_remote.sh root@192.168.1.164 /opt/telegram-group-bot
```

---

## Тестирование и контроль качества
```bash
.venv/bin/pytest -v
.venv/bin/ruff check .
```
