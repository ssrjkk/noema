# Noema — Engineering Mind

> Фреймворк генерации мощных технических решений на любом стеке, который **сам проверяет то, что сгенерировал** — и учится чинить себя без человека.

Noema — это не ещё один генератор кода. Это **инженерный разум**: система, которая не просто выдаёт решение, а проверяет его формальной верификацией, выполняет в изолированной песочнице, ведёт аудируемый трейл каждой мысли и сама открывает pull-request'ы на свои же инциденты.

<p align="center">
<img src="https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white" alt="Python 3.11+ (recommended 3.12)">
<img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT">
<img src="https://img.shields.io/badge/tests-1659-blue?label=unit%20tests" alt="1659 unit tests">
<img src="https://img.shields.io/badge/version-1.4.0-brightgreen" alt="v1.4.0">
<img src="https://img.shields.io/badge/domains-22-orange" alt="22 domain modules">
</p>

---

## ⚡ Quick Start — 30 секунд

Не хочется читать? Запусти это:

```bash
git clone https://github.com/ssrjkk/noema && cd noema
pip install -e .
python demo.py      # запускает 12 живых демо: security, quality, DB schema, Docker, Terraform, RBAC, GraphQL и т.д.
```

Если `python demo.py` вывел `=== ALL DEMOS COMPLETE ===` — всё работает. Дальше:

```bash
noema --help                                       # или: python -m noema --help
noema think "Auth service on FastAPI" --tags "python,fastapi,auth"
```

---

## Проблема, которую решает Noema

Большие языковые модели хороши в одном — в генерации. Всё остальное они делают плохо:

- **Говорят уверенно, но ошибаются.** Решение выглядит идеально, а по факту — синтаксическая ошибка, несуществующий импорт, дыра в безопасности.
- **Никто не проверяет результат.** Сгенерированный код летит в репозиторий без единой формальной проверки.
- **Нельзя переиграть.** Если результат плохой — непонятно, где именно система ошиблась и почему.
- **Память короткая.** Каждая задача решается с нуля: прошлые решения, ошибки и инциденты не учитываются.
- **Нет экономики.** Токены жгутся без контроля, стоимости каждой сгенерированной строки никто не знает.
- **Ручное сопровождение.** Исправление багов, ревью, гейты — всё руками.

Noema превращает генерацию из одноразовой ставки в **инженерный процесс с чекпоинтами, верификацией и аудитом**.

## Идея в одной строке

> Нейронная сторона **предлагает**, символьная — **распоряжается**, и весь обмен между ними записывается в аудируемый артефакт.

LLM генерирует гипотезу. Символьный движок (Z3) проверяет её против формального контракта. AST-анализатор проверяет код до запуска. Песочница выполняет его под ограничениями. Если что-то не сходится — система рефайнится, а не выдаёт непроверенный мусор. **Ни один непроверенный результат не принимается.**

## Что умеет Noema

| Возможность | Что даёт |
|---|---|
| **Генерация решений** | `noema think "Real-time Chat App"` — полное решение: архитектура, стек, код, оптимизации, безопасность |
| **Формальная верификация (Z3)** | Каждая гипотеза проверяется против извлечённого символического контракта. Solver недоступен → решение не принимается (fail-closed) |
| **Статический анализ до запуска** | Чистый AST-проход: синтаксис, гигиена импортов, неопределённые имена — вердикт ещё до запуска непроверенного кода |
| **Изолированная песочница** | Docker-запуск без сети, с лимитами CPU/памяти/времени; статический вердикт короткозамыкает запуск |
| **Аудит рассуждений** | Каждый шаг мысли, вердикт Z3 и AST фиксируются в reasoning-trace — старый вердикт перепроверяется детерминированно, без LLM; трейсы персистентны на диске (`trace_dir`, по умолчанию `.noema/traces`), OTLP-экспорт зарезервирован (`NOEMA_OBS__TRACING_ENDPOINT`)
| **Автономность** | Инцидент из Sentry/webhook → фикс → ветка → PR с прошедшей валидацией. Merge-гейт по judge-оценке в CI |
| **Самоэволюция под доказательствами** | Мутации применяются только когда проходят тесты (`evolution_test_before_apply`) |
| **22 доменных модуля** | auth, database, gateway, graphql, ml_ops, mobile, terraform, websocket и другие — работают автономно и вместе |
| **Ядра и агенты** | Архитектор, кодер, security, devops, DBA, AI-engineer; пайплайны fullstack/quick/security/arch-review |
| **Память и знания** | Эпизодическая, семантическая и процедурная память + доменная база знаний |
| **Экономика токенов** | Каждый вызов LLM трассируется, атрибутируется и конвертируется в денежную оценку; бюджеты и circuit breakers |
| **Воспроизводимые бенчмарки** | Одна матрица задач по провайдерам/моделям → `results.json` + CSV-сводки, в т.ч. пофайловое разложение токенов/стоимости каждой сгенерированной строки |
| **API для продакшена** | FastAPI: rate limiting, API-ключи, квоты по тенантам, request-timeout guard (504 + exempt-пути), RFC 7807, метрики Prometheus, streaming |
| **Grid-федерация** | `noema grid federate`: подзадачи делегируются пирам по gRPC с circuit breaker и ретраями, при недоступности пиров — локальный фолбэк; вклад каждой ноды пишется в аудируемый ledger |
| **Дашборд грида** | `GET /grid` и `noema grid status`: живое состояние флота воркеров — латентность, токены, ошибки по каждой ноде + итоги по кластеру |

## Архитектура в одном взгляде

```
            ┌──────────────────────────────────────────────────┐
            │                   NoemaEngine                     │
            │   ChainOfThought (DAG)  ·  NeuroSymbolicEngine     │
            └───────┬───────────────────────┬───────────────────┘
                    │ propose               │ verify
        ┌───────────▼──────────┐   ┌────────▼────────────────────────┐
        │ NeuralInterface (LLM)│   │ SymbolicEngine (Z3) · static.py │
        └───────────┬──────────┘   └────────┬────────────────────────┘
                    │ hypothesis            │ verdict (fail-closed)
                    └───────► refine loop ◄─┘
                        ┌─────────────┼──────────────┐
                        │ trace       │ sandbox       │ memory / knowledge
                        │ replay      │ static+run    │ (episodic, domain)
```

Цикл рассуждения ограничен `max_refinement_attempts`: задача → символьный граф → гипотеза → верификация → рефайн → успех или исчерпание попыток.

## Установка

```bash
# Минимальный набор
pip install -e .

# Полный (верификация Z3, векторный поиск, провайдеры LLM)
pip install -e ".[full]"

# Всё, включая dev-инструменты, БД, gRPC, vault
pip install -e ".[dev,full,db,grpc,vault]"
```

Python **3.11+** (рекомендуется 3.12 — основная разработка и CI идут на 3.12).
Провайдеры LLM: `openai`, `anthropic`, `ollama` + встроенный **fallback-провайдер** (работает без ключей, для демо и CI — просто ставь и запускай).
Управление через env: `NOEMA_LLM__PROVIDER=openai`, `NOEMA_LLM__MODEL=...`, `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.

### Проверка установки

```bash
# Убедиться, что всё поставилось:
noema --help                  # вариант 1 — CLI скрипт
python -m noema --help        # вариант 2 — модуль Python (надёжнее на Windows)

# Запустить 12 живых демо без единого API-ключа:
python demo.py
# Должно вывестись === ALL DEMOS COMPLETE ===
```

## Production Setup — важные настройки по умолчанию

**По умолчанию Noema настроен для разработки и демо, а не для продакшена.** Перед деплоем обязательно настройте:

### Обязательные настройки

1. **LLM-провайдер** — по умолчанию `fallback` (шаблонные ответы, не LLM)
   ```bash
   export NOEMA_LLM__PROVIDER=openai  # или anthropic, ollama
   export OPENAI_API_KEY=sk-...
   ```

2. **API-ключ** — по умолчанию auth выключен (пустой ключ)
   ```bash
   export NOEMA_API__API_KEY=your-secret-key-here
   ```

3. **Neurosymbolic verification** — по умолчанию выключена (`NOEMA_NS__ENABLED=false`)
   ```bash
   export NOEMA_NS__ENABLED=true
   pip install -e ".[full]"  # нужен Z3
   ```

4. **Webhook secret** — по умолчанию HMAC-верификация выключена
   ```bash
   export NOEMA_API__WEBHOOK_SECRET=your-webhook-secret
   ```

5. **Database** — для персистентности, аудита и billing
   ```bash
   export NOEMA_DB__URL=postgresql+asyncpg://user:pass@localhost/noema
   pip install -e ".[db]"
   ```

### Опциональные, но рекомендуемые

- **Redis** — для rate limiting в multi-worker deployments
- **Sentry** — для мониторинга ошибок (`NOEMA_OBS__SENTRY_DSN`)
- **Trusted proxies** — если за reverse proxy (`NOEMA_API__TRUSTED_PROXIES`)

### Fail-closed поведение

Noema следует принципу **fail-closed**: если компонент недоступен, система отказывает безопасно, а не фабрикуют ответ:
- LLM недоступен → `RuntimeError`, а не шаблонный ответ
- Webhook secret пустой → запрос отклоняется
- Z3 solver недоступен → верификация не проходит

## Быстрый старт

### CLI

```bash
# Везде, где написано `noema ...`, можно писать `python -m noema ...`
# (полезно на Windows, где CLI-скрипт иногда не попадает в PATH)

noema think "Real-time Chat App" --tags "python,websocket,redis" --complexity complex --output full

# Экспорт в структуру проекта на диск
noema think "Auth service" --scaffold --scaffold-dir ./out

# Пайплайны ядер
noema pipeline fullstack --title "My Project"
noema pipeline security --title "API audit"

# Знания, память, граф
noema knowledge search -q "database optimization"
noema memory stats
noema graph suggest --tags "python,fastapi,redis"

# Автономия и самоэволюция
noema evolve
noema agents
noema modules list
```

### API-сервер

```bash
noema serve
# или
python -m noema serve
# http://localhost:8000
```

Все эндпоинты доступны и в корне (`/think`), и под версионированным префиксом `/api/v1/think`.

| Метод | Эндпоинт | Назначение |
|---|---|---|
| POST | `/think` | Полное решение по задаче |
| POST | `/think/detail` | Решение + ход мыслей |
| POST | `/think/stream` | SSE-поток шагов рассуждения |
| DELETE | `/think/{task_id}` | Отмена задачи |
| POST | `/tasks/enqueue` | Асинхронная задача через Redis/arq |
| POST | `/experiments` | Бенчмарк-матрица как сервис |
| POST | `/webhooks/incident` | Инцидент → автономный фикс → PR |
| GET | `/health`, `/ready`, `/diagnostics`, `/features` | Ops |
| GET | `/kernels`, `/agents`, `/knowledge/stats`, `/knowledge/search` | Разведка |

Пример:

```bash
curl -X POST http://localhost:8000/think \
  -H "Content-Type: application/json" \
  -d '{"title": "Event bus on RabbitMQ", "complexity": "complex", "tags": ["python", "rabbitmq"]}'
```

## Автономность: система чинит себя сама

1. **Инцидент** — Sentry-алерт или `POST /webhooks/incident` нормализуется в `Incident`.
2. **Фикс** — задача прогоняется через `NoemaEngine`, результат валидируется тестами (`validate_solution(run_tests=True)`). Без `all_valid` PR не создаётся.
3. **PR** — httpx-клиент GitHub (без PyGithub) открывает ветку и pull-request.
4. **Merge-гейт** — CI-джоба (`gate` в `.github/workflows/ci.yml`) блокирует мёрж, если `judge_score` ниже порога или песочница упала.
5. **Эволюция** — кандидаты промптов/мутаций применяются только при зелёных тестах.

Настроить: `NOEMA_AUTONOMY__GITHUB_TOKEN`, `NOEMA_AUTONOMY__GITHUB_REPO`, `NOEMA_AUTONOMY__GITHUB_BASE_BRANCH`.

## Эксперименты и бенчмарки

Воспроизводимый раннер: одна и та же матрица задач по провайдерам и моделям, сбор wall-time, токенов, judge-оценки, стоимости и (опционально) валидации песочницы в `results/`.

```bash
# Демо без ключей (fallback-провайдер)
python -m noema.experiments.runner experiments/experiments.yaml --out results

# Реальные модели: укажите провайдера в experiments.yaml и экспортируйте ключ
# results/<experiment>/<run_id>/results.json — по записи на (task, provider, model, repetition)
# results/<experiment>/<run_id>/runs.csv      — то же в CSV
# results/<experiment>/<run_id>/summary.csv   — агрегаты по (provider, model)
```

CI-джоба `.github/workflows/experiments.yml` гоняет smoke-бенчмарк ночью на fallback-провайдере и заливает артефакты. Тот же раннер доступен как сервис: `POST /experiments`.

## Как мы проверяем то, что строим

- **1659 unit-тестов** (pytest + hypothesis + pytest-benchmark), включая гейты: автономия, reasoning-trace round-trip, статический вердикт, извлечение контрактов из требований, доменные знания.
- **Ruff + mypy** в CI, **pre-commit** хуки.
- **Проверка кодировки и mojibake-гейт** — сломанные юникод-строки не проходят CI.
- **Security-сканеры** (bandit, safety, pip-audit) в пайплайне.

## Структура проекта

```
noema/
  autonomy/        # инциденты → фиксы → PR
  neurosymbolic/   # Z3-верификация + AST-анализ в одном пайплайне
  sandbox/         # Docker-песочница + статические проверки до запуска
  tracing/         # reasoning-trace: перепроверка вердиктов без LLM
  experiments/     # воспроизводимый бенчмарк-раннер + merge-gate
  knowledge/       # база знаний + доменные модули (22 шт)
  memory/          # эпизодическая / семантическая / процедурная память
  api/             # FastAPI: think, webhooks, experiments, admin, rate limits
  workers/         # arq-воркеры, иерархия задач, пул
  modules/         # pluggable доменные модули
  kernels/ agents/ # специализированные ядра и агенты
  llm/             # провайдеры: openai, anthropic, ollama, fallback
  billing/ budget/ # экономика токенов, квоты, бюджеты
  security/        # валидация, схемы, тенант-изоляция
  grpc/            # gRPC сервер/клиент + protos
  observability/   # Prometheus-метрики, Sentry
  vault/ audit/    # секреты, аудит-трейл
```

## Документация

- **Whitepaper** — видение и дизайн-принципы: почему аудируемая нейросимвольная композиция важна → [docs/WHITEPAPER.md](docs/WHITEPAPER.md)
- **Roadmap** — три фазы: Architect → Autopoietic Enterprise → Global Noema Grid → [docs/ROADMAP.md](docs/ROADMAP.md)
- **Configuration** — все env-переменные и YAML → [docs/configuration.md](docs/configuration.md)
- **Deployment** — Docker/Compose/K8s/Helm/Terraform → [docs/deployment.md](docs/deployment.md)
- **Getting started / API examples** — [docs/getting-started.md](docs/getting-started.md), [docs/api-examples.md](docs/api-examples.md)
- **Operations** — production checklist, мониторинг, перформанс, troubleshooting → [docs/production-checklist.md](docs/production-checklist.md), [docs/monitoring-setup.md](docs/monitoring-setup.md), [docs/performance-tuning.md](docs/performance-tuning.md), [docs/troubleshooting.md](docs/troubleshooting.md)
- MkDocs-сайт с API-справочником — `mkdocs serve`

## Статус и дорога

- **Phase 1 — The Architect (сейчас):** генерация + верификация + песочница + бенчмарки + доменные знания + reasoning-trace. Готово.
- **Phase 2 — The Autopoietic Enterprise:** инцидент → PR, merge-гейт, эволюция с авто-апплаем, бенчмарк-сервис. Почти готово (остался точный учёт стоимости на строку кода).
- **Phase 3 — Global Noema Grid:** многоузловой пул, gRPC-федерация, token/ledger-экономика, живой дашборд. В работе.

## Лицензия

MIT. Открытая разработка — идеи, инциденты и PR'ы приветствуются.
