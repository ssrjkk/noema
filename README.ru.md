# Noema

**Инженерный разум, который проверяет то, что генерирует.**

Noema — это не очередной генератор кода. Это нейросимвольная система: LLM предлагает решения, Z3 проверяет их против формальных контрактов, песочница валидирует в изоляции, а каждый шаг рассуждения записывается в аудируемый трейл — и всё это умеет чинить свои же инциденты от начала до конца.

> [!NOTE]
> **Фаза 1 — Архитектор** готова к продакшену. Фаза 2 (автономное самоисцеление) и Фаза 3 (федерация мульти-нод грида) в активной разработке.

---

<p align="center">
  <a href="#быстрый-старт">Быстрый старт</a> •
  <a href="#возможности">Возможности</a> •
  <a href="#архитектура">Архитектура</a> •
  <a href="#установка">Установка</a> •
  <a href="#использование">Использование</a> •
  <a href="#документация">Документация</a> •
  <a href="../README.md">English</a>
</p>

---

## Быстрый старт

```bash
git clone https://github.com/ssrjkk/noema && cd noema
pip install -e .
python demo.py        # 12 живых демо — API-ключи не нужны
```

Если видишь `=== ALL DEMOS COMPLETE ===` — всё работает. Дальше:

```bash
noema think "Auth service on FastAPI" --tags "python,fastapi,auth"
```

## Проблема, которую решает Noema

LLM хороши в генерации. Всё остальное они делают плохо:

| Проблема | Что происходит |
|---|---|
| **Уверенно, но ошибочно** | Решение выглядит идеально, а по факту — ошибка импорта, дыра в безопасности, баг в логике |
| **Никто не проверяет** | Сгенерированный код летит в продакшен без единой формальной проверки |
| **Нельзя переиграть** | Плохой результат — непонятно, где система ошиблась и почему |
| **Короткая память** | Каждая задача решается с нуля — прошлые решения и инциденты забываются |
| **Нет экономики** | Токены жгутся без учёта — стоимость каждой строки неизвестна |
| **Всё руками** | Фиксы, ревью, гейты — всё ручное |

Noema превращает генерацию из одноразовой ставки в **инженерный процесс с чекпоинтами, верификацией и аудитом**.

## Возможности

### Основной цикл

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

Нейронная сторона **предлагает**, символьная — **распоряжается**, и весь обмен записывается в аудируемый артефакт. Ни один непроверенный результат не принимается.

### Что внутри

| Возможность | Описание |
|---|---|
| **Генерация решений** | `noema think "Real-time Chat"` — полная архитектура, стек, код, оптимизации, безопасность |
| **Формальная верификация (Z3)** | Каждая гипотеза проверяется против символического контракта, извлечённого из требований. Solver недоступен → решение отклоняется (fail-closed) |
| **Статический анализ до запуска** | Чистый AST-проход: синтаксис, гигиена импортов, неопределённые имена — вердикт ещё до запуска непроверенного кода |
| **Изолированная песочница** | Docker без сети, лимиты CPU/памяти/времени. Статический вердикт короткозамыкает запуск |
| **Аудит рассуждений** | Каждый шаг мысли, вердикт Z3 и AST фиксируются. Старые вердикты перепроверяются детерминированно — без LLM |
| **Автономное самоисцеление** | Инцидент (Sentry/webhook) → фикс → ветка → PR с пройденной валидацией. Merge-гейт блокирует, если оценка judge ниже порога |
| **Самоэволюция** | Мутации промптов/стратегий применяются только при зелёных тестах (`evolution_test_before_apply`) |
| **22 доменных модуля** | auth, database, gateway, graphql, ml_ops, mobile, terraform, websocket и другие — работают автономно и вместе |
| **Специализированные ядра и агенты** | Архитектор, кодер, security, devops, DBA, AI-engineer. Пайплайны: fullstack, quick, security, arch-review |
| **Память и знания** | Эпизодическая, семантическая и процедурная память + доменная база знаний |
| **Экономика токенов** | Каждый вызов LLM трассируется, атрибутируется и конвертируется в стоимость. Бюджеты и circuit breakers |
| **Воспроизводимые бенчмарки** | Одна матрица задач по провайдерам/моделям → `results.json` + CSV-сводки с пофайловым разложением токенов/стоимости |
| **Production API** | FastAPI: rate limiting, API-ключи, квоты по тенантам, request timeout guard (504 + exempt-пути), RFC 7807, Prometheus-метрики, SSE-стриминг |
| **Грид-федерация** | `noema grid federate`: подзадачи делегируются пирам по gRPC с circuit breaker и ретраями. Локальный фолбэк при недоступности пиров. Вклад каждой ноды пишется в аудируемый ledger |
| **Дашборд грида** | `GET /grid` и `noema grid status`: живое состояние флота воркеров — латентность, токены, ошибки по каждой ноде + итоги по кластеру |

## Установка

```bash
# Минимальная
pip install -e .

# Полная (Z3-верификация, векторный поиск, LLM-провайдеры)
pip install -e ".[full]"

# Всё (dev-инструменты, БД, gRPC, vault)
pip install -e ".[dev,full,db,grpc,vault]"
```

**Python 3.11+** (рекомендуется 3.12 — основная разработка и CI).

**LLM-провайдеры:** `openai`, `anthropic`, `ollama` + встроенный **fallback-провайдер** (работает без ключей — для демо и CI).

Настройка через env: `NOEMA_LLM__PROVIDER=openai`, `NOEMA_LLM__MODEL=...`, `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.

### Проверка установки

```bash
noema --help                  # CLI
python -m noema --help        # Модуль (надёжнее на Windows)

python demo.py                # 12 живых демо, ноль API-ключей
# Ожидай: === ALL DEMOS COMPLETE ===
```

## Использование

### CLI

```bash
# Генерация полного решения
noema think "Real-time Chat App" --tags "python,websocket,redis" --complexity complex --output full

# Scaffold на диск
noema think "Auth service" --scaffold --scaffold-dir ./out

# Пайплайны ядер
noema pipeline fullstack --title "My Project"
noema pipeline security --title "API audit"

# Знания, память, граф
noema knowledge search -q "database optimization"
noema memory stats
noema graph suggest --tags "python,fastapi,redis"

# Автономия и эволюция
noema evolve
noema agents
noema modules list

# Грид-федерация
noema grid federate --peer localhost:50051
noema grid status
```

### API-сервер

```bash
noema serve
# http://localhost:8000
```

Все эндпоинты доступны в корне (`/think`) и под версионированным префиксом (`/api/v1/think`).

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
| GET | `/grid` | Живое состояние грида |
| GET | `/admin/metrics` | Prometheus-метрики |

```bash
curl -X POST http://localhost:8000/think \
  -H "Content-Type: application/json" \
  -d '{"title": "Event bus on RabbitMQ", "complexity": "complex", "tags": ["python", "rabbitmq"]}'
```

## Продакшен-настройка

**Дефолтная конфигурация — для разработки и демо, не для продакшена.** Перед деплоем обязательно настрой:

### Обязательные

1. **LLM-провайдер** — по умолчанию `fallback` (шаблонные ответы, не LLM)
   ```bash
   export NOEMA_LLM__PROVIDER=openai
   export OPENAI_API_KEY=sk-...
   ```

2. **API-ключ** — по умолчанию auth выключен (пустой ключ)
   ```bash
   export NOEMA_API__API_KEY=your-secret-key
   ```

3. **Neurosymbolic verification** — по умолчанию выключена
   ```bash
   export NOEMA_NS__ENABLED=true
   pip install -e ".[full]"   # нужен Z3
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

### Рекомендуемые

- **Redis** — для rate limiting в multi-worker deployment
- **Sentry** — мониторинг ошибок (`NOEMA_OBS__SENTRY_DSN`)
- **Trusted proxies** — если за reverse proxy (`NOEMA_API__TRUSTED_PROXIES`)

### Fail-closed поведение

Noema следует принципу **fail-closed**: если компонент недоступен, система отказывает безопасно, а не фабрикирует ответ:

- LLM недоступен → `RuntimeError`, а не шаблонный ответ
- Webhook secret пустой → запрос отклоняется
- Z3 solver недоступен → верификация не проходит

## Автономное самоисцеление

1. **Инцидент** — Sentry-алерт или `POST /webhooks/incident` нормализуется в `Incident`
2. **Фикс** — задача прогоняется через `NoemaEngine`, результат валидируется тестами (`validate_solution(run_tests=True)`). Без `all_valid` PR не создаётся
3. **PR** — httpx-клиент GitHub (без PyGithub) открывает ветку и pull-request
4. **Merge-гейт** — CI-джоба блокирует мёрж, если `judge_score` ниже порога или песочница упала
5. **Эволюция** — кандидаты промптов/мутаций применяются только при зелёных тестах

Настройка: `NOEMA_AUTONOMY__GITHUB_TOKEN`, `NOEMA_AUTONOMY__GITHUB_REPO`, `NOEMA_AUTONOMY__GITHUB_BASE_BRANCH`.

## Эксперименты и бенчмарки

Воспроизводимый раннер: одна матрица задач по провайдерам и моделям, сбор wall-time, токенов, judge-оценки, стоимости и опциональной валидации песочницы в `results/`.

```bash
# Демо без ключей (fallback-провайдер)
python -m noema.experiments.runner experiments/experiments.yaml --out results

# Реальные модели: укажи провайдера в experiments.yaml и экспортируй ключ
# results/<experiment>/<run_id>/results.json — по записи на (task, provider, model, repetition)
# results/<experiment>/<run_id>/runs.csv      — то же в CSV
# results/<experiment>/<run_id>/summary.csv   — агрегаты по (provider, model)
```

CI-джоба гоняет smoke-бенчмарк ночью на fallback-провайдере и заливает артефакты. Тот же раннер доступен как сервис: `POST /experiments`.

## Структура проекта

```
noema/
  autonomy/        # инциденты → фиксы → PR
  neurosymbolic/   # Z3-верификация + AST-анализ в одном пайплайне
  sandbox/         # Docker-песочница + статические проверки до запуска
  tracing/         # reasoning-trace: перепроверка вердиктов без LLM
  experiments/     # воспроизводимый бенчмарк-раннер + merge-gate
  knowledge/       # база знаний + 22 доменных модуля
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

## Качество и тестирование

| Метрика | Значение |
|---|---|
| **Тесты** | 2 681 (pytest + hypothesis + pytest-benchmark) |
| **Исходные файлы** | 219 Python-модулей |
| **Строки кода** | ~40 000+ |
| **CI** | Ruff lint + format, mypy, bandit, safety, pip-audit |
| **Платформы** | Ubuntu + Windows, Python 3.12 + 3.13 |
| **Кодировка** | Mojibake-гейт — сломанные юникод-строки не проходят CI |

## Документация

- [Whitepaper](docs/WHITEPAPER.md) — видение и дизайн-принципы
- [Roadmap](docs/ROADMAP.md) — три фазы: Архитектор → Автопеитическое предприятие → Глобальный грид Noema
- [Конфигурация](docs/configuration.md) — все env-переменные и YAML-опции
- [Деплой](docs/deployment.md) — Docker / Compose / K8s / Helm / Terraform
- [Быстрый старт](docs/getting-started.md) — туториал
- [Примеры API](docs/api-examples.md) — примеры запросов и ответов
- [Production-чеклист](docs/production-checklist.md) — готовность к деплою
- [Мониторинг](docs/monitoring-setup.md) — настройка наблюдаемости
- [Производительность](docs/performance-tuning.md) — руководство по оптимизации
- [Troubleshooting](docs/troubleshooting.md) — частые проблемы и решения

## Дорожная карта

| Фаза | Название | Статус |
|---|---|---|
| **1** | **Архитектор** — генерация + верификация + песочница + бенчмарки + знания + reasoning-trace | **Готово** |
| **2** | **Автопеитическое предприятие** — инцидент → PR, merge-гейт, эволюция с авто-апплаем, бенчмарк-сервис | **В работе** |
| **3** | **Глобальный грид Noema** — мульти-нодный пул, gRPC-федерация, токен/ledger-экономика, живой дашборд | **В работе** |

## Лицензия

MIT. Открытая разработка — идеи, 이슈 и PR приветствуются.
