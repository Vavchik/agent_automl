# AutoML-агент для хакатона «Альфа Хак: AutoML»

Агент получает `task.md` с данными и сам доводит задачу до `submission.csv`: понимает задачу,
строит честную валидацию, обучает бейзлайн и затем ищет улучшения по дереву решений (как AIDE):
каждая гипотеза (признаки от LLM, log-таргет, веса классов, тюнинг, отбор и слияние признаков) —
новый узел; узлы без устойчивого прироста становятся тупиками, поиск откатывается к лучшим веткам,
удачные ветки объединяются. Упавший код LLM чинит по traceback. В конце — бленд моделей,
AutoGluon best_quality на тех же фолдах и ансамбль лучших узлов.

```bash
python solution.py path/to/task.md            # как на платформе
AUTOML_FAST=1 python solution.py path/to/task.md   # быстрый режим для отладки (~1-2 мин)
```

LLM включается переменной `OPENAI_API_KEY` (+ `OPENAI_MODEL`, `OPENAI_BASE_URL`, иначе значения из `config.py`).
На платформе их выставляет проверяющая система, как для `baseline.py`, — ключ в код не вписываем.
Локально без ключа агент работает на эвристиках; первая строка `agent.log` говорит, включена ли LLM.

## Как устроен запуск

```
solution.py            страховка: при любом сбое до первого сабмита -> аварийный ответ
 └ core/pipeline.py    порядок стадий
    1. task/loader.py + llm_parser.py   тип задачи, метрика, формат ответа -> TaskSpec
    2. task/dtypes.py                   признаки; категории общие для train и test
    3. validation/splits.py             фолды под тест (время / stratified / KFold), adversarial-проверка
    4. ml/trainer.py                    бейзлайн LightGBM -> submission.csv v0
    5. agent/research_loop.py + tree.py дерево решений: узел -> действие (planner) -> кандидат (actions,
                                        починка кода по traceback) -> обучение -> правило принятия
    6. ml/trainer.py, autogluon_stage.py, ensemble.py
                                        бленд LightGBM + CatBoost + XGBoost, AutoGluon на наших фолдах, ансамбль
```

После шага 4 `submission.csv` существует всегда, дальше он только улучшается.
Рядом с `task.md` появляются `agent.log`, `experiments.jsonl` (журнал гипотез) и `report.json`.

## Структура и роли

| Папка | Роль | За что отвечает |
| --- | --- | --- |
| `solution.py`, `config.py`, `core/`, `runtime/`, `docker/`, `scripts/` | Р1 Платформа | запуск в Docker, дедлайн 3 ч, аварийный сабмит, сборка ZIP, сабмиты |
| `task/` | Р2 Понимание задачи | метрики, формат ответа, типы колонок, LLM-разбор `task.md` |
| `ml/` | Р3 ML-ядро | модели, кросс-валидация, тюнинг, постобработка, ансамбли |
| `agent/` | Р4 Агент | LLM, действия, планировщик, песочница, журнал |
| `validation/`, `bench/` | Р5 Валидация и стенд | схема фолдов, правило принятия, стенд, синтетические задачи |
| `core/contracts.py` | все | общие структуры; менять только через PR с аппрувом Р1 |
| `legacy/` | — | старая версия, в ZIP не входит |

В начале каждого файла написано, что он делает и что в нём стоит улучшать.

## Проверка перед сабмитом

```bash
pip install -r requirements.txt pytest
pytest -q                          # юнит-тесты + сквозной прогон на синтетике (~5 мин)
python bench/run_bench.py          # все задачи в быстром режиме -> bench/results.csv
python bench/run_bench.py --only bench/tasks   # только синтетические «private»-задачи
docker/run_task.sh "given_data/Регрессия"      # как на платформе: 8 CPU, 32 GB
scripts/build_zip.sh               # solution.zip: в корне solution.py (все модули внутри) + requirements.txt
```

Сабмит отправляет только Р1 и только после зелёного стенда, включая синтетические задачи:
на private платформа сама перезапустит версию с лучшим public score.

## Главные настройки (`config.py`)

`TIME_LIMIT_SEC`, `SAFETY_MARGIN_SEC` — бюджет времени; `N_FOLDS`; `ACCEPT_K` — строгость принятия
гипотез; `RESEARCH_MODELS` / `FINAL_MODELS`; `LLM_TOKEN_BUDGET`. Любое значение можно переопределить
переменной окружения `AUTOML_*` без правки кода.
