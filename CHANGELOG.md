# Changelog

## v1.0.0 — 2026-08-30
### Added
- Unit-тесты (pytest) + тест-шаг в CI
- Structured logging вместо print()
- Retries публикации на HuggingFace (tenacity)
- Error handling + Telegram-алерт при сбое
- LICENSE (MIT), requirements.txt, .env.example
- Type hints и валидация входных данных
- Бейджи и схема пайплайна в README

### Fixed
- Куратор переведён с депрекейтнутого llama-3.1-8b-instant на openai/gpt-oss-120b (Groq deprecation 16.08.2026)
- Обрезка текстов до 700 символов под TPM-лимит Groq

## v0.9.0 — 2026-08-16
### Added
- Первая версия пайплайна: n8n-сбор → LLM-курация → LoRA-обучение на GitHub Actions → публикация на HuggingFace
- Telegram-отчёты после каждого цикла