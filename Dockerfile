# 1) Беремо базовий образ з Python
FROM python:3.12-slim

# 2) Робоча папка всередині контейнера
WORKDIR /app

# 3) Копіюємо тільки файли залежностей (щоб кеш Docker працював швидше)
COPY pyproject.toml ./
COPY src ./src

# 4) Ставимо залежності
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e .

# 5) Копіюємо наш код
COPY src ./src
COPY artifacts ./artifacts

# 6) Кажемо: контейнер слухає порт 8000
EXPOSE 8000

# 7) Команда запуску (API)
CMD ["uvicorn", "mini02.api:app", "--host", "0.0.0.0", "--port", "8000"]
