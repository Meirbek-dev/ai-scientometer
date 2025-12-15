#!/usr/bin/env python3
"""
Монитор AI Scientometer - показывает процесс загрузки и обучения системы
"""

import time

import requests

BASE_URL = "http://localhost:8000"


def check_system_status():
    """Проверка состояния системы"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        return response.json()
    except:
        return None


def get_data_stats():
    """Получение статистики данных"""
    try:
        response = requests.get(f"{BASE_URL}/api/v1/data/stats")
        return response.json()
    except:
        return None


def test_ai_search():
    """Тестирование AI поиска"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/analysis/search",
            json={"query": "transformer attention mechanism", "limit": 3},
        )
        return response.json()
    except:
        return None


def load_more_data():
    """Загрузка дополнительных данных"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/data/load",
            json={
                "query": "computer vision CNN ResNet",
                "papers_limit": 15,
                "journals_limit": 5,
            },
        )
        return response.json()
    except:
        return None


def print_separator() -> None:
    print("=" * 80)


def main() -> None:
    print("🔍 AI Scientometer Monitor - Проверка самообучения системы")
    print_separator()

    # 1. Проверка состояния системы
    print("1️⃣ Проверка состояния системы:")
    status = check_system_status()
    if status:
        print("   ✅ Сервер работает")
        print(f"   🤖 AI загружен: {status.get('ai_loaded', False)}")
        print(f"   🗄️ MongoDB: {status.get('mongodb_connected', False)}")
    else:
        print("   ❌ Сервер недоступен")
        return

    print()

    # 2. Текущая статистика данных
    print("2️⃣ Текущая статистика данных:")
    stats = get_data_stats()
    if stats and "error" not in stats:
        print(f"   📄 Статей в базе: {stats.get('papers_count', 0)}")
        print(f"   📚 Журналов в базе: {stats.get('journals_count', 0)}")
        print("   🏷️ Топ концепты:")
        for i, concept in enumerate(stats.get("top_concepts", [])[:5]):
            print(f"      {i + 1}. {concept['name']} ({concept['count']} статей)")
    else:
        print("   ⚠️ Данные недоступны или используются тестовые данные")

    print()

    # 3. Тестирование AI поиска
    print("3️⃣ Тестирование AI поиска:")
    search_result = test_ai_search()
    if search_result:
        print("   🔍 Запрос: 'transformer attention mechanism'")
        print(f"   📊 Найдено статей: {search_result.get('total', 0)}")
        print(f"   🤖 AI включен: {search_result.get('ai_enabled', False)}")

        papers = search_result.get("papers", [])
        if papers:
            print("   📄 Найденные статьи:")
            for i, paper in enumerate(papers[:2]):
                similarity = paper.get("similarity_score", 0)
                print(f"      {i + 1}. {paper.get('title', 'Без названия')[:60]}...")
                print(f"         Релевантность: {similarity:.3f}")
    else:
        print("   ❌ Ошибка поиска")

    print()

    # 4. Демонстрация загрузки новых данных
    print("4️⃣ Демонстрация самообучения - загрузка новых данных:")
    print("   🔄 Запускаем загрузку данных по теме 'computer vision'...")

    load_result = load_more_data()
    if load_result:
        print(f"   ✅ {load_result.get('message', 'Загрузка запущена')}")
        print("   ⏳ Ждем завершения загрузки...")

        # Ждем и проверяем изменения
        time.sleep(10)

        new_stats = get_data_stats()
        if new_stats and "error" not in new_stats:
            print("   📊 Обновленная статистика:")
            print(f"      📄 Статей: {new_stats.get('papers_count', 0)}")
            print(f"      📚 Журналов: {new_stats.get('journals_count', 0)}")

            # Показываем новые концепты
            new_concepts = new_stats.get("top_concepts", [])
            if new_concepts:
                print("   🆕 Обновленные концепты:")
                for i, concept in enumerate(new_concepts[:5]):
                    print(
                        f"      {i + 1}. {concept['name']} ({concept['count']} статей)"
                    )

    else:
        print("   ❌ Ошибка загрузки")

    print()
    print_separator()

    # 5. Где хранятся данные
    print("5️⃣ Где хранятся данные и модели:")
    print("   🗄️ Научные данные: MongoDB Atlas (облачная база)")
    print("      - URL: mongodb+srv://...cluster0.bcuhj7j.mongodb.net/")
    print("      - База: scientometer")
    print("      - Коллекции: papers, journals")
    print()
    print("   🤖 AI модель: Локальный кэш HuggingFace")
    print("      - Путь: ~/.cache/huggingface/transformers/")
    print("      - Модель: sentence-transformers/all-MiniLM-L6-v2")
    print("      - Размер: ~90MB")
    print()
    print("   📊 Векторы и индексы: MongoDB Atlas")
    print("      - Эмбеддинги статей сохраняются в поле 'embedding'")
    print("      - AI поиск использует косинусное сходство")
    print()

    # 6. Процесс самообучения
    print("6️⃣ Как работает самообучение:")
    print("   1. 🔄 Автоматическая загрузка данных из OpenAlex API")
    print("   2. 🤖 Векторизация текстов через sentence-transformers")
    print("   3. 💾 Сохранение в MongoDB с индексацией")
    print("   4. 📊 Обновление статистик и трендов")
    print("   5. 🔍 Улучшение качества поиска с новыми данными")
    print()
    print("   ⏰ Периодичность: каждые 24 часа или по запросу")
    print("   📈 Адаптация: система учится на новых научных данных")

    print()
    print_separator()
    print("✅ Мониторинг завершен!")
    print("🌐 Swagger UI: http://localhost:8000/docs")
    print("📊 Статистика: http://localhost:8000/api/v1/data/stats")


if __name__ == "__main__":
    main()
