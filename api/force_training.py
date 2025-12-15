#!/usr/bin/env python3
"""
🔥 ПРИНУДИТЕЛЬНЫЙ ЗАПУСК ОБУЧЕНИЯ AI
Создает модели немедленно с существующими данными
"""

import time

import requests

BASE_URL = "http://localhost:8000"


def force_start_training() -> bool | None:
    """Принудительный запуск обучения"""
    print("🔥 ПРИНУДИТЕЛЬНЫЙ ЗАПУСК ОБУЧЕНИЯ AI")
    print("=" * 50)

    # 1. Останавливаем текущее обучение
    print("🛑 Останавливаем текущее обучение...")
    try:
        response = requests.post(f"{BASE_URL}/api/v1/training/stop", timeout=5)
        print("✅ Обучение остановлено")
    except:
        print("⚠️ Обучение уже остановлено")

    time.sleep(2)

    # 2. Проверяем данные
    print("📊 Проверяем данные в системе...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/data/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            papers_count = data.get("papers_count", 0)
            print(f"📚 Найдено статей: {papers_count}")

            if papers_count == 0:
                print("❌ Нет данных для обучения!")
                print("🔍 Загружаем данные...")

                # Загружаем данные
                load_response = requests.post(
                    f"{BASE_URL}/api/v1/data/load",
                    json={
                        "query": "artificial intelligence machine learning",
                        "papers_limit": 50,
                    },
                    timeout=30,
                )

                if load_response.status_code == 200:
                    print("✅ Данные загружены")
                    time.sleep(10)  # Ждем загрузки
                else:
                    print("❌ Ошибка загрузки данных")
                    return False
        else:
            print("❌ Не удается получить статистику")
            return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

    # 3. Запускаем обучение
    print("🚀 Запускаем обучение...")
    try:
        response = requests.post(f"{BASE_URL}/api/v1/training/start", timeout=5)
        if response.status_code == 200:
            print("✅ Обучение запущено!")

            # Ждем и проверяем результат
            print("⏳ Ждем результатов обучения...")

            for i in range(12):  # Ждем до 2 минут
                time.sleep(10)

                try:
                    status_response = requests.get(
                        f"{BASE_URL}/api/v1/training/status", timeout=5
                    )
                    if status_response.status_code == 200:
                        status = status_response.json()

                        epoch = status.get("current_stats", {}).get("epoch", 0)
                        samples = status.get("data_samples", 0)

                        print(
                            f"📊 Проверка {i + 1}/12: Эпоха {epoch}, Образцов {samples}"
                        )

                        if epoch > 0:
                            print("🎉 ОБУЧЕНИЕ НАЧАЛОСЬ!")

                            # Проверяем модели
                            models_response = requests.get(
                                f"{BASE_URL}/api/v1/training/models", timeout=5
                            )
                            if models_response.status_code == 200:
                                models = models_response.json()
                                model_count = len(models.get("models", []))
                                print(f"🏆 Создано моделей: {model_count}")

                                if model_count > 0:
                                    print("✅ МОДЕЛИ СОЗДАНЫ УСПЕШНО!")
                                    return True

                        elif samples > 0:
                            print(f"📚 Данные загружены: {samples} образцов")

                except Exception as e:
                    print(f"⚠️ Ошибка проверки: {e}")

            print("⏰ Время ожидания истекло")
            return False

        print("❌ Не удалось запустить обучение")
        return False

    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        return False


def check_models() -> None:
    """Проверка созданных моделей"""
    print("\n🏆 ПРОВЕРКА СОЗДАННЫХ МОДЕЛЕЙ:")
    print("-" * 40)

    try:
        response = requests.get(f"{BASE_URL}/api/v1/training/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])

            if models:
                print(f"📁 Найдено моделей: {len(models)}")
                for i, model in enumerate(models[:3]):
                    print(f"  {i + 1}. {model['filename']} ({model['size_mb']} MB)")
                    if model.get("is_best"):
                        print("     🏆 ЛУЧШАЯ МОДЕЛЬ")
            else:
                print("❌ Модели не найдены")
        else:
            print("❌ Не удается получить список моделей")
    except Exception as e:
        print(f"❌ Ошибка: {e}")


def main() -> None:
    print("🤖 AI SCIENTOMETER - ПРИНУДИТЕЛЬНОЕ ОБУЧЕНИЕ")
    print("🎯 Этот скрипт заставит систему создать модели НЕМЕДЛЕННО!")
    print()

    success = force_start_training()

    if success:
        print("\n🎉 УСПЕХ! Система обучается и создает модели!")
        check_models()

        print("\n📊 Для мониторинга запустите:")
        print("python3 live_training_monitor.py")

    else:
        print("\n❌ Не удалось запустить обучение")
        print("🔧 Проверьте что сервер работает:")
        print("python3 scientometer.py")


if __name__ == "__main__":
    main()
