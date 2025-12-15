#!/usr/bin/env python3
"""
🤖 Тестирование AI Chat функциональности
Демонстрирует возможности чат-бота как ChatGPT
"""

import time

import requests

BASE_URL = "http://localhost:8000"


def test_chat_message(message: str) -> bool | None:
    """Тестирование отправки сообщения в чат"""
    print(f"\n👤 Пользователь: {message}")
    print("-" * 50)

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/chat", json={"message": message}, timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            ai_response = data["ai_response"]

            print(f"🤖 AI Scientometer: {ai_response['response']}")
            print(f"🎯 Уверенность: {ai_response['confidence']:.0%}")

            if ai_response["recommendations"]:
                print("\n💡 Рекомендации:")
                for i, rec in enumerate(ai_response["recommendations"], 1):
                    print(f"   {i}. {rec}")

            if ai_response["papers"]:
                print("\n📚 Найденные статьи:")
                for i, paper in enumerate(ai_response["papers"], 1):
                    print(f"   {i}. {paper['title']} ({paper['year']})")
                    print(f"      Цитирований: {paper['citations']:,}")

            if ai_response["journals"]:
                print("\n📰 Рекомендованные журналы:")
                for i, journal in enumerate(ai_response["journals"], 1):
                    print(f"   {i}. {journal['name']}")
                    print(
                        f"      IF: {journal['impact_factor']} | {journal['quartile']}"
                    )

            return True

        print(f"❌ Ошибка: {response.status_code}")
        return False

    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False


def test_evaluation(message: str) -> bool | None:
    """Тестирование оценки исследований"""
    print(f"\n🎯 Оценка: {message}")
    print("-" * 50)

    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/chat/evaluate", json={"message": message}, timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            evaluation = data["evaluation"]

            print(f"🤖 AI Оценка: {evaluation['response']}")

            if "detailed_scores" in evaluation:
                print("\n📊 Детальные оценки:")
                for metric, score in evaluation["detailed_scores"].items():
                    print(f"   • {metric.title()}: {score}/10")

            if "improvement_plan" in evaluation:
                print("\n📋 План улучшений:")
                for item in evaluation["improvement_plan"]:
                    print(
                        f"   🔸 {item['priority']}: {item['area']} - {item['suggestion']}"
                    )

            return True

        print(f"❌ Ошибка: {response.status_code}")
        return False

    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False


def test_suggestions() -> bool | None:
    """Тестирование получения предложений"""
    print("\n💡 Получение предложений вопросов...")
    print("-" * 50)

    try:
        response = requests.get(f"{BASE_URL}/api/v1/chat/suggestions", timeout=5)

        if response.status_code == 200:
            data = response.json()

            print("📝 Категории вопросов:")
            for category in data["suggestions"]:
                print(f"\n{category['category']}")
                for question in category["questions"][:2]:  # Показываем первые 2
                    print(f"   • {question}")

            print(f"\n⚡ Быстрые действия: {', '.join(data['quick_actions'])}")
            return True

        print(f"❌ Ошибка: {response.status_code}")
        return False

    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False


def main() -> None:
    print("🤖 AI SCIENTOMETER CHAT - ТЕСТИРОВАНИЕ")
    print("=" * 60)
    print("Демонстрация возможностей AI чат-бота как ChatGPT")
    print("=" * 60)

    # Проверяем доступность сервера
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Сервер недоступен. Запустите: python3 scientometer.py")
            return
    except:
        print("❌ Сервер недоступен. Запустите: python3 scientometer.py")
        return

    print("✅ Сервер доступен, начинаем тестирование...")

    # 1. Тестируем предложения
    test_suggestions()
    time.sleep(1)

    # 2. Тестируем разные типы запросов
    test_cases = [
        "Привет! Что ты умеешь?",
        "Найди статьи про машинное обучение",
        "Посоветуй журнал для публикации по искусственному интеллекту",
        "Какие тренды в области компьютерного зрения?",
        "Оцени качество моего исследования по нейронным сетям",
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 20} ТЕСТ {i}/5 {'=' * 20}")
        success = test_chat_message(test_case)
        if not success:
            print("⚠️ Тест не прошел, продолжаем...")
        time.sleep(2)  # Пауза между тестами

    # 3. Тестируем специальную оценку
    print(f"\n{'=' * 20} ТЕСТ ОЦЕНКИ {'=' * 20}")
    test_evaluation("Моя статья про deep learning с точностью 95% на MNIST")

    print("\n" + "=" * 60)
    print("🎉 ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("💡 Откройте http://localhost:3002 для веб-интерфейса")
    print("📚 API документация: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
