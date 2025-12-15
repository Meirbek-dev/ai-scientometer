#!/usr/bin/env python3
"""
🔥 LIVE МОНИТОРИНГ НЕПРЕРЫВНОГО ОБУЧЕНИЯ AI В РЕАЛЬНОМ ВРЕМЕНИ!
Показывает как AI становится умнее каждую секунду
"""

import os
import time
from datetime import datetime

import requests

BASE_URL = "http://localhost:8000"


class LiveTrainingMonitor:
    def __init__(self) -> None:
        self.training_data = []
        self.start_time = datetime.now()

    def clear_screen(self) -> None:
        os.system("clear" if os.name == "posix" else "cls")

    def print_header(self) -> None:
        print("🔥" * 50)
        print("🧠 AI SCIENTOMETER - LIVE TRAINING MONITOR")
        print("🚀 НЕПРЕРЫВНОЕ ОБУЧЕНИЕ В РЕАЛЬНОМ ВРЕМЕНИ!")
        print("🔥" * 50)
        print()

    def get_training_status(self):
        try:
            response = requests.get(f"{BASE_URL}/api/v1/training/status", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None

    def get_training_metrics(self):
        try:
            response = requests.get(f"{BASE_URL}/api/v1/training/metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None

    def start_training_if_not_running(self):
        try:
            response = requests.post(f"{BASE_URL}/api/v1/training/start", timeout=5)
            return response.status_code == 200
        except:
            return False

    def display_status(self, status) -> None:
        if not status:
            print("❌ Не удается подключиться к серверу")
            print("🔧 Убедитесь что сервер запущен: python3 scientometer.py")
            return

        current_stats = status.get("current_stats", {})

        # Основная информация
        print(
            f"📊 СТАТУС: {'🟢 ОБУЧАЕТСЯ' if status['is_training'] else '🔴 ОСТАНОВЛЕНО'}"
        )

        if status["is_training"]:
            print(f"⏱️  ВРЕМЯ: {status.get('training_duration_formatted', 'N/A')}")
            print(f"🔢 ЭПОХА: {current_stats.get('epoch', 0)}")
            print(f"📉 LOSS: {current_stats.get('loss', 0):.4f}")
            print(f"🎯 ACCURACY: {current_stats.get('accuracy', 0):.4f}")
            print(f"📚 ОБРАЗЦОВ: {status.get('data_samples', 0)}")
            print(f"🔄 ОБРАБОТАНО: {current_stats.get('samples_processed', 0)}")

            # Индикаторы прогресса
            progress = status.get("progress_indicators", {})
            loss_trend = progress.get("loss_trend", "stable")
            accuracy_trend = progress.get("accuracy_trend", "stable")

            print("📈 ТРЕНДЫ:")
            print(
                f"   Loss: {'📉 Снижается' if loss_trend == 'decreasing' else '📊 Стабильно'}"
            )
            print(
                f"   Accuracy: {'📈 Растет' if accuracy_trend == 'increasing' else '📊 Стабильно'}"
            )

            # Улучшения
            improvements = len(current_stats.get("improvements", []))
            if improvements > 0:
                print(f"🎉 УЛУЧШЕНИЙ МОДЕЛИ: {improvements}")

        print()

    def display_recent_history(self, status) -> None:
        history = status.get("recent_history", [])
        if not history:
            return

        print("📊 ПОСЛЕДНИЕ 5 ЭПОХ:")
        print("-" * 60)
        print(f"{'Эпоха':<8} {'Loss':<10} {'Accuracy':<10} {'Время'}")
        print("-" * 60)

        for entry in history[-5:]:
            timestamp = entry["timestamp"][:19]  # Убираем микросекунды
            print(
                f"{entry['epoch']:<8} {entry['loss']:<10.4f} {entry['accuracy']:<10.4f} {timestamp}"
            )

        print()

    def display_improvements(self, status) -> None:
        improvements = status.get("current_stats", {}).get("improvements", [])
        if not improvements:
            return

        print("🏆 УЛУЧШЕНИЯ МОДЕЛИ:")
        print("-" * 50)

        for imp in improvements[-3:]:  # Последние 3 улучшения
            print(
                f"🎯 Эпоха {imp['epoch']}: Accuracy {imp['accuracy']:.4f} (Loss {imp['loss']:.4f})"
            )

        print()

    def create_live_chart(self, metrics) -> None:
        if not metrics or not metrics.get("metrics", {}).get("epochs"):
            return

        try:
            import matplotlib.pyplot as plt

            plt.ion()  # Включаем интерактивный режим

            data = metrics["metrics"]
            epochs = data["epochs"]
            loss_values = data["loss_values"]
            accuracy_values = data["accuracy_values"]

            if len(epochs) < 2:
                return

            # Создаем график
            _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            # График Loss
            ax1.plot(epochs, loss_values, "r-", linewidth=2, label="Loss")
            ax1.set_title("🔥 Live Training Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # График Accuracy
            ax2.plot(epochs, accuracy_values, "g-", linewidth=2, label="Accuracy")
            ax2.set_title("🎯 Live Training Accuracy")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.pause(0.1)  # Обновляем график

        except Exception as e:
            print(f"⚠️ Ошибка создания графика: {e}")

    def run_live_monitor(self) -> None:
        print("🚀 Запуск Live Training Monitor...")
        print("📊 Обновление каждые 5 секунд")
        print("🔥 Нажмите Ctrl+C для выхода")
        print()

        # Пытаемся запустить обучение
        if self.start_training_if_not_running():
            print("✅ Обучение запущено!")

        try:
            iteration = 0
            while True:
                iteration += 1

                self.clear_screen()
                self.print_header()

                print(
                    f"🔄 Обновление #{iteration} - {datetime.now().strftime('%H:%M:%S')}"
                )
                print()

                # Получаем статус
                status = self.get_training_status()
                self.display_status(status)

                if status and status["is_training"]:
                    self.display_recent_history(status)
                    self.display_improvements(status)

                    # Получаем метрики для графика
                    metrics = self.get_training_metrics()
                    if metrics:
                        summary = metrics.get("summary", {})
                        print("📈 ОБЩАЯ СТАТИСТИКА:")
                        print(f"   Всего эпох: {summary.get('total_epochs', 0)}")
                        print(
                            f"   Лучшая точность: {summary.get('best_accuracy', 0):.4f}"
                        )
                        print(f"   Лучший loss: {summary.get('best_loss', 0):.4f}")
                        print(f"   Улучшений: {summary.get('improvements_count', 0)}")
                        print()

                print("⏳ Следующее обновление через 5 секунд...")
                print("💡 Откройте http://localhost:8000/docs для API")

                time.sleep(5)

        except KeyboardInterrupt:
            print("\n🛑 Мониторинг остановлен")
            print("💡 Обучение продолжается в фоне")
            print(
                "🌐 Проверить статус: curl http://localhost:8000/api/v1/training/status"
            )


def main() -> None:
    monitor = LiveTrainingMonitor()
    monitor.run_live_monitor()


if __name__ == "__main__":
    main()
