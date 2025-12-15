# 🛠️ Руководство разработчика AI Scientometer

## 🏗️ Архитектура системы

### **Общая схема**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend│    │   FastAPI       │    │   MongoDB       │
│   (TypeScript)  │◄──►│   Backend       │◄──►│   Atlas         │
│   Port: 3002    │    │   Port: 8000    │    │   Cloud DB      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   AI Services   │              │
         │              │ • SentenceTransf│              │
         │              │ • Scikit-learn  │              │
         │              │ • OpenAlex API  │              │
         │              └─────────────────┘              │
         │                                               │
         └───────────────────────────────────────────────┘
```

### **Структура проекта**
```
ai-scientometer/
├── 📁 ai-dashboard/          # React Frontend
│   ├── 📁 src/
│   │   ├── 📁 components/    # React компоненты
│   │   │   ├── AIChat.tsx    # AI чат интерфейс
│   │   │   ├── Dashboard.tsx # Главная панель
│   │   │   └── ...
│   │   ├── App.tsx           # Главный компонент
│   │   └── App.css          # Стили
│   ├── package.json         # NPM зависимости
│   └── tsconfig.json        # TypeScript конфиг
├── 📁 datasets/             # Данные и модели
│   ├── 📁 models/           # Обученные ML модели
│   ├── 📁 processed/        # Обработанные данные
│   └── 📁 versions/         # Версии датасетов
├── scientometer.py          # Главный backend файл
├── requirements.txt         # Python зависимости
├── .env                     # Переменные окружения
└── README.md               # Документация
```

## 🔧 Backend разработка

### **Основные компоненты**

#### **1. AIService класс**
```python
class AIService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = SGDClassifier()
        self.db = get_database()

    def generate_chat_response(self, message: str) -> Dict:
        """Главный метод для обработки чата"""
        # Анализ типа запроса
        # Генерация ответа
        # Возврат структурированного ответа

    def search_papers(self, query: str) -> List[Dict]:
        """Семантический поиск статей"""
        # Векторизация запроса
        # Поиск в базе данных
        # Ранжирование результатов
```

#### **2. API Endpoints**
```python
@app.post("/api/v1/chat")
async def chat_endpoint(request: ChatMessage):
    """Основной endpoint для чата"""

@app.get("/api/v1/search")
async def search_papers(query: str, limit: int = 10):
    """Поиск научных статей"""

@app.get("/api/v1/trends")
async def get_trends(field: str = None):
    """Анализ трендов в науке"""
```

#### **3. Модели данных**
```python
class ChatMessage(BaseModel):
    message: str
    context: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    recommendations: List[str]
    papers: List[Dict]
    journals: List[Dict]
    confidence: float
```

### **Добавление новых функций**

#### **1. Новый тип запроса в чате**
```python
# В методе generate_chat_response добавить:
elif any(word in message_lower for word in ['новый', 'тип', 'запроса']):
    return self._handle_new_query_type(message)

# Создать новый обработчик:
def _handle_new_query_type(self, message: str) -> Dict:
    """Обработка нового типа запросов"""
    response = "Ваш новый функционал здесь"

    return {
        "response": response,
        "recommendations": ["Рекомендация 1", "Рекомендация 2"],
        "papers": [],
        "journals": [],
        "confidence": 0.8
    }
```

#### **2. Новый API endpoint**
```python
@app.get("/api/v1/new-feature")
async def new_feature_endpoint(param: str):
    """Описание нового endpoint"""
    try:
        ai = get_ai_service()
        result = ai.new_feature_method(param)
        return {"status": "success", "data": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

### **Работа с базой данных**

#### **Подключение к MongoDB**
```python
def get_database():
    client = MongoClient(MONGODB_URL)
    return client[DATABASE_NAME]

def get_collection(name: str):
    db = get_database()
    return db[name]
```

#### **Операции с данными**
```python
# Сохранение статьи
def save_paper(paper_data: Dict):
    collection = get_collection("papers")
    collection.insert_one(paper_data)

# Поиск статей
def find_papers(query: Dict, limit: int = 10):
    collection = get_collection("papers")
    return list(collection.find(query).limit(limit))

# Обновление статьи
def update_paper(paper_id: str, update_data: Dict):
    collection = get_collection("papers")
    collection.update_one({"_id": paper_id}, {"$set": update_data})
```

## ⚛️ Frontend разработка

### **Структура React приложения**

#### **1. Главный компонент App.tsx**
```typescript
function App() {
  const [activePage, setActivePage] = useState<ActivePage>('dashboard');

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard': return <Dashboard />;
      case 'chat': return <AIChat />;
      default: return <Dashboard />;
    }
  };

  return (
    <div className="App">
      <Navigation activePage={activePage} setActivePage={setActivePage} />
      {renderPage()}
    </div>
  );
}
```

#### **2. AI Chat компонент**
```typescript
const AIChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);

  const sendMessage = async (message: string) => {
    setIsThinking(true);
    // Симуляция "думания"
    await simulateThinking();

    // Отправка запроса к API
    const response = await fetch('/api/v1/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message})
    });

    const data = await response.json();
    setMessages(prev => [...prev, data]);
    setIsThinking(false);
  };

  return (
    <div className="ai-chat">
      {/* Интерфейс чата */}
    </div>
  );
};
```

### **Добавление новых компонентов**

#### **1. Создание нового компонента**
```typescript
// src/components/NewComponent.tsx
import React, { useState, useEffect } from 'react';

interface NewComponentProps {
  data: any[];
  onUpdate: (data: any) => void;
}

const NewComponent: React.FC<NewComponentProps> = ({ data, onUpdate }) => {
  const [state, setState] = useState(null);

  useEffect(() => {
    // Логика инициализации
  }, []);

  return (
    <div className="new-component">
      {/* JSX разметка */}
    </div>
  );
};

export default NewComponent;
```

#### **2. Добавление стилей**
```css
/* src/App.css */
.new-component {
  display: flex;
  flex-direction: column;
  padding: 20px;
  background: #1e293b;
  border-radius: 12px;
}

.new-component-item {
  margin-bottom: 16px;
  padding: 12px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  transition: all 0.2s ease;
}

.new-component-item:hover {
  background: rgba(255, 255, 255, 0.15);
  transform: translateY(-2px);
}
```

### **API интеграция**

#### **Создание API клиента**
```typescript
// src/services/api.ts
class APIClient {
  private baseURL = 'http://localhost:8000/api/v1';

  async chat(message: string, context?: string) {
    const response = await fetch(`${this.baseURL}/chat`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message, context})
    });
    return response.json();
  }

  async searchPapers(query: string, limit = 10) {
    const response = await fetch(
      `${this.baseURL}/search?query=${encodeURIComponent(query)}&limit=${limit}`
    );
    return response.json();
  }
}

export const apiClient = new APIClient();
```

#### **Использование в компонентах**
```typescript
import { apiClient } from '../services/api';

const MyComponent = () => {
  const [data, setData] = useState([]);

  const fetchData = async () => {
    try {
      const result = await apiClient.searchPapers('machine learning');
      setData(result.papers);
    } catch (error) {
      console.error('Ошибка загрузки данных:', error);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);
};
```

## 🧪 Тестирование

### **Backend тесты**
```python
# test_api.py
import pytest
import requests

def test_chat_endpoint():
    response = requests.post('http://localhost:8000/api/v1/chat',
                           json={'message': 'Привет'})
    assert response.status_code == 200
    data = response.json()
    assert 'ai_response' in data
    assert data['ai_response']['confidence'] > 0

def test_search_endpoint():
    response = requests.get('http://localhost:8000/api/v1/search?query=AI')
    assert response.status_code == 200
    data = response.json()
    assert 'papers' in data
```

### **Frontend тесты**
```typescript
// src/components/__tests__/AIChat.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import AIChat from '../AIChat';

test('отправка сообщения в чате', async () => {
  render(<AIChat />);

  const input = screen.getByPlaceholderText('Задайте вопрос AI агенту...');
  const button = screen.getByText('Отправить');

  fireEvent.change(input, { target: { value: 'Тестовое сообщение' } });
  fireEvent.click(button);

  expect(screen.getByText('Тестовое сообщение')).toBeInTheDocument();
});
```

## 🚀 Развертывание

### **Docker контейнеризация**
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "scientometer.py"]
```

```dockerfile
# ai-dashboard/Dockerfile
FROM node:16-alpine

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

EXPOSE 3002
CMD ["npm", "start"]
```

### **Docker Compose**
```yaml
# docker-compose.yml
version: '3.8'

services:
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URL=${MONGODB_URL}
    depends_on:
      - mongodb

  frontend:
    build: ./ai-dashboard
    ports:
      - "3002:3002"
    depends_on:
      - backend

  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

volumes:
  mongodb_data:
```

### **CI/CD Pipeline**
```yaml
# .github/workflows/deploy.yml
name: Deploy AI Scientometer

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # Команды развертывания
```

## 📊 Мониторинг и логирование

### **Настройка логирования**
```python
import logging
from datetime import datetime

# Настройка логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scientometer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Использование в коде
logger.info("Запуск AI Scientometer")
logger.error(f"Ошибка: {error}")
```

### **Метрики производительности**
```python
import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"{func.__name__} выполнен за {end-start:.2f}s")
        return result
    return wrapper

@measure_time
def search_papers(query: str):
    # Логика поиска
    pass
```

## 🔒 Безопасность

### **Валидация входных данных**
```python
from pydantic import BaseModel, validator

class ChatMessage(BaseModel):
    message: str
    context: Optional[str] = None

    @validator('message')
    def validate_message(cls, v):
        if len(v) > 1000:
            raise ValueError('Сообщение слишком длинное')
        if not v.strip():
            raise ValueError('Сообщение не может быть пустым')
        return v.strip()
```

### **Rate Limiting**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v1/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, message: ChatMessage):
    # Логика чата
    pass
```

## 📈 Оптимизация производительности

### **Кэширование**
```python
from functools import lru_cache
import redis

# In-memory кэш
@lru_cache(maxsize=1000)
def get_paper_embedding(paper_id: str):
    # Вычисление эмбеддинга
    pass

# Redis кэш
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_search_results(query: str, results: List):
    redis_client.setex(f"search:{query}", 3600, json.dumps(results))

def get_cached_results(query: str):
    cached = redis_client.get(f"search:{query}")
    return json.loads(cached) if cached else None
```

### **Асинхронная обработка**
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_multiple_queries(queries: List[str]):
    with ThreadPoolExecutor(max_workers=4) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, process_single_query, query)
            for query in queries
        ]
        results = await asyncio.gather(*tasks)
    return results
```

---

**🔧 Готовы к разработке? Начните с изучения кодовой базы и создания первого Pull Request!**
