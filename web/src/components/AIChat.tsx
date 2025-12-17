import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import {
  Send,
  Plus,
  Menu,
  User,
  Bot,
  Sparkles,
  Target,
  BookOpen,
  Newspaper,
  TrendingUp,
  Lightbulb,
  Clock,
  MessageSquare,
  Brain,
  Loader,
} from "lucide-react";
import { API_BASE_URL } from "../config/api";

interface ChatMessage {
  id: string;
  type: "user" | "ai";
  content: string;
  timestamp: Date;
  topic?: string;
  aiResponse?: {
    response: string;
    recommendations: string[];
    papers: any[];
    journals: any[];
    confidence: number;
  };
}

interface ChatSuggestion {
  category: string;
  questions: string[];
}

interface ChatTopic {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
  messageCount: number;
}

const AIChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [thinkingText, setThinkingText] = useState("");
  const [suggestions, setSuggestions] = useState<ChatSuggestion[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [topics, setTopics] = useState<ChatTopic[]>([]);
  const [currentTopic, setCurrentTopic] = useState<string | null>(null);
  const [showSidebar, setShowSidebar] = useState(window.innerWidth > 768);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Автоскролл к последнему сообщению
  const scrollToBottom = () => {
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "end",
        inline: "nearest",
      });
    }, 100);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isThinking, isLoading]);

  // Обработчик изменения размера экрана
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 768) {
        setShowSidebar(true);
      } else {
        setShowSidebar(false);
      }
    };

    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Загрузка предложений при первом запуске
  useEffect(() => {
    fetchSuggestions();
    // Добавляем приветственное сообщение
    const welcomeMessage: ChatMessage = {
      id: "welcome",
      type: "ai",
      content:
        "Привет! Я AI Scientometer Assistant. Я помогу вам с поиском статей, рекомендациями журналов, анализом трендов и оценкой исследований. Задайте любой вопрос!",
      timestamp: new Date(),
      aiResponse: {
        response:
          "Добро пожаловать в AI Scientometer! Я готов помочь вам с научными исследованиями.",
        recommendations: [],
        papers: [],
        journals: [],
        confidence: 1.0,
      },
    };
    setMessages([welcomeMessage]);
  }, []);

  const fetchSuggestions = async () => {
    try {
      const response = await axios.get(
        `${API_BASE_URL}/api/v1/chat/suggestions`
      );
      setSuggestions(response.data.suggestions);
    } catch (error) {
      console.error("Error fetching suggestions:", error);
    }
  };

  // Эффект "думания" как в ChatGPT
  const simulateThinking = () => {
    const thinkingSteps = [
      "Анализирую ваш запрос...",
      "Ищу релевантную информацию...",
      "Обрабатываю данные из научных баз...",
      "Формирую персонализированный ответ...",
      "Проверяю точность рекомендаций...",
      "Готовлю финальный ответ...",
    ];

    let stepIndex = 0;
    setIsThinking(true);
    setThinkingText(thinkingSteps[0]);

    const thinkingInterval = setInterval(() => {
      stepIndex++;
      if (stepIndex < thinkingSteps.length) {
        setThinkingText(thinkingSteps[stepIndex]);
      } else {
        clearInterval(thinkingInterval);
      }
    }, 800);

    return () => clearInterval(thinkingInterval);
  };

  // Создание или обновление топика
  const updateTopic = (message: string) => {
    const topicTitle =
      message.length > 50 ? message.substring(0, 50) + "..." : message;

    if (!currentTopic) {
      const newTopic: ChatTopic = {
        id: Date.now().toString(),
        title: topicTitle,
        lastMessage: message,
        timestamp: new Date(),
        messageCount: 1,
      };
      setTopics((prev) => [newTopic, ...prev]);
      setCurrentTopic(newTopic.id);
    } else {
      setTopics((prev) =>
        prev.map((topic) =>
          topic.id === currentTopic
            ? {
                ...topic,
                lastMessage: message,
                timestamp: new Date(),
                messageCount: topic.messageCount + 1,
              }
            : topic
        )
      );
    }
  };

  const sendMessage = async (messageText: string) => {
    if (!messageText.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: "user",
      content: messageText,
      timestamp: new Date(),
      topic: currentTopic || undefined,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);
    setShowSuggestions(false);

    // Обновляем топик
    updateTopic(messageText);

    // Запускаем эффект "думания"
    const clearThinking = simulateThinking();

    try {
      // Добавляем случайную задержку для реалистичности (2-4 секунды)
      const thinkingTime = Math.random() * 2000 + 2000;
      await new Promise((resolve) => setTimeout(resolve, thinkingTime));

      const response = await axios.post(`${API_BASE_URL}/api/v1/chat`, {
        message: messageText,
      });

      clearThinking();
      setIsThinking(false);
      setThinkingText("");

      const aiMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: "ai",
        content: response.data.ai_response.response,
        timestamp: new Date(),
        topic: currentTopic || undefined,
        aiResponse: response.data.ai_response,
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      clearThinking();
      setIsThinking(false);
      setThinkingText("");

      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: "ai",
        content:
          "Извините, произошла ошибка при обработке вашего сообщения. Попробуйте еще раз.",
        timestamp: new Date(),
        topic: currentTopic || undefined,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const startNewTopic = () => {
    setCurrentTopic(null);
    setMessages([]);
    setShowSuggestions(true);
  };

  const switchToTopic = (topicId: string) => {
    setCurrentTopic(topicId);
    // В реальном приложении здесь бы загружались сообщения из этого топика
    setMessages([]);
    setShowSuggestions(false);
  };

  const handleSuggestionClick = (question: string) => {
    sendMessage(question);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(inputMessage);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputMessage(e.target.value);

    // Автоматическое изменение высоты
    const textarea = e.target;
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 128) + "px";
  };

  const formatAIResponse = (content: string) => {
    // Простое форматирование Markdown-подобного текста
    return content
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/\n/g, "<br/>");
  };

  const renderMessage = (message: ChatMessage) => {
    const isUser = message.type === "user";

    return (
      <div
        key={message.id}
        className={`mb-6 ${isUser ? "flex justify-end" : "flex justify-start"}`}
      >
        <div className={`max-w-3xl ${isUser ? "order-2" : "order-1"}`}>
          <div className="flex items-start space-x-3 mb-2">
            <div
              className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                isUser
                  ? "bg-gradient-to-br from-blue-500 to-purple-600"
                  : "bg-gradient-to-br from-emerald-500 to-teal-600"
              }`}
            >
              {isUser ? (
                <User className="w-4 h-4 text-white" />
              ) : (
                <Bot className="w-4 h-4 text-white" />
              )}
            </div>
            <div className="flex-1">
              <div className="flex items-center space-x-2 mb-1">
                <span className="font-semibold text-white text-sm">
                  {isUser ? "Вы" : "AI Scientometer"}
                </span>
                <span className="text-xs text-slate-400 flex items-center space-x-1">
                  <Clock className="w-3 h-3" />
                  <span>{message.timestamp.toLocaleTimeString()}</span>
                </span>
                {!isUser && message.aiResponse && (
                  <div className="flex items-center space-x-1 px-2 py-1 bg-emerald-500/20 border border-emerald-500/30 rounded-full">
                    <Target className="w-3 h-3 text-emerald-400" />
                    <span className="text-xs font-medium text-emerald-400">
                      {Math.round(message.aiResponse.confidence * 100)}%
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div
            className={`p-4 rounded-2xl ${
              isUser
                ? "bg-gradient-to-br from-blue-500 to-purple-600 text-white ml-8"
                : "bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 text-slate-100"
            }`}
          >
            <div
              className="text-sm leading-relaxed"
              dangerouslySetInnerHTML={{
                __html: formatAIResponse(message.content),
              }}
            />

            {/* Рекомендации */}
            {!isUser &&
              message.aiResponse?.recommendations &&
              message.aiResponse.recommendations.length > 0 && (
                <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-xl">
                  <div className="flex items-center space-x-2 mb-2">
                    <Lightbulb className="w-4 h-4 text-blue-400" />
                    <h4 className="font-semibold text-blue-400">
                      Рекомендации:
                    </h4>
                  </div>
                  <ul className="space-y-1 text-sm text-blue-300">
                    {message.aiResponse.recommendations.map((rec, index) => (
                      <li key={index} className="flex items-start space-x-2">
                        <span className="text-blue-400 mt-1">•</span>
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

            {/* Найденные статьи */}
            {!isUser &&
              message.aiResponse?.papers &&
              message.aiResponse.papers.length > 0 && (
                <div className="mt-4 p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-xl">
                  <div className="flex items-center space-x-2 mb-3">
                    <BookOpen className="w-4 h-4 text-emerald-400" />
                    <h4 className="font-semibold text-emerald-400">
                      Найденные статьи:
                    </h4>
                  </div>
                  <div className="space-y-3">
                    {message.aiResponse.papers.map((paper, index) => (
                      <div
                        key={index}
                        className="bg-slate-900/50 border border-slate-700/30 rounded-lg p-3"
                      >
                        <div className="font-medium text-white mb-1">
                          {paper.title}
                        </div>
                        <div className="text-sm text-slate-400 mb-1">
                          Авторы: {paper.authors?.join(", ")}
                        </div>
                        <div className="text-xs text-slate-500 flex items-center space-x-4">
                          <span>Год: {paper.year}</span>
                          <span>
                            Цитирований: {paper.citations?.toLocaleString()}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            {/* Рекомендованные журналы */}
            {!isUser &&
              message.aiResponse?.journals &&
              message.aiResponse.journals.length > 0 && (
                <div className="mt-4 p-3 bg-purple-500/10 border border-purple-500/20 rounded-xl">
                  <div className="flex items-center space-x-2 mb-3">
                    <Newspaper className="w-4 h-4 text-purple-400" />
                    <h4 className="font-semibold text-purple-400">
                      Рекомендованные журналы:
                    </h4>
                  </div>
                  <div className="space-y-3">
                    {message.aiResponse.journals.map((journal, index) => (
                      <div
                        key={index}
                        className="bg-slate-900/50 border border-slate-700/30 rounded-lg p-3"
                      >
                        <div className="font-medium text-white mb-1">
                          {journal.name}
                        </div>
                        <div className="text-sm text-slate-400 mb-1 flex items-center space-x-4">
                          <span>IF: {journal.impact_factor}</span>
                          <span>{journal.quartile}</span>
                          <span className="flex items-center space-x-1">
                            <TrendingUp className="w-3 h-3" />
                            <span>
                              Релевантность:{" "}
                              {Math.round(journal.relevance_score * 100)}%
                            </span>
                          </span>
                        </div>
                        <div className="text-xs text-slate-500">
                          {journal.reason}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="flex h-full bg-slate-900">
      {/* Mobile overlay */}
      {showSidebar && window.innerWidth <= 768 && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
          onClick={() => setShowSidebar(false)}
        />
      )}

      {/* Sidebar with topics */}
      <div
        className={`${
          showSidebar ? "translate-x-0" : "-translate-x-full"
        } fixed lg:relative lg:translate-x-0 w-80 h-full bg-slate-800/50 backdrop-blur-sm border-r border-slate-700/50 transition-transform duration-300 z-50 flex flex-col`}
      >
        <div className="p-4 border-b border-slate-700/50">
          <button
            className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-medium rounded-xl transition-all duration-200"
            onClick={startNewTopic}
          >
            <Plus className="w-4 h-4" />
            <span>Новый чат</span>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">
          <div className="text-sm font-medium text-slate-400 mb-3 uppercase tracking-wide">
            Недавние чаты
          </div>
          <div className="space-y-2">
            {topics.map((topic) => (
              <div
                key={topic.id}
                className={`p-3 rounded-xl cursor-pointer transition-all duration-200 ${
                  currentTopic === topic.id
                    ? "bg-blue-500/20 border border-blue-500/30 text-blue-400"
                    : "bg-slate-700/30 hover:bg-slate-700/50 text-slate-300 hover:text-white"
                }`}
                onClick={() => switchToTopic(topic.id)}
              >
                <div className="font-medium text-sm mb-1 line-clamp-2">
                  {topic.title}
                </div>
                <div className="text-xs text-slate-500 flex items-center space-x-2">
                  <MessageSquare className="w-3 h-3" />
                  <span>{topic.messageCount} сообщений</span>
                  <span>•</span>
                  <span>{topic.timestamp.toLocaleDateString()}</span>
                </div>
              </div>
            ))}

            {topics.length === 0 && (
              <div className="text-center py-8 text-slate-500">
                <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">Нет сохраненных чатов</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col bg-slate-900">
        <div className="flex items-center justify-between p-4 border-b border-slate-700/50 bg-slate-800/30 backdrop-blur-sm">
          <button
            className="lg:hidden p-2 text-slate-400 hover:text-white hover:bg-slate-700/50 rounded-lg transition-colors duration-200"
            onClick={() => setShowSidebar(!showSidebar)}
          >
            <Menu className="w-5 h-5" />
          </button>

          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-lg">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-white">AI Scientometer</h1>
              <p className="text-xs text-slate-400">Ваш научный ассистент</p>
            </div>
          </div>

          <button
            className="p-2 text-slate-400 hover:text-white hover:bg-slate-700/50 rounded-lg transition-colors duration-200"
            onClick={startNewTopic}
          >
            <Sparkles className="w-5 h-5" />
          </button>
        </div>

        <div className="flex-1 flex flex-col min-h-0">
          {/* Область сообщений */}
          <div className="flex-1 overflow-y-auto">
            <div className="p-6 space-y-6">
              {messages.map(renderMessage)}

              {isThinking && (
                <div className="flex justify-start">
                  <div className="max-w-3xl">
                    <div className="flex items-start space-x-3 mb-2">
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                        <Brain className="w-4 h-4 text-white animate-pulse" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <span className="font-semibold text-white text-sm">
                            AI Scientometer
                          </span>
                          <span className="text-xs text-slate-400">
                            думает...
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-4">
                      <div className="flex items-center space-x-3">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"></div>
                          <div
                            className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"
                            style={{ animationDelay: "0.1s" }}
                          ></div>
                          <div
                            className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"
                            style={{ animationDelay: "0.2s" }}
                          ></div>
                        </div>
                        <span className="text-sm text-emerald-400 font-medium">
                          {thinkingText}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {isLoading && !isThinking && (
                <div className="flex justify-start">
                  <div className="max-w-3xl">
                    <div className="flex items-start space-x-3 mb-2">
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center">
                        <Loader className="w-4 h-4 text-white animate-spin" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <span className="font-semibold text-white text-sm">
                            AI Scientometer
                          </span>
                          <span className="text-xs text-slate-400">
                            печатает...
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-4">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-slate-400 rounded-full animate-pulse"></div>
                        <div
                          className="w-2 h-2 bg-slate-400 rounded-full animate-pulse"
                          style={{ animationDelay: "0.2s" }}
                        ></div>
                        <div
                          className="w-2 h-2 bg-slate-400 rounded-full animate-pulse"
                          style={{ animationDelay: "0.4s" }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Предложения вопросов */}
              {showSuggestions && suggestions.length > 0 && (
                <div className="bg-slate-800/30 border border-slate-700/50 rounded-2xl p-6">
                  <div className="flex items-center space-x-2 mb-4">
                    <Lightbulb className="w-5 h-5 text-yellow-400" />
                    <h4 className="font-semibold text-white">
                      Примеры вопросов:
                    </h4>
                  </div>
                  <div className="space-y-4">
                    {suggestions.map((category, index) => (
                      <div key={index}>
                        <div className="text-sm font-medium text-slate-400 mb-2">
                          {category.category}
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {category.questions.map((question, qIndex) => (
                            <button
                              key={qIndex}
                              className="px-3 py-2 text-sm bg-slate-700/50 hover:bg-slate-600/50 text-slate-300 hover:text-white border border-slate-600/50 hover:border-slate-500 rounded-lg transition-all duration-200"
                              onClick={() => handleSuggestionClick(question)}
                            >
                              {question}
                            </button>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Закрепленное поле ввода */}
          <div className="flex-shrink-0 border-t border-slate-700/50 bg-slate-800/90 backdrop-blur-sm">
            <div className="p-6">
              <div className="flex items-end space-x-3 max-w-4xl mx-auto">
                <div className="flex-1">
                  <textarea
                    value={inputMessage}
                    onChange={handleInputChange}
                    onKeyPress={handleKeyPress}
                    placeholder="Задайте вопрос AI агенту... (Enter для отправки, Shift+Enter для новой строки)"
                    className="w-full px-4 py-3 bg-slate-700/50 border border-slate-600 rounded-2xl text-white placeholder-slate-400 resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 min-h-[48px] max-h-32"
                    rows={1}
                    disabled={isLoading}
                  />
                </div>
                <button
                  onClick={() => sendMessage(inputMessage)}
                  disabled={isLoading || !inputMessage.trim()}
                  className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:from-slate-600 disabled:to-slate-600 text-white rounded-2xl transition-all duration-200 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
                >
                  {isLoading ? (
                    <Loader className="w-5 h-5 animate-spin" />
                  ) : (
                    <Send className="w-5 h-5" />
                  )}
                </button>
              </div>

              <div className="flex flex-wrap justify-center gap-4 mt-3 text-xs text-slate-500 max-w-4xl mx-auto">
                <div className="flex items-center space-x-1">
                  <Lightbulb className="w-3 h-3" />
                  <span>Попробуйте: "Найди статьи про AI"</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Newspaper className="w-3 h-3" />
                  <span>Или: "Посоветуй журнал для публикации"</span>
                </div>
                <div className="flex items-center space-x-1">
                  <TrendingUp className="w-3 h-3" />
                  <span>Или: "Какие тренды в ML?"</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AIChat;
