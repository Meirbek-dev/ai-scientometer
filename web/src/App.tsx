import { useState } from "react";
import Dashboard from "./components/Dashboard";
import AIChat from "./components/AIChat";
import { Brain, BarChart3, MessageSquare } from "lucide-react";
import "./App.css";

type ActivePage = "dashboard" | "chat";

function App() {
  const [activePage, setActivePage] = useState<ActivePage>("dashboard");

  const renderPage = () => {
    switch (activePage) {
      case "dashboard":
        return <Dashboard />;
      case "chat":
        return (
          <div className="h-full">
            <AIChat />
          </div>
        );
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Navigation Header */}
      <header className="bg-slate-800/90 backdrop-blur-sm border-b border-slate-700/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center justify-between h-16">
            {/* Brand */}
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                  AI Scientometer
                </h1>
                <p className="text-xs text-slate-400">
                  Intelligent Research Analytics
                </p>
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex items-center space-x-2">
              <button
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                  activePage === "dashboard"
                    ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                    : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                }`}
                onClick={() => setActivePage("dashboard")}
              >
                <BarChart3 className="w-4 h-4" />
                <span>Dashboard</span>
              </button>
              <button
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                  activePage === "chat"
                    ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                    : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                }`}
                onClick={() => setActivePage("chat")}
              >
                <MessageSquare className="w-4 h-4" />
                <span>AI Chat</span>
              </button>
            </nav>

            {/* Status */}
            <div className="flex items-center space-x-2 px-3 py-1.5 bg-green-500/10 border border-green-500/20 rounded-full">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm font-medium text-green-400">Live</span>
            </div>
          </div>
        </div>
      </header>

      {/* Page Content */}
      <main className="flex-1 overflow-hidden">{renderPage()}</main>
    </div>
  );
}

export default App;
