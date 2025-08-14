import React, { useState, useRef, useEffect, FormEvent } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getApiUrl, default as API_CONFIG } from "./config/api";
import "./App.css";

interface Message {
  id: number;
  text: string;
  isUser: boolean;
  timestamp: Date;
  isStreaming?: boolean;
  sources?: {
    documents?: Array<{
      content: string;
      score: number;
    }>;
    entities?: Array<{
      name: string;
      type: string;
    }>;
    relationships?: Array<{
      source: string;
      target: string;
      type: string;
      description?: string;
    }>;
  };
}

interface Session {
  id: string;
  title: string;
  timestamp: Date;
  messages: Message[];
}

// localStorage工具函数
const STORAGE_KEYS = {
  SESSIONS: 'hag_sessions',
  CURRENT_SESSION: 'hag_current_session'
};

const saveSessionsToStorage = (sessions: Session[]) => {
  try {
    const serializedSessions = sessions.map(session => ({
      ...session,
      timestamp: session.timestamp.toISOString(),
      messages: session.messages.map(msg => ({
        ...msg,
        timestamp: msg.timestamp.toISOString()
      }))
    }));
    localStorage.setItem(STORAGE_KEYS.SESSIONS, JSON.stringify(serializedSessions));
  } catch (error) {
    console.error('保存会话数据失败:', error);
  }
};

const loadSessionsFromStorage = (): Session[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEYS.SESSIONS);
    if (!stored) return [];
    
    const parsed = JSON.parse(stored);
    return parsed.map((session: any) => ({
      ...session,
      timestamp: new Date(session.timestamp),
      messages: session.messages.map((msg: any) => ({
        ...msg,
        timestamp: new Date(msg.timestamp)
      }))
    }));
  } catch (error) {
    console.error('加载会话数据失败:', error);
    return [];
  }
};

const saveCurrentSessionToStorage = (sessionId: string | null) => {
  try {
    if (sessionId) {
      localStorage.setItem(STORAGE_KEYS.CURRENT_SESSION, sessionId);
    } else {
      localStorage.removeItem(STORAGE_KEYS.CURRENT_SESSION);
    }
  } catch (error) {
    console.error('保存当前会话ID失败:', error);
  }
};

const loadCurrentSessionFromStorage = (): string | null => {
  try {
    return localStorage.getItem(STORAGE_KEYS.CURRENT_SESSION);
  } catch (error) {
    console.error('加载当前会话ID失败:', error);
    return null;
  }
};

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Set<number>>(
    new Set(),
  );
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sidebarExpanded, setSidebarExpanded] = useState(false);
  const [sidebarHovered, setSidebarHovered] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 初始化加载会话数据
  useEffect(() => {
    const loadedSessions = loadSessionsFromStorage();
    setSessions(loadedSessions);
    
    const currentSessionId = loadCurrentSessionFromStorage();
    if (currentSessionId && loadedSessions.length > 0) {
      const currentSession = loadedSessions.find(s => s.id === currentSessionId);
      if (currentSession) {
        setSessionId(currentSessionId);
        setMessages(currentSession.messages);
        if (currentSession.messages.length > 0) {
          setIsExpanded(true);
        }
      }
    }
  }, []);

  // 保存会话数据到localStorage
  useEffect(() => {
    if (sessions.length > 0) {
      saveSessionsToStorage(sessions);
    }
  }, [sessions]);

  // 保存当前会话ID
  useEffect(() => {
    saveCurrentSessionToStorage(sessionId);
  }, [sessionId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const toggleSourcesExpansion = (messageId: number) => {
    setExpandedSources((prev: Set<number>) => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
      }
      return newSet;
    });
  };

  const clearSession = () => {
    setMessages([]);
    setSessionId(null);
    setIsExpanded(false);
    setExpandedSources(new Set());
  };

  // 保存当前会话的消息到sessions中
  const saveCurrentSession = () => {
    if (sessionId && messages.length > 0) {
      setSessions(prevSessions => {
        const existingSessionIndex = prevSessions.findIndex(s => s.id === sessionId);
        if (existingSessionIndex >= 0) {
          // 更新现有会话
          const updatedSessions = [...prevSessions];
          updatedSessions[existingSessionIndex] = {
            ...updatedSessions[existingSessionIndex],
            messages: [...messages],
            timestamp: new Date()
          };
          return updatedSessions;
        } else {
          // 创建新会话（如果不存在）
          const firstUserMessage = messages.find(m => m.isUser);
          const title = firstUserMessage ? 
            (firstUserMessage.text.length > 30 ? 
              firstUserMessage.text.substring(0, 30) + '...' : 
              firstUserMessage.text) : 
            '新对话';
          
          return [{
            id: sessionId,
            title,
            timestamp: new Date(),
            messages: [...messages]
          }, ...prevSessions];
        }
      });
    }
  };

  const toggleSidebar = () => {
    setSidebarExpanded(!sidebarExpanded);
  };

  const handleSidebarMouseEnter = () => {
    setSidebarHovered(true);
  };

  const handleSidebarMouseLeave = () => {
    setSidebarHovered(false);
  };

  const createNewSession = () => {
    // 保存当前会话
    saveCurrentSession();
    
    const newSessionId = Date.now().toString();
    clearSession();
    setSessionId(newSessionId);
    setSidebarExpanded(false);
  };

  const switchToSession = (targetSessionId: string) => {
    // 保存当前会话
    saveCurrentSession();
    
    // 切换到目标会话
    const targetSession = sessions.find(s => s.id === targetSessionId);
    if (targetSession) {
      setSessionId(targetSessionId);
      setMessages(targetSession.messages);
      setIsExpanded(targetSession.messages.length > 0);
    }
    setSidebarExpanded(false);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // 如果没有当前会话ID，创建新会话
    let currentSessionId = sessionId;
    if (!currentSessionId) {
      currentSessionId = Date.now().toString();
      setSessionId(currentSessionId);
    }

    // 添加用户消息
    const userMessage: Message = {
      id: Date.now(),
      text: inputValue,
      isUser: true,
      timestamp: new Date(),
    };

    // 创建助手消息（初始状态为思考中）
    const botMessageId = Date.now() + 1;
    const botMessage: Message = {
      id: botMessageId,
      text: "",
      isUser: false,
      timestamp: new Date(),
      isStreaming: true,
    };

    setMessages((prev: Message[]) => [...prev, userMessage, botMessage]);
    const question = inputValue;
    setInputValue("");
    setIsLoading(true);
    setIsExpanded(true);

    try {
      const response = await fetch(
        getApiUrl(API_CONFIG.ENDPOINTS.QUERY_STREAM),
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question,
            session_id: sessionId,
            include_history: true,
          }),
        },
      );

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value);
          const lines = chunk.split("\n");

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));

                if (data.type === "session") {
                  // 保存会话ID
                  if (!sessionId) {
                    setSessionId(data.session_id);
                  }
                } else if (data.type === "content") {
                  // 更新消息内容
                  setMessages((prev: Message[]) =>
                    prev.map((msg: Message) =>
                      msg.id === botMessageId
                        ? { ...msg, text: data.content, isStreaming: false }
                        : msg,
                    ),
                  );
                } else if (data.type === "sources") {
                  // 添加来源信息
                  setMessages((prev: Message[]) =>
                    prev.map((msg: Message) =>
                      msg.id === botMessageId
                        ? { ...msg, sources: data.sources, isStreaming: true }
                        : msg,
                    ),
                  );
                } else if (data.type === "done") {
                  // 完成流式响应
                  setMessages((prev: Message[]) =>
                    prev.map((msg: Message) =>
                      msg.id === botMessageId
                        ? { ...msg, isStreaming: false }
                        : msg,
                    ),
                  );
                } else if (data.type === "error") {
                  // 处理错误
                  setMessages((prev: Message[]) =>
                    prev.map((msg: Message) =>
                      msg.id === botMessageId
                        ? { ...msg, text: data.message, isStreaming: false }
                        : msg,
                    ),
                  );
                }
              } catch (parseError) {
                // 解析流数据失败，跳过此行
              }
            }
          }
        }
      }
    } catch (error) {
      // 处理网络错误
      setMessages((prev: Message[]) =>
        prev.map((msg: Message) =>
          msg.id === botMessageId
            ? {
                ...msg,
                text: "抱歉，连接服务器时出现错误。请稍后再试。",
                isStreaming: false,
              }
            : msg,
        ),
      );
    } finally {
      setIsLoading(false);
    }
    
    // 保存当前会话（延迟执行以确保消息已更新）
    setTimeout(() => {
      saveCurrentSession();
    }, 100);
  };

  return (
    <div className={`app ${isExpanded ? "expanded" : "collapsed"}`}>
      {/* 侧边栏悬停触发区域 */}
      <div 
        className="sidebar-hover-zone"
        onMouseEnter={handleSidebarMouseEnter}
        onMouseLeave={handleSidebarMouseLeave}
      />
      
      {/* Linear风格会话管理侧边栏 */}
      <div className={`linear-sidebar ${sidebarExpanded ? 'expanded' : ''} ${sidebarHovered ? 'hovered' : ''}`}>
        {/* 侧边栏触发器 */}
        <div 
          className="sidebar-trigger"
          onClick={toggleSidebar}
          onMouseEnter={handleSidebarMouseEnter}
          onMouseLeave={handleSidebarMouseLeave}
        >
          <div className="trigger-arrow">
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M4 2L8 6L4 10" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
        </div>
        
        {/* 侧边栏内容 */}
        <div className="sidebar-content">
          <div className="sidebar-header">
            <h3 className="sidebar-title">会话管理</h3>
            <button className="new-session-btn" onClick={createNewSession}>
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M8 2V14M2 8H14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
              新对话
            </button>
          </div>
          
          <div className="sessions-list">
            {sessions.length === 0 ? (
              <div className="empty-sessions">
                <p>暂无会话记录</p>
              </div>
            ) : (
              sessions.map((session) => (
                <div 
                  key={session.id} 
                  className={`session-item ${sessionId === session.id ? 'active' : ''}`}
                  onClick={() => switchToSession(session.id)}
                >
                  <div className="session-title">{session.title}</div>
                  <div className="session-time">{session.timestamp.toLocaleDateString()}</div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
      {!isExpanded ? (
        // 折叠状态 - 初始界面
        <div className="initial-screen">
          <div className="initial-content">
            <div className="logo">
              <h1 className="logo-text">HAG</h1>
              <p className="logo-subtitle">智能知识问答助手</p>
            </div>

            <div className="initial-input-container">
              <form onSubmit={handleSubmit} className="initial-input-form">
                <div className="initial-input-wrapper">
                  <textarea
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    placeholder="请输入您的问题..."
                    className="initial-message-input"
                    rows={3}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSubmit(e);
                      }
                    }}
                  />
                  <button
                    type="submit"
                    disabled={!inputValue.trim() || isLoading}
                    className="initial-send-button"
                  >
                    发送
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      ) : (
        // 展开状态 - 完整聊天界面
        <div className="chat-screen">
          <div className="chat-header">
            <div className="header-left">
              <h1 className="chat-title">HAG</h1>
              <p className="chat-subtitle">智能知识问答助手</p>
            </div>
          </div>

          <div className="messages-container">
            <div className="messages-list">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`message ${message.isUser ? "user" : "assistant"}`}
                >
                  <div className="message-content">
                    <div className="message-text">
                      {message.isUser ? (
                        message.text
                      ) : (
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            // 自定义代码块样式
                            code: ({
                              node,
                              className,
                              children,
                              ...props
                            }: any) => {
                              const match = /language-(\w+)/.exec(
                                className || "",
                              );
                              // 如果有语言标识或者在pre标签内，则为代码块
                              const isCodeBlock =
                                match || node?.parent?.tagName === "pre";

                              if (isCodeBlock) {
                                return (
                                  <pre className="code-block">
                                    <code className={className} {...props}>
                                      {children}
                                    </code>
                                  </pre>
                                );
                              } else {
                                return (
                                  <code className="inline-code" {...props}>
                                    {children}
                                  </code>
                                );
                              }
                            },
                            // 自定义表格样式
                            table: ({ children }) => (
                              <div className="table-wrapper">
                                <table className="markdown-table">
                                  {children}
                                </table>
                              </div>
                            ),
                            // 自定义链接样式
                            a: ({ children, href }) => (
                              <a
                                href={href}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="markdown-link"
                              >
                                {children}
                              </a>
                            ),
                          }}
                        >
                          {message.text}
                        </ReactMarkdown>
                      )}
                      {message.isStreaming && (
                        <span className="thinking-indicator">
                          思考中<span className="thinking-dots">...</span>
                        </span>
                      )}
                    </div>

                    {/* 参考来源 */}
                    {message.sources && (
                      <div className="message-sources">
                        <div
                          className="sources-header"
                          onClick={() => toggleSourcesExpansion(message.id)}
                        >
                          <span className="sources-title">参考文档</span>
                          <span
                            className={`expand-icon ${expandedSources.has(message.id) ? "expanded" : ""}`}
                          >
                            ▼
                          </span>
                        </div>

                        {expandedSources.has(message.id) && (
                          <div className="sources-content">
                            {message.sources.documents &&
                              message.sources.documents.length > 0 && (
                                <div className="sources-section">
                                  <div className="sources-section-title">
                                    参考文档：
                                  </div>
                                  <div className="sources-list">
                                    {message.sources.documents.map(
                                      (doc: any, index: number) => (
                                        <div
                                          key={index}
                                          className="source-item document"
                                        >
                                          <div className="source-content">
                                            {doc.content}
                                          </div>
                                          <div className="source-score">
                                            相似度:{" "}
                                            {(doc.score * 100).toFixed(1)}%
                                          </div>
                                        </div>
                                      ),
                                    )}
                                  </div>
                                </div>
                              )}

                            {message.sources.entities &&
                              message.sources.entities.length > 0 && (
                                <div className="sources-section">
                                  <div className="sources-section-title">
                                    相关实体：
                                  </div>
                                  <div className="sources-list">
                                    {message.sources.entities.map(
                                      (entity: any, index: number) => (
                                        <div
                                          key={index}
                                          className="source-item entity"
                                        >
                                          <span className="entity-name">
                                            {entity.name}
                                          </span>
                                          <span className="entity-type">
                                            ({entity.type})
                                          </span>
                                        </div>
                                      ),
                                    )}
                                  </div>
                                </div>
                              )}

                            {message.sources.relationships &&
                              message.sources.relationships.length > 0 && (
                                <div className="sources-section">
                                  <div className="sources-section-title">
                                    相关关系：
                                  </div>
                                  <div className="sources-list">
                                    {message.sources.relationships.map(
                                      (rel: any, index: number) => (
                                        <div
                                          key={index}
                                          className="source-item relationship"
                                        >
                                          <span className="relationship-text">
                                            {rel.source} → {rel.target} (
                                            {rel.type})
                                          </span>
                                          {rel.description && (
                                            <div className="relationship-desc">
                                              {rel.description}
                                            </div>
                                          )}
                                        </div>
                                      ),
                                    )}
                                  </div>
                                </div>
                              )}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                  <div className="message-time">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              ))}

              <div ref={messagesEndRef} />
            </div>
          </div>

          <div className="input-container">
            <form onSubmit={handleSubmit} className="input-form">
              <div className="input-wrapper">
                <textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="请输入您的问题..."
                  className="message-input"
                  rows={2}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit(e);
                    }
                  }}
                />
                <button
                  type="submit"
                  disabled={!inputValue.trim() || isLoading}
                  className="send-button"
                >
                  发送
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
