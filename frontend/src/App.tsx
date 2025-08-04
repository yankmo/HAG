import React, { useState, useRef, useEffect } from 'react';
import './App.css';

interface Message {
  id: number;
  text: string;
  isUser: boolean;
  timestamp: Date;
  sources?: any;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set());
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const toggleSourcesExpansion = (messageId: number) => {
    setExpandedSources(prev => {
      const newSet = new Set(prev);
      if (newSet.has(messageId)) {
        newSet.delete(messageId);
      } else {
        newSet.add(messageId);
      }
      return newSet;
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // 首次发送消息时展开界面
    if (!isExpanded) {
      setIsExpanded(true);
    }

    const userMessage: Message = {
      id: Date.now(),
      text: inputValue,
      isUser: true,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: inputValue }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      
      const botMessage: Message = {
        id: Date.now() + 1,
        text: data.answer || '抱歉，我无法处理您的请求。',
        isUser: false,
        timestamp: new Date(),
        sources: data.sources || {}
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage: Message = {
        id: Date.now() + 1,
        text: '抱歉，连接服务器时出现错误。请稍后再试。',
        isUser: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`app ${isExpanded ? 'expanded' : 'collapsed'}`}>
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
                      if (e.key === 'Enter' && !e.shiftKey) {
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
            <h1 className="chat-title">HAG</h1>
            <p className="chat-subtitle">智能知识问答助手</p>
          </div>

          <div className="messages-container">
            <div className="messages-list">
              {messages.map((message) => (
                <div key={message.id} className={`message ${message.isUser ? 'user' : 'assistant'}`}>
                  <div className="message-content">
                    <div className="message-text">{message.text}</div>
                    
                    {/* 参考来源 */}
                    {message.sources && (
                      <div className="message-sources">
                        <div 
                          className="sources-header"
                          onClick={() => toggleSourcesExpansion(message.id)}
                        >
                          <span className="sources-title">参考文档</span>
                          <span className={`expand-icon ${expandedSources.has(message.id) ? 'expanded' : ''}`}>
                            ▼
                          </span>
                        </div>
                        
                        {expandedSources.has(message.id) && (
                          <div className="sources-content">
                            {message.sources.documents && message.sources.documents.length > 0 && (
                              <div className="sources-section">
                                <div className="sources-section-title">参考文档：</div>
                                <div className="sources-list">
                                  {message.sources.documents.map((doc: any, index: number) => (
                                    <div key={index} className="source-item document">
                                      <div className="source-content">{doc.content}</div>
                                      <div className="source-score">相似度: {(doc.score * 100).toFixed(1)}%</div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                            
                            {message.sources.entities && message.sources.entities.length > 0 && (
                              <div className="sources-section">
                                <div className="sources-section-title">相关实体：</div>
                                <div className="sources-list">
                                  {message.sources.entities.map((entity: any, index: number) => (
                                    <div key={index} className="source-item entity">
                                      <span className="entity-name">{entity.name}</span>
                                      <span className="entity-type">({entity.type})</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                            
                            {message.sources.relationships && message.sources.relationships.length > 0 && (
                              <div className="sources-section">
                                <div className="sources-section-title">相关关系：</div>
                                <div className="sources-list">
                                  {message.sources.relationships.map((rel: any, index: number) => (
                                    <div key={index} className="source-item relationship">
                                      <span className="relationship-text">
                                        {rel.source} → {rel.target} ({rel.type})
                                      </span>
                                      {rel.description && (
                                        <div className="relationship-desc">{rel.description}</div>
                                      )}
                                    </div>
                                  ))}
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
              {isLoading && (
                <div className="message assistant">
                  <div className="message-content">
                    <div className="loading-indicator">
                      <div className="loading-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                      <span className="loading-text">正在思考...</span>
                    </div>
                  </div>
                </div>
              )}
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
                    if (e.key === 'Enter' && !e.shiftKey) {
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