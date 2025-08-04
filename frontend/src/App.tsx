import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

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

    // 添加用户消息
    const userMessage: Message = {
      id: Date.now(),
      text: inputValue,
      isUser: true,
      timestamp: new Date()
    };

    // 创建助手消息（初始状态为思考中）
    const botMessageId = Date.now() + 1;
    const botMessage: Message = {
      id: botMessageId,
      text: '',
      isUser: false,
      timestamp: new Date(),
      isStreaming: true
    };

    setMessages(prev => [...prev, userMessage, botMessage]);
    const question = inputValue;
    setInputValue('');
    setIsLoading(true);
    setIsExpanded(true);

    try {
      const response = await fetch('http://localhost:8000/query/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });

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
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'content') {
                  // 更新消息内容
                  setMessages(prev => prev.map(msg => 
                    msg.id === botMessageId 
                      ? { ...msg, text: data.content, isStreaming: false }
                      : msg
                  ));
                } else if (data.type === 'sources') {
                  // 添加来源信息
                  setMessages(prev => prev.map(msg => 
                    msg.id === botMessageId 
                      ? { ...msg, sources: data.sources, isStreaming: true }
                      : msg
                  ));
                } else if (data.type === 'done') {
                  // 完成流式响应
                  setMessages(prev => prev.map(msg => 
                    msg.id === botMessageId 
                      ? { ...msg, isStreaming: false }
                      : msg
                  ));
                } else if (data.type === 'error') {
                  // 处理错误
                  setMessages(prev => prev.map(msg => 
                    msg.id === botMessageId 
                      ? { ...msg, text: data.message, isStreaming: false }
                      : msg
                  ));
                }
              } catch (parseError) {
                console.error('解析流数据失败:', parseError);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => prev.map(msg => 
        msg.id === botMessageId 
          ? { 
              ...msg, 
              text: '抱歉，连接服务器时出现错误。请稍后再试。',
              isStreaming: false
            }
          : msg
      ));
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
                    <div className="message-text">
                      {message.isUser ? (
                        message.text
                      ) : (
                        <ReactMarkdown 
                          remarkPlugins={[remarkGfm]}
                          components={{
                            // 自定义代码块样式
                            code: ({node, className, children, ...props}: any) => {
                              const match = /language-(\w+)/.exec(className || '');
                              // 如果有语言标识或者在pre标签内，则为代码块
                              const isCodeBlock = match || (node?.parent?.tagName === 'pre');
                              
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
                            table: ({children}) => (
                              <div className="table-wrapper">
                                <table className="markdown-table">{children}</table>
                              </div>
                            ),
                            // 自定义链接样式
                            a: ({children, href}) => (
                              <a href={href} target="_blank" rel="noopener noreferrer" className="markdown-link">
                                {children}
                              </a>
                            )
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