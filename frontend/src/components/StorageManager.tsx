import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Database, FileText } from 'lucide-react';
import { getApiUrl, default as API_CONFIG } from '../config/api';

interface StorageManagerProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ProcessingResult {
  neo4j: {
    entities: number;
    relationships: number;
    entityTypes: string[];
  };
  weaviate: {
    documents: number;
    vectors: number;
    avgSimilarity: string;
  };
  processingTime?: string;
}

const StorageManager: React.FC<StorageManagerProps> = ({ isOpen, onClose }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStage, setProcessingStage] = useState('');
  const [processingResults, setProcessingResults] = useState<ProcessingResult | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any>(null);

  const [storageStats, setStorageStats] = useState<any>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const fetchStorageStats = useCallback(async () => {
    try {
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.STORAGE_STATS));
      
      if (response.ok) {
        const stats = await response.json();
        setStorageStats(stats);
      }
    } catch (error) {
      console.error('获取存储统计失败:', error);
    }
  }, []);

  const pollProcessingProgress = useCallback(async (taskId: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(getApiUrl(`${API_CONFIG.ENDPOINTS.STORAGE_PROGRESS}/${taskId}`));
        
        if (!response.ok) {
          throw new Error('获取进度失败');
        }
        
        const progress = await response.json();
        
        setProcessingStage(progress.current_stage);
        setUploadProgress(progress.progress);
        
        if (progress.status === 'completed') {
          clearInterval(pollInterval);
          setIsProcessing(false);
          setProcessingResults({
            neo4j: {
              entities: progress.details.neo4j_entities || 0,
              relationships: progress.details.neo4j_relationships || 0,
              entityTypes: ['疾病', '症状', '药物', '治疗方法']
            },
            weaviate: {
              documents: 1,
              vectors: progress.details.weaviate_vectors || 0,
              avgSimilarity: '0.89'
            },
            processingTime: progress.details.processing_time || '未知'
          });
          
          // 刷新存储统计
          fetchStorageStats();
        } else if (progress.status === 'failed') {
          clearInterval(pollInterval);
          setIsProcessing(false);
          alert(`处理失败: ${progress.message}`);
        }
        
      } catch (error) {
        console.error('获取处理进度失败:', error);
        clearInterval(pollInterval);
        setIsProcessing(false);
        alert('获取处理进度失败');
      }
    }, 1000); // 每秒轮询一次
  }, [fetchStorageStats]);

  const handleFileUpload = useCallback(async (files: FileList) => {
    if (files.length === 0) return;
    
    const file = files[0];
    const allowedTypes = ['text/plain', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    
    if (!allowedTypes.includes(file.type)) {
      alert('仅支持 TXT、PDF、DOCX 格式的文件');
      return;
    }
    
    setIsProcessing(true);
    setUploadProgress(0);
    setProcessingStage('上传中...');
    
    try {
      // 创建FormData对象
      const formData = new FormData();
      formData.append('file', file);
      
      // 上传文件
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.STORAGE_UPLOAD), {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || '上传失败');
      }
      
      const uploadResult = await response.json();
      
      // 开始轮询处理进度
      pollProcessingProgress(uploadResult.task_id);
      
    } catch (error) {
      console.error('文件上传失败:', error);
      alert(`上传失败: ${error instanceof Error ? error.message : '未知错误'}`);
      setIsProcessing(false);
      setUploadProgress(0);
      setProcessingStage('');
    }
  }, [pollProcessingProgress]);

  const handleFileSelect = (file: File) => {
    const allowedTypes = ['text/plain', 'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
    if (allowedTypes.includes(file.type)) {
      setSelectedFile(file);
    } else {
      alert('仅支持 TXT、PDF、DOCX 格式的文件');
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  // 移除旧的handleUpload函数，因为已经被handleFileUpload替代

  const handleSearch = useCallback(async () => {
    if (!searchQuery.trim()) return;
    
    try {
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.STORAGE_SEARCH_TEST), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          query: searchQuery,
          search_type: 'both'
        })
      });
      
      if (!response.ok) {
        throw new Error('搜索失败');
      }
      
      const results = await response.json();
      
      // 将后端返回的数据结构映射为前端期望的格式
      const mappedResults = {
        neo4j: results.neo4j_results || { entities: [], relationships: [] },
        weaviate: results.weaviate_results || { documents: [] }
      };
      
      setSearchResults(mappedResults);
      
    } catch (error) {
      console.error('搜索失败:', error);
      alert(`搜索失败: ${error instanceof Error ? error.message : '未知错误'}`);
    }
  }, [searchQuery]);

  const resetAll = useCallback(() => {
    setUploadProgress(0);
    setIsProcessing(false);
    setProcessingStage('');
    setProcessingResults(null);
    setSearchQuery('');
    setSearchResults(null);
    setStorageStats(null);
    fetchStorageStats();
  }, [fetchStorageStats]);
  
  // 组件加载时获取存储统计
  useEffect(() => {
    if (isOpen) {
      fetchStorageStats();
    }
  }, [isOpen, fetchStorageStats]);



  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content storage-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>存储管理</h2>
          <button className="modal-close" onClick={onClose}>
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M12 4L4 12M4 4L12 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
          </button>
        </div>

        <div className="storage-content">
          {/* 文档上传区域 */}
          <div className="upload-section">
            <h3>文档上传</h3>
            <div 
              className={`upload-area ${isDragging ? 'dragging' : ''} ${selectedFile ? 'has-file' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".txt,.pdf,.docx"
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                style={{ display: 'none' }}
              />
              {selectedFile ? (
                <div className="file-info">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" strokeWidth="2"/>
                    <polyline points="14,2 14,8 20,8" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                  <span>{selectedFile.name}</span>
                  <button 
                    className="remove-file"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedFile(null);
                    }}
                  >
                    ×
                  </button>
                </div>
              ) : (
                <div className="upload-placeholder">
                  <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
                    <path d="M24 8v24M12 20l12-12 12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <p>拖拽文件到此处或点击选择</p>
                  <p className="upload-hint">支持 TXT、PDF、DOCX 格式</p>
                </div>
              )}
            </div>
            
            {selectedFile && !isProcessing && (
               <button className="upload-btn" onClick={() => {
                 const fileList = new DataTransfer();
                 fileList.items.add(selectedFile);
                 handleFileUpload(fileList.files);
               }}>
                 开始处理
               </button>
             )}
          </div>

          {/* 处理进度 */}
          {isProcessing && (
            <div className="progress-section">
              <h3>处理进度</h3>
              <div className="progress-info">
                <div className="progress-stage">{processingStage}</div>
              </div>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              <div className="progress-percent">{uploadProgress}%</div>
            </div>
          )}

          {/* 处理结果 */}
          {processingResults && (
            <div className="results-section">
              <h3>处理结果</h3>
              <div className="storage-results-grid">
                <div className="storage-result-card">
                  <div className="storage-result-header">
                    <Database className="storage-result-icon" />
                    <h4>Neo4j 知识图谱</h4>
                  </div>
                  <div className="storage-result-stats">
                    <div className="stat-item">
                      <span className="stat-label">实体数量:</span>
                      <span className="stat-value">{processingResults.neo4j.entities}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">关系数量:</span>
                      <span className="stat-value">{processingResults.neo4j.relationships}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">实体类型:</span>
                      <span className="stat-value">{processingResults.neo4j.entityTypes.join(', ')}</span>
                    </div>
                    {processingResults.processingTime && (
                      <div className="stat-item">
                        <span className="stat-label">处理时间:</span>
                        <span className="stat-value">{processingResults.processingTime}</span>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="storage-result-card">
                  <div className="storage-result-header">
                    <FileText className="storage-result-icon" />
                    <h4>Weaviate 向量存储</h4>
                  </div>
                  <div className="storage-result-stats">
                    <div className="stat-item">
                      <span className="stat-label">文档数量:</span>
                      <span className="stat-value">{processingResults.weaviate.documents}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">向量数量:</span>
                      <span className="stat-value">{processingResults.weaviate.vectors}</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-label">平均相似度:</span>
                      <span className="stat-value">{processingResults.weaviate.avgSimilarity}</span>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* 显示总体存储统计 */}
              {storageStats && (
                <div className="storage-overall-stats">
                  <h4>总体存储统计</h4>
                  <div className="storage-results-grid">
                    <div className="storage-result-card">
                      <div className="storage-result-header">
                        <Database className="storage-result-icon" />
                        <h5>Neo4j 总计</h5>
                      </div>
                      <div className="storage-result-stats">
                        <div className="stat-item">
                          <span className="stat-label">总实体:</span>
                          <span className="stat-value">{storageStats.neo4j_stats?.entities || 0}</span>
                        </div>
                        <div className="stat-item">
                          <span className="stat-label">总关系:</span>
                          <span className="stat-value">{storageStats.neo4j_stats?.relationships || 0}</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="storage-result-card">
                      <div className="storage-result-header">
                        <FileText className="storage-result-icon" />
                        <h5>Weaviate 总计</h5>
                      </div>
                      <div className="storage-result-stats">
                        <div className="stat-item">
                          <span className="stat-label">总向量:</span>
                          <span className="stat-value">{storageStats.weaviate_stats?.vectors || 0}</span>
                        </div>
                        <div className="stat-item">
                          <span className="stat-label">总文档:</span>
                          <span className="stat-value">{storageStats.weaviate_stats?.documents || 0}</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 检索示例 */}
          <div className="search-section">
            <h3>检索示例</h3>
            <div className="search-input-container">
              <input
                type="text"
                placeholder="输入查询内容，如：帕金森"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="search-input"
              />
              <button className="search-btn" onClick={handleSearch}>
                搜索
              </button>
            </div>
            
            <div className="example-queries">
              <span>示例查询：</span>
              {['帕金森', '阿尔茨海默', '糖尿病'].map(query => (
                <button 
                  key={query}
                  className="example-query"
                  onClick={() => {
                    setSearchQuery(query);
                    handleSearch();
                  }}
                >
                  {query}
                </button>
              ))}
            </div>
          </div>

          {/* 搜索结果 */}
          {searchResults && (
            <div className="search-results">
              <h3>搜索结果</h3>
              <div className="results-grid">
                <div className="result-card neo4j-results">
                  <h4>Neo4j 图谱结果</h4>
                  <div className="entities-list">
                    <h5>相关实体</h5>
                    {searchResults.neo4j?.entities?.map((entity: any, index: number) => (
                      <div key={index} className="entity-item">
                        <span className="entity-name">{entity.name}</span>
                        <span className="entity-type">{entity.type}</span>
                      </div>
                    )) || []}
                  </div>
                  <div className="relationships-list">
                    <h5>关联关系</h5>
                    {searchResults.neo4j?.relationships?.map((rel: any, index: number) => (
                      <div key={index} className="relationship-item">
                        <span>{rel.source}</span>
                        <span className="relationship-type">{rel.type}</span>
                        <span>{rel.target}</span>
                      </div>
                    )) || []}
                  </div>
                </div>
                
                <div className="result-card weaviate-results">
                  <h4>Weaviate 文档结果</h4>
                  <div className="documents-list">
                    {searchResults.weaviate?.documents?.map((doc: any, index: number) => (
                      <div key={index} className="document-item">
                        <div className="document-content">{doc.content}</div>
                        <div className="document-score">相似度: {(doc.score * 100).toFixed(1)}%</div>
                      </div>
                    )) || []}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="modal-footer">
          <button className="btn-secondary" onClick={resetAll}>
            重置
          </button>
          <button className="btn-primary" onClick={onClose}>
            关闭
          </button>
        </div>
      </div>
    </div>
  );
};

export default StorageManager;