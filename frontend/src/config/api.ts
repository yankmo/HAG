// API配置文件
const API_CONFIG = {
  // 从环境变量获取API基础URL，如果没有则使用默认值
  BASE_URL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
  
  // API端点
  ENDPOINTS: {
    QUERY_STREAM: '/query/stream',
    QUERY: '/query'
  }
};

// 获取完整的API URL
export const getApiUrl = (endpoint: string): string => {
  return `${API_CONFIG.BASE_URL}${endpoint}`;
};

// 导出配置
export default API_CONFIG;