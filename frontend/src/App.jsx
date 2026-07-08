import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import { Bot, User, Send, FileText, Plus, MessageSquare, Upload, X, Loader2, Link, MoreVertical, Edit2, Trash2, RefreshCw } from 'lucide-react';

const API_BASE = 'http://localhost:8002';

const PROCESSING_STEPS = [
  'Prepare', 'Check service',
  'Submit', 'Queue',
  'Parse', 'Download',
  'Build outputs', 'Done'
];

export default function App() {
  const [chats, setChats] = useState(() => {
    const saved = localStorage.getItem('pike_rag_chats');
    return saved ? JSON.parse(saved) : [{ id: '1', title: 'New Chat', messages: [] }];
  });
  const [currentChatId, setCurrentChatId] = useState(() => {
    const saved = localStorage.getItem('pike_rag_current_chat');
    return saved || '1';
  });
  const [input, setInput] = useState('');
  const [documents, setDocuments] = useState([]);
  const [viewingDoc, setViewingDoc] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const [uploading, setUploading] = useState(false);
  
  const messagesEndRef = useRef(null);

  const [dropdownId, setDropdownId] = useState(null);
  const [editingChatId, setEditingChatId] = useState(null);
  const [editingTitle, setEditingTitle] = useState('');
  const [chatToDelete, setChatToDelete] = useState(null);

  // Get the current chat object
  const currentChat = chats.find(c => c.id === currentChatId) || chats[0];

  useEffect(() => {
    localStorage.setItem('pike_rag_chats', JSON.stringify(chats));
  }, [chats]);

  useEffect(() => {
    localStorage.setItem('pike_rag_current_chat', currentChatId);
    fetchDocuments();
    setViewingDoc(null);
  }, [currentChatId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [currentChat.messages]);

  const fetchDocuments = async () => {
    try {
      const res = await axios.get(`${API_BASE}/documents?chat_id=${currentChatId}`);
      setDocuments(res.data.documents || []);
    } catch (err) {
      console.error("Failed to fetch documents", err);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    const tempMsgId = Date.now().toString();
    const loadingMessage = { 
      id: tempMsgId, 
      role: 'system', 
      content: `${file.name}`,
      isLoading: true,
      progress: 0,
      closable: false
    };
    
    setChats(prev => prev.map(c => {
      if (c.id === currentChatId) {
        return { ...c, messages: [...c.messages, loadingMessage] };
      }
      return c;
    }));

    const simInterval = setInterval(() => {
      setChats(prev => prev.map(c => {
        if (c.id === currentChatId) {
          return {
            ...c,
            messages: c.messages.map(m => (m.id === tempMsgId && m.progress < 4) ? { ...m, progress: m.progress + 1 } : m)
          };
        }
        return c;
      }));
    }, 1500);

    const formData = new FormData();
    formData.append('chat_id', currentChatId);
    formData.append('file', file);

    try {
      await axios.post(`${API_BASE}/upload`, formData);
      clearInterval(simInterval);
      
      await fetchDocuments();
      setViewingDoc(`${API_BASE}/documents/${currentChatId}/${file.name}`);
      
      setChats(prev => prev.map(c => {
        if (c.id === currentChatId) {
          return { 
            ...c, 
            title: c.title === 'New Chat' ? file.name : c.title,
            messages: c.messages.map(m => m.id === tempMsgId ? {
              ...m, 
              content: `✅ **${file.name}** successfully parsed and indexed! Ready for analysis.`,
              isLoading: false,
              closable: true
            } : m) 
          };
        }
        return c;
      }));
    } catch (err) {
      clearInterval(simInterval);
      console.error("Upload failed", err);
      setChats(prev => prev.map(c => {
        if (c.id === currentChatId) {
          return { 
            ...c, 
            messages: c.messages.map(m => m.id === tempMsgId ? {
              ...m, 
              content: `❌ Error parsing **${file.name}**. The server may have timed out or failed.`,
              isLoading: false,
              error: true,
              closable: true
            } : m) 
          };
        }
        return c;
      }));
    } finally {
      setUploading(false);
      e.target.value = '';
    }
  };

  const createNewChat = () => {
    const newId = Date.now().toString();
    setChats([{ id: newId, title: 'New Chat', messages: [] }, ...chats]);
    setCurrentChatId(newId);
  };

  const handleRenameSubmit = (chatId) => {
    setChats(chats.map(c => c.id === chatId ? { ...c, title: editingTitle } : c));
    setEditingChatId(null);
  };

  const handleDeleteChat = (chatId, e) => {
    e.stopPropagation();
    setChatToDelete(chatId);
    setDropdownId(null);
  };

  const confirmDeleteChat = async () => {
    if (!chatToDelete) return;
    try {
      await axios.delete(`${API_BASE}/chat/${chatToDelete}`);
      
      setChats(prev => {
        const newChats = prev.filter(c => c.id !== chatToDelete);
        const successMsg = { 
          id: Date.now().toString(), 
          role: 'system', 
          content: '✅ Chat and all associated files/images were completely deleted.', 
          closable: true 
        };
        
        if (newChats.length === 0) {
          const newId = Date.now().toString();
          setCurrentChatId(newId);
          setDocuments([]);
          setViewingDoc(null);
          return [{ id: newId, title: 'New Chat', messages: [successMsg] }];
        } else {
          const nextActiveId = currentChatId === chatToDelete ? newChats[0].id : currentChatId;
          setCurrentChatId(nextActiveId);
          if (currentChatId === chatToDelete) {
             setDocuments([]);
             setViewingDoc(null);
          }
          return newChats.map(c => c.id === nextActiveId ? {
            ...c,
            messages: [...c.messages, successMsg]
          } : c);
        }
      });
      
    } catch (err) {
      console.error("Delete failed", err);
    }
    setChatToDelete(null);
  };

  const handleRefreshChat = async (chatId, e) => {
    e.stopPropagation();
    setDropdownId(null);
    if (chatId !== currentChatId) {
      setCurrentChatId(chatId);
    }

    const tempMsgId = Date.now().toString();
    const loadingMessage = { 
      id: tempMsgId, 
      role: 'system', 
      content: `🔄 Refreshing and re-parsing all documents for this chat...`,
      isLoading: true
    };
    
    setChats(prev => prev.map(c => c.id === chatId ? { ...c, messages: [...c.messages, loadingMessage] } : c));

    try {
      await axios.post(`${API_BASE}/chat/${chatId}/refresh`);
      const docsRes = await axios.get(`${API_BASE}/documents?chat_id=${chatId}`);
      const docs = docsRes.data.documents || [];
      setDocuments(docs);
      
      let newTitle = docs.length > 0 ? docs[0] : null;
      if (newTitle && chatId === currentChatId) {
        setViewingDoc(`${API_BASE}/documents/${chatId}/${docs[0]}`);
      }
      
      setChats(prev => prev.map(c => {
        if (c.id === chatId) {
          return { 
            ...c, 
            title: newTitle || c.title,
            messages: c.messages.map(m => m.id === tempMsgId ? {
              ...m, 
              content: docs.length > 0 ? `✅ Documents successfully re-parsed and indexed!` : `⚠️ No documents found to refresh for this chat.`,
              isLoading: false,
              closable: true,
              error: docs.length === 0
            } : m) 
          };
        }
        return c;
      }));
    } catch (err) {
      console.error("Refresh failed", err);
      setChats(prev => prev.map(c => {
        if (c.id === chatId) {
          return { 
            ...c, 
            messages: c.messages.map(m => m.id === tempMsgId ? {
              ...m, 
              content: `❌ Error refreshing documents. Check server logs.`,
              isLoading: false,
              error: true,
              closable: true
            } : m) 
          };
        }
        return c;
      }));
    }
  };

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    const updatedChats = chats.map(c => {
      if (c.id === currentChatId) {
        // Automatically title the chat based on first message
        const title = c.messages.length === 0 ? input.substring(0, 30) + "..." : c.title;
        return { ...c, title, messages: [...c.messages, userMessage] };
      }
      return c;
    });
    
    setChats(updatedChats);
    setInput('');
    setIsTyping(true);

    try {
      const res = await axios.post(`${API_BASE}/query`, { question: userMessage.content, chat_id: currentChatId });
      
      setChats(prev => prev.map(c => {
        if (c.id === currentChatId) {
          return { ...c, messages: [...c.messages, { role: 'ai', content: res.data.answer }] };
        }
        return c;
      }));
    } catch (err) {
      console.error(err);
      setChats(prev => prev.map(c => {
        if (c.id === currentChatId) {
          return { ...c, messages: [...c.messages, { role: 'ai', content: '⚠️ Error connecting to PIKE-RAG engine.' }] };
        }
        return c;
      }));
    } finally {
      setIsTyping(false);
    }
  };



  const removeMessage = (chatId, msgId) => {
    setChats(prev => prev.map(c => {
      if (c.id === chatId) {
        return { ...c, messages: c.messages.filter(m => m.id !== msgId) };
      }
      return c;
    }));
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <div className="brand">
          <Bot size={28} /> PIKE-RAG
        </div>

        <button className="new-chat-btn" onClick={createNewChat}>
          <Plus size={20} /> New Chat
        </button>

        <div className="section-title">Recent Chats</div>
        <div className="item-list">
          {chats.map(chat => (
            <div 
              key={chat.id} 
              className={`list-item ${chat.id === currentChatId ? 'active' : ''}`}
              onClick={() => setCurrentChatId(chat.id)}
              style={{ position: 'relative' }}
            >
              <MessageSquare />
              
              {editingChatId === chat.id ? (
                <input 
                  autoFocus
                  value={editingTitle}
                  onChange={(e) => setEditingTitle(e.target.value)}
                  onBlur={() => handleRenameSubmit(chat.id)}
                  onKeyDown={(e) => { if (e.key === 'Enter') handleRenameSubmit(chat.id) }}
                  style={{ background: 'transparent', border: '1px solid var(--primary)', color: 'white', flex: 1, padding: '2px 4px', borderRadius: '4px' }}
                />
              ) : (
                <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', flex: 1 }}>
                  {chat.title}
                </span>
              )}

              <button 
                className="chat-options-btn"
                onClick={(e) => { e.stopPropagation(); setDropdownId(dropdownId === chat.id ? null : chat.id); }}
              >
                <MoreVertical size={16} />
              </button>

              {dropdownId === chat.id && (
                <div className="chat-dropdown">
                  <div onClick={(e) => { e.stopPropagation(); setEditingChatId(chat.id); setEditingTitle(chat.title); setDropdownId(null); }}>
                    <Edit2 size={14} /> Rename
                  </div>
                  <div onClick={(e) => handleRefreshChat(chat.id, e)}>
                    <RefreshCw size={14} /> Refresh
                  </div>
                  <div onClick={(e) => handleDeleteChat(chat.id, e)} className="danger">
                    <Trash2 size={14} /> Delete
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="section-title">Knowledge Base</div>
        <div className="item-list" style={{ flex: 0.5 }}>
          {documents.map((doc, idx) => (
            <div 
              key={idx} 
              className="list-item"
              onClick={() => setViewingDoc(`${API_BASE}/documents/${currentChatId}/${doc}`)}
            >
              <FileText />
              <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {doc}
              </span>
            </div>
          ))}
        </div>

      </div>

      {/* Main Content Area */}
      <div className="main-content">
        {/* Chat Interface */}
        <div className="chat-area">
          <div className="messages-container">
            {currentChat.messages.length === 0 && (
              <div style={{ textAlign: 'center', margin: 'auto', color: 'var(--text-muted)' }}>
                <Bot size={48} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
                <h2>How can I assist you with your industrial equipment today?</h2>
                <p>Upload a manual or SOP to get started.</p>
              </div>
            )}
            
            {currentChat.messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.role} ${msg.isLoading ? 'loading' : ''} ${msg.error ? 'error' : ''}`}>
                {msg.role !== 'system' && (
                  <div className={`avatar ${msg.role}`}>
                    {msg.role === 'ai' ? <Bot size={20} /> : <User size={20} />}
                  </div>
                )}
                <div className={`message-content ${msg.role === 'system' ? 'system-msg-box' : 'markdown-body'}`}>
                  {msg.role === 'system' && msg.closable && (
                    <button className="close-system-btn" onClick={() => removeMessage(currentChatId, msg.id)}>
                      <X size={14} />
                    </button>
                  )}
                  
                  {msg.isLoading ? (
                    <div style={{ width: '100%', minWidth: '300px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', justifyContent: 'center' }}>
                        <Loader2 className="animate-spin" size={18} style={{ color: 'var(--primary)' }} />
                        <span style={{ fontWeight: '500', color: 'white' }}>Processing {msg.content}</span>
                      </div>
                      <div className="processing-grid">
                        {PROCESSING_STEPS.map((step, i) => {
                          let statusClass = 'pending';
                          if (i < msg.progress) statusClass = 'completed';
                          else if (i === msg.progress) statusClass = 'active';
                          
                          return (
                            <div key={i} className={`status-item ${statusClass}`}>
                              <div className={`status-dot ${statusClass}`}></div>
                              {step}
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ) : (
                    <div className={msg.role === 'system' ? 'system-msg' : ''}>
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  )}
                </div>
              </div>
            ))}
            
            {isTyping && (
              <div className="message ai">
                <div className="avatar ai"><Bot size={20} /></div>
                <div className="message-content">
                  <div className="typing-indicator">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-area">
            <form className="input-box" onSubmit={sendMessage}>
              <label className="chat-upload-btn" title="Upload Document">
                {uploading ? <Loader2 className="animate-spin" size={18} /> : <Plus size={20} />}
                <input type="file" hidden accept=".pdf,.docx,.txt" onChange={handleFileUpload} />
              </label>
              <input 
                type="text" 
                placeholder="Ask PIKE-RAG about your equipment..." 
                value={input}
                onChange={(e) => setInput(e.target.value)}
                disabled={isTyping}
              />
              <button type="submit" className="send-btn" disabled={!input.trim() || isTyping}>
                <Send size={18} />
              </button>
            </form>
          </div>
        </div>

        {/* Document Viewer Split Panel */}
        {viewingDoc && (
          <div className="doc-viewer">
            <div className="doc-header">
              <h3><FileText size={18} className="text-primary" /> Document Viewer</h3>
              <button className="close-doc-btn" onClick={() => setViewingDoc(null)}>
                <X size={20} />
              </button>
            </div>
            <iframe 
              src={viewingDoc} 
              className="doc-iframe"
              title="Document Viewer"
            />
          </div>
        )}
      </div>
      
      {chatToDelete && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h3>Delete Chat?</h3>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', marginBottom: '24px' }}>
              Are you sure you want to delete this chat? All uploaded PDFs, extracted images, and parsed data associated with this chat will be permanently erased.
            </p>
            <div className="modal-actions">
              <button className="btn-secondary" onClick={() => setChatToDelete(null)}>Cancel</button>
              <button className="btn-primary" style={{ background: '#ef4444' }} onClick={confirmDeleteChat}>Delete Permanently</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
