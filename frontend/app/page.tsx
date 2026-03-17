"use client"

import { useState, useEffect, useRef } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

type Message = {
  role: "user" | "assistant"
  content: string
}

type Chat = {
  id: string
  title: string
  messages: Message[]
}

export default function Home() {

  const [chats, setChats] = useState<Chat[]>([])
  const [currentChatId, setCurrentChatId] = useState<string | null>(null)
  const [input, setInput] = useState("")
  const [loading, setLoading] = useState(false)

  const bottomRef = useRef<HTMLDivElement>(null)

  const currentChat = chats.find(c => c.id === currentChatId)

  // Load saved chats
  useEffect(() => {

    const saved = localStorage.getItem("research_chats")

    if (saved) {
      const parsed = JSON.parse(saved)
      setChats(parsed)

      if (parsed.length > 0) {
        setCurrentChatId(parsed[0].id)
      }
    }

  }, [])

  // Save chats
  useEffect(() => {

    localStorage.setItem("research_chats", JSON.stringify(chats))
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })

  }, [chats])

  function newChat() {

    const chat: Chat = {
      id: Date.now().toString(),
      title: "New Chat",
      messages: []
    }

    setChats(prev => [chat, ...prev])
    setCurrentChatId(chat.id)

  }

  async function sendMessage() {

    if (!input.trim() || !currentChat) return

    const userMessage: Message = {
      role: "user",
      content: input
    }

    updateMessages([...currentChat.messages, userMessage])

    setInput("")
    setLoading(true)

    try {

      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          question: input
        })
      })

      const reader = res.body?.getReader()
      const decoder = new TextDecoder()

      let botText = ""

      // Add empty assistant message first
      updateMessages([
        ...currentChat.messages,
        userMessage,
        { role: "assistant", content: "" }
      ])

      while (true) {

        const { done, value } = await reader!.read()

        if (done) break

        botText += decoder.decode(value)

        setChats(prev =>
          prev.map(chat =>
            chat.id === currentChatId
              ? {
                  ...chat,
                  messages: chat.messages.map((m, i) =>
                    i === chat.messages.length - 1
                      ? { ...m, content: botText }
                      : m
                  )
                }
              : chat
          )
        )

      }

    } catch {

      updateMessages([
        ...currentChat.messages,
        userMessage,
        { role: "assistant", content: "⚠️ Error contacting AI." }
      ])

    }

    setLoading(false)

  }

  function updateMessages(messages: Message[]) {

    setChats(prev =>
      prev.map(chat =>
        chat.id === currentChatId
          ? {
              ...chat,
              title:
                chat.title === "New Chat"
                  ? messages[0]?.content.slice(0, 40)
                  : chat.title,
              messages
            }
          : chat
      )
    )

  }

  return (

    <div style={{
      display: "flex",
      height: "100vh",
      background: "#111",
      color: "white"
    }}>

      {/* Sidebar */}

      <div style={{
        width: "260px",
        background: "#1a1a1a",
        padding: "20px",
        borderRight: "1px solid #333"
      }}>

        <h2>Research AI</h2>

        <button
          onClick={newChat}
          style={{
            marginTop: "20px",
            padding: "10px",
            width: "100%",
            background: "#333",
            border: "none",
            color: "white",
            borderRadius: "6px",
            cursor: "pointer"
          }}
        >
          + New Chat
        </button>

        <div style={{ marginTop: "30px" }}>

          <div style={{
            fontSize: "12px",
            color: "#aaa",
            marginBottom: "10px"
          }}>
            Previous Chats
          </div>

          {chats.map(chat => (

            <div
              key={chat.id}
              onClick={() => setCurrentChatId(chat.id)}
              style={{
                padding: "8px",
                cursor: "pointer",
                borderRadius: "6px",
                background:
                  chat.id === currentChatId
                    ? "#333"
                    : "transparent",
                marginBottom: "6px"
              }}
            >
              {chat.title}
            </div>

          ))}

        </div>

      </div>

      {/* Chat Area */}

      <div style={{
        flex: 1,
        display: "flex",
        flexDirection: "column"
      }}>

        {/* Messages */}

        <div style={{
          flex: 1,
          overflowY: "auto",
          padding: "30px"
        }}>

          {currentChat?.messages.map((msg, index) => (

            <div
              key={index}
              style={{
                display: "flex",
                justifyContent:
                  msg.role === "user" ? "flex-end" : "flex-start",
                marginBottom: "18px"
              }}
            >

              <div
                style={{
                  maxWidth: "70%",
                  padding: "12px 16px",
                  borderRadius: "10px",
                  background:
                    msg.role === "user"
                      ? "#2563eb"
                      : "#2a2a2a"
                }}
              >

                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {msg.content}
                </ReactMarkdown>

              </div>

            </div>

          ))}

          {loading && (
            <div style={{ color: "#aaa" }}>
              Assistant is typing...
            </div>
          )}

          <div ref={bottomRef} />

        </div>

        {/* Input */}

        <div style={{
          padding: "20px",
          borderTop: "1px solid #333",
          display: "flex",
          gap: "10px"
        }}>

          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a research question..."
            style={{
              flex: 1,
              padding: "12px",
              borderRadius: "8px",
              border: "none"
            }}
          />

          <button
            onClick={sendMessage}
            style={{
              padding: "12px 20px",
              borderRadius: "8px",
              border: "none",
              background: "#2563eb",
              color: "white",
              cursor: "pointer"
            }}
          >
            Send
          </button>

        </div>

      </div>

    </div>
  )
}