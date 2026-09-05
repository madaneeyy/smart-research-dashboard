import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  BookOpen,
  ChevronDown,
  FileText,
  GitBranch,
  Paperclip,
  Loader2,
  MessageSquare,
  Plus,
  Send,
  Settings2,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";

import {
  addChatMessage,
  addChatSource,
  createChat,
  deleteChat,
  getChatMessages,
  getChatSources,
  getWorkspaceChats,
  getWorkspaceDocuments,
  getWorkspaceSources,
  removeChatSource,
  updateChatTitle,
  uploadWorkspaceDocument,
  streamAsk,
  tryRecordWorkspaceActivity,
  type Chat,
  type ChatMessage,
  type Workspace,
  type WorkspaceDocument,
  type WorkspaceSource,
} from "../lib/api";

interface ChatPageProps {
  workspace: Workspace;
  initialDocumentIds?: string[];
  initialGithubSourceIds?: string[];
  onInitialDocumentsConsumed?: () => void;
  onInitialGithubSourcesConsumed?: () => void;
  onBack?: () => void;
}

type LocalMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  metadata?: AskResponseMetadata;
};

type AskResponseMetadata = {
  model?: string;
  retriever?: string;
  context_origin?: string;
  github_url?: string | null;
  chunks_created?: number;
  chunks_retrieved?: number;
  sources?: RetrievedSource[];
  retrieval?: {
    top_k?: number;
    query_type?: string;
    candidate_pool_size?: number;
    post_filter_pool_size?: number;
  };
};

type RetrievedSource = {
  source?: string;
  section?: string;
  content?: string;
  document_id?: string;
  filename?: string;
  path?: string;
  page?: number | null;
  chunk_id?: string;
  relevance_score?: number;
};

const SUGGESTIONS = [
  "Summarize the main ideas",
  "Explain the key methodology",
  "What are the strongest findings?",
  "What should I investigate next?",
];

export function ChatPage({
  workspace,
  initialDocumentIds = [],
  initialGithubSourceIds = [],
  onInitialDocumentsConsumed,
  onInitialGithubSourcesConsumed,
  onBack,
}: ChatPageProps) {
  const [chats, setChats] = useState<Chat[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [messages, setMessages] = useState<LocalMessage[]>([]);
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
  const [selectedDocumentIds, setSelectedDocumentIds] =
    useState<Set<string>>(new Set());

  const [githubSources, setGithubSources] = useState<
    WorkspaceSource[]
  >([]);
  const [arxivSources, setArxivSources] = useState<WorkspaceSource[]>([]);
  const [selectedArxivSourceIds, setSelectedArxivSourceIds] = useState<Set<string>>(
    new Set(),
  );
  const [selectedGithubSourceIds, setSelectedGithubSourceIds] =
    useState<Set<string>>(new Set());

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(true);
  const [loadingChat, setLoadingChat] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [showSources, setShowSources] = useState(false);
  const [expandedEvidence, setExpandedEvidence] = useState<
    Record<string, boolean>
  >({});

  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    void bootstrap();
  }, [workspace.id]);

  useEffect(() => {
    requestAnimationFrame(() => {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    });
  }, [messages, sending]);

  async function bootstrap() {
    setLoading(true);
    setError(null);

    try {
      const [
        workspaceChats,
        workspaceDocuments,
        workspaceSources,
      ] = await Promise.all([
        getWorkspaceChats(workspace.id),
        getWorkspaceDocuments(workspace.id),
        getWorkspaceSources(workspace.id),
      ]);

      setChats(workspaceChats);

      setGithubSources(
        workspaceSources.filter((source) => {
          const sourceType = source.source_type.toLowerCase();

          return (
            sourceType === "github" ||
            sourceType === "github_repository"
          );
        }),
      );
      setArxivSources(
        workspaceSources.filter((source) =>
          source.source_type.toLowerCase() === "arxiv",
        ),
      );

      const arxivDocumentIds = new Set(
        workspaceSources
          .filter((source) => source.source_type.toLowerCase() === "arxiv")
          .map((source) => {
            const metadata = source.metadata ?? {};
            const documentId = metadata["document_id"];
            return documentId ? String(documentId) : "";
          })
          .filter(Boolean),
      );

      setDocuments(
        workspaceDocuments.filter(
          (document) =>
            document.status !== "error" &&
            !arxivDocumentIds.has(String(document.document_id)),
        ),
      );

      if (
        initialDocumentIds.length > 0 ||
        initialGithubSourceIds.length > 0
      ) {
        await createNewChat(
          workspaceChats,
          initialDocumentIds,
          initialGithubSourceIds,
        );

        if (initialDocumentIds.length > 0) {
          onInitialDocumentsConsumed?.();
        }

        if (initialGithubSourceIds.length > 0) {
          onInitialGithubSourcesConsumed?.();
        }
      } else if (workspaceChats.length > 0) {
        await loadChat(workspaceChats[0].id);
      } else {
        await createNewChat(workspaceChats);
      }
    } catch (chatError) {
      setError(
        chatError instanceof Error
          ? chatError.message
          : "Unable to open research chat.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function createNewChat(
    currentChats = chats,
    initialIds: string[] = [],
    initialGithubIds: string[] = [],
    recordActivity = false,
  ) {
    setError(null);

    try {
      const normalizedIds = Array.from(
        new Set(
          initialIds
            .map((id) => String(id).trim())
            .filter(Boolean),
        ),
      );

      // Attach initial documents as part of chat creation. This avoids
      // concurrent POST /sources calls from Promise.all(), which was
      // triggering the shared synchronous Supabase client's HTTP/2
      // WinError 10035 on Windows.
      const normalizedGithubIds = Array.from(
        new Set(
          initialGithubIds
            .map((id) => String(id).trim())
            .filter(Boolean),
        ),
      );

      const arxivDocumentIds = new Set(
        arxivSources
          .map((source) => String((source.metadata ?? {})["document_id"] ?? "").trim())
          .filter(Boolean),
      );

      const initialArxivIds = normalizedIds.filter((id) =>
        arxivDocumentIds.has(id),
      );

      const result = await createChat({
        workspace_id: workspace.id,
        title: "New Chat",
        document_ids:
          normalizedIds.length > 0
            ? normalizedIds
            : undefined,
        sources:
          normalizedGithubIds.length > 0
            ? normalizedGithubIds.map((sourceId) => ({
                source_type: "github",
                source_id: sourceId,
              }))
            : undefined,
      });

      const chatId = result.chat.id;

      // SourcesPage historically opens Chat with arXiv document IDs. Convert
      // those generic document chat_sources to the explicit arXiv provenance
      // model so the Chat source picker can display the paper title and keep
      // selection/removal semantics consistent.
      for (const documentId of initialArxivIds) {
        try {
          await removeChatSource(chatId, "document", documentId);
        } catch {
          // The backend may already have stored the source as arXiv.
        }

        try {
          await addChatSource(chatId, {
            source_type: "arxiv",
            source_id: documentId,
          });
        } catch {
          // Keep the document source as a fallback if arXiv provenance is
          // unavailable for an older backend. Retrieval still uses documentIds.
        }
      }

      setChats([result.chat, ...currentChats]);
      setActiveChatId(chatId);
      setMessages([]);
      setSelectedDocumentIds(new Set(normalizedIds));
      setSelectedArxivSourceIds(new Set(initialArxivIds));
      setSelectedGithubSourceIds(new Set(normalizedGithubIds));
      setInput("");

      if (recordActivity) {
        void tryRecordWorkspaceActivity(workspace.id, {
          activity_type: "chat_started",
          title: "Started a new chat",
          description: result.chat.title || "New Chat",
          reference_id: chatId,
          reference_type: "chat",
          metadata: {
            source_count: normalizedIds.length + normalizedGithubIds.length,
          },
        });
      }
    } catch (chatError) {
      setError(
        chatError instanceof Error
          ? chatError.message
          : "Unable to create conversation.",
      );
    }
  }

  async function loadChat(chatId: string) {
    setLoadingChat(true);
    setError(null);

    try {
      const [loadedMessages, loadedSources] = await Promise.all([
        getChatMessages(chatId),
        getChatSources(chatId),
      ]);

      setActiveChatId(chatId);

      setMessages(
        loadedMessages
          .filter(
            (message) =>
              message.role === "user" ||
              message.role === "assistant",
          )
          .map((message, index) => ({
            id: message.id ?? `${chatId}-${index}`,
            role: message.role,
            content: message.content,
            metadata: parseMetadata(message),
          })),
      );

      const loadedArxivSources = loadedSources.filter(
        (source) => source.source_type.toLowerCase() === "arxiv",
      );

      const arxivDocumentIds = new Set(
        arxivSources
          .map((source) => String((source.metadata ?? {})["document_id"] ?? "").trim())
          .filter(Boolean),
      );

      const loadedArxivDocumentIds = loadedSources
        .filter((source) => {
          const type = source.source_type.toLowerCase();
          return (
            (type === "arxiv" || type === "document") &&
            arxivDocumentIds.has(String(source.source_id).trim())
          );
        })
        .map((source) => String(source.source_id).trim());

      setSelectedArxivSourceIds(
        new Set([
          ...loadedArxivSources.map((source) => String(source.source_id)),
          ...loadedArxivDocumentIds,
        ]),
      );

      setSelectedDocumentIds(
        new Set(
          loadedSources
            .filter((source) => {
              const type = source.source_type.toLowerCase();
              return type === "document" || type === "arxiv";
            })
            .map((source) => String(source.source_id)),
        ),
      );

      setSelectedGithubSourceIds(
        new Set(
          loadedSources
            .filter((source) => {
              const sourceType =
                source.source_type.toLowerCase();

              return (
                sourceType === "github" ||
                sourceType === "github_repository"
              );
            })
            .map((source) => String(source.source_id)),
        ),
      );
    } catch (chatError) {
      setError(
        chatError instanceof Error
          ? chatError.message
          : "Unable to load conversation.",
      );
    } finally {
      setLoadingChat(false);
    }
  }

  async function sendMessage(forcedPrompt?: string) {
    const question = (forcedPrompt ?? input).trim();

    if (!question || sending || !activeChatId) {
      return;
    }

    setInput("");
    setError(null);

    const previousHistory = messages.map((message) => ({
      role: message.role,
      content: message.content,
    }));

    const userMessage: LocalMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: question,
    };

    const assistantId = crypto.randomUUID();

    setMessages((current) => [
      ...current,
      userMessage,
      {
        id: assistantId,
        role: "assistant",
        content: "",
      },
    ]);

    setSending(true);

    try {
      await addChatMessage(activeChatId, {
        role: "user",
        content: question,
      });

      const answer = await streamAsk({
        question,
        history: previousHistory,
        chatId: activeChatId,
        documentIds: Array.from(selectedDocumentIds),
        githubUrls: Array.from(selectedGithubSourceIds),
        onToken: (token) => {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? {
                    ...message,
                    content: message.content + token,
                  }
                : message,
            ),
          );
        },
      });

      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content:
                  answer.answer ||
                  "I couldn't produce an answer.",
                metadata: answer,
              }
            : message,
        ),
      );

      await addChatMessage(activeChatId, {
        role: "assistant",
        content:
          answer.answer ||
          "I couldn't produce an answer.",
      });

      if (activeChat?.title === "New Chat") {
        const generatedTitle = buildChatTitle(
          question,
          documents.filter((document) =>
            selectedDocumentIds.has(document.document_id),
          ),
        );

        try {
          const updatedChat = await updateChatTitle(
            activeChatId,
            generatedTitle,
          );

          setChats((current) => {
            const next = current.map((chat) =>
              chat.id === activeChatId
                ? updatedChat
                : chat,
            );

            const active = next.find(
              (chat) => chat.id === activeChatId,
            );
            const others = next.filter(
              (chat) => chat.id !== activeChatId,
            );

            return active ? [active, ...others] : next;
          });
        } catch {
          // Keep the local title if the optional title update fails.
          setChats((current) =>
            current.map((chat) =>
              chat.id === activeChatId
                ? { ...chat, title: generatedTitle }
                : chat,
            ),
          );
        }
      }
    } catch (chatError) {
      setMessages((current) =>
        current.filter(
          (message) =>
            message.id !== userMessage.id &&
            message.id !== assistantId,
        ),
      );

      setError(
        chatError instanceof Error
          ? chatError.message
          : "The AI request failed.",
      );
    } finally {
      setSending(false);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }

  async function handleDeleteChat(chatId: string) {
    setError(null);

    try {
      await deleteChat(chatId);

      const remaining = chats.filter((chat) => chat.id !== chatId);
      setChats(remaining);

      if (activeChatId === chatId) {
        if (remaining.length > 0) {
          await loadChat(remaining[0].id);
        } else {
          await createNewChat([]);
        }
      }
    } catch (chatError) {
      setError(
        chatError instanceof Error
          ? chatError.message
          : "Unable to delete conversation.",
      );
    }
  }

  async function toggleDocument(documentId: string) {
    if (!activeChatId || sending) {
      return;
    }

    setError(null);

    const wasSelected = selectedDocumentIds.has(documentId);

    // Optimistic UI: selection changes immediately.
    setSelectedDocumentIds((current) => {
      const next = new Set(current);

      if (wasSelected) {
        next.delete(documentId);
      } else {
        next.add(documentId);
      }

      return next;
    });

    try {
      if (wasSelected) {
        await removeChatSource(
          activeChatId,
          "document",
          documentId,
        );
      } else {
        await addChatSource(activeChatId, {
          source_type: "document",
          source_id: documentId,
        });
      }
    } catch (chatError) {
      // Roll back only if persistence failed.
      setSelectedDocumentIds((current) => {
        const next = new Set(current);

        if (wasSelected) {
          next.add(documentId);
        } else {
          next.delete(documentId);
        }

        return next;
      });

      setError(
        chatError instanceof Error
          ? chatError.message
          : wasSelected
            ? "Unable to remove document from this chat."
            : "Unable to attach document to this chat.",
      );
    }
  }


  async function toggleArxiv(source: WorkspaceSource) {
    if (!activeChatId || sending) {
      return;
    }

    const documentId = String(
      (source.metadata ?? {})["document_id"] ?? "",
    ).trim();

    if (!documentId) {
      setError("This arXiv paper is missing its document reference.");
      return;
    }

    const sourceId = documentId;
    const wasSelected = selectedArxivSourceIds.has(sourceId);
    setError(null);

    setSelectedArxivSourceIds((current) => {
      const next = new Set(current);
      if (wasSelected) next.delete(sourceId);
      else next.add(sourceId);
      return next;
    });

    setSelectedDocumentIds((current) => {
      const next = new Set(current);
      if (wasSelected) next.delete(documentId);
      else next.add(documentId);
      return next;
    });

    try {
      if (wasSelected) {
        await removeChatSource(activeChatId, "arxiv", sourceId);
      } else {
        await addChatSource(activeChatId, {
          source_type: "arxiv",
          source_id: sourceId,
        });
      }
    } catch (chatError) {
      setSelectedArxivSourceIds((current) => {
        const next = new Set(current);
        if (wasSelected) next.add(sourceId);
        else next.delete(sourceId);
        return next;
      });
      setSelectedDocumentIds((current) => {
        const next = new Set(current);
        if (wasSelected) next.add(documentId);
        else next.delete(documentId);
        return next;
      });
      setError(
        chatError instanceof Error
          ? chatError.message
          : wasSelected
            ? "Unable to remove paper from this chat."
            : "Unable to attach paper to this chat.",
      );
    }
  }

  async function toggleGithub(sourceId: string) {
    if (!activeChatId || sending) {
      return;
    }

    const normalizedId = sourceId.trim();

    if (!normalizedId) {
      return;
    }

    setError(null);

    const wasSelected =
      selectedGithubSourceIds.has(normalizedId);

    setSelectedGithubSourceIds((current) => {
      const next = new Set(current);

      if (wasSelected) {
        next.delete(normalizedId);
      } else {
        next.add(normalizedId);
      }

      return next;
    });

    try {
      if (wasSelected) {
        await removeChatSource(
          activeChatId,
          "github",
          normalizedId,
        );
      } else {
        await addChatSource(activeChatId, {
          source_type: "github",
          source_id: normalizedId,
        });
      }
    } catch (chatError) {
      setSelectedGithubSourceIds((current) => {
        const next = new Set(current);

        if (wasSelected) {
          next.add(normalizedId);
        } else {
          next.delete(normalizedId);
        }

        return next;
      });

      setError(
        chatError instanceof Error
          ? chatError.message
          : wasSelected
            ? "Unable to remove GitHub source from this chat."
            : "Unable to attach GitHub source to this chat.",
      );
    }
  }

  async function handleAttachFile(
    event: React.ChangeEvent<HTMLInputElement>,
  ) {
    const file = event.target.files?.[0];
    event.target.value = "";

    if (!file || !activeChatId) {
      return;
    }

    setError(null);
    setSending(true);

    try {
      const uploaded = await uploadWorkspaceDocument(
        workspace.id,
        file,
      );

      if (!uploaded.already_exists) {
        void tryRecordWorkspaceActivity(workspace.id, {
          activity_type: "document_added",
          title: "Document added",
          description: file.name,
          reference_id: uploaded.document_id,
          reference_type: "document",
          metadata: {
            filename: file.name,
            content_type: file.type || null,
          },
        });
      }

      setDocuments((current) => {
        const existingIndex = current.findIndex(
          (document) =>
            document.id === uploaded.id,
        );

        if (existingIndex >= 0) {
          const next = [...current];
          next[existingIndex] = uploaded;
          return next;
        }

        return [uploaded, ...current];
      });

      await addChatSource(activeChatId, {
        source_type: "document",
        source_id: uploaded.document_id,
      });

      setSelectedDocumentIds((current) => {
        const next = new Set(current);
        next.add(uploaded.document_id);
        return next;
      });
    } catch (attachError) {
      setError(
        attachError instanceof Error
          ? attachError.message
          : `Could not attach document — ${file.name}`,
      );
    } finally {
      setSending(false);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId),
    [chats, activeChatId],
  );

  const visibleSelectedSourceCount =
    Math.max(0, selectedDocumentIds.size - selectedArxivSourceIds.size) +
    selectedArxivSourceIds.size +
    selectedGithubSourceIds.size;

  if (loading) {
    return (
      <div className="flex min-h-[calc(100vh-72px)] items-center justify-center bg-[var(--paper)]">
        <div className="text-center">
          <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl bg-[var(--ink)]">
            <Sparkles
              size={17}
              className="animate-pulse text-[var(--paper)]"
            />
          </div>
          <p className="mt-4 font-[var(--font-display)] text-sm font-semibold">
            Opening research chat
          </p>
          <p className="mt-1 font-[var(--font-mono)] text-[9px] uppercase tracking-[0.12em] text-[var(--muted)]">
            Loading workspace context
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex h-[calc(100vh-72px)] min-h-[650px] flex-col bg-[var(--paper)] text-[var(--ink)]">
      <header className="flex h-16 shrink-0 items-center justify-between border-b border-[var(--line)] px-5 sm:px-7">
        <div className="flex min-w-0 items-center gap-3">
          {onBack && (
            <button
              type="button"
              onClick={onBack}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-[var(--line)] text-[var(--ink-soft)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] hover:shadow-sm"
              title="Back to workspace"
            >
              <ArrowLeft size={15} />
            </button>
          )}

          <div className="h-7 w-px bg-[var(--line-soft)]" />

          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <MessageSquare
                size={14}
                className="text-[var(--cyan)]"
              />
              <p className="truncate font-[var(--font-display)] text-sm font-semibold">
                {activeChat?.title ?? "New Chat"}
              </p>
            </div>

            <p className="mt-0.5 truncate font-[var(--font-mono)] text-[9px] uppercase tracking-[0.12em] text-[var(--muted)]">
              {workspace.name} · Research AI
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setShowSources((open) => !open)}
            className={[
              "inline-flex items-center gap-2 rounded-md px-3 py-2 text-xs font-medium transition-all duration-200",
              showSources
                ? "bg-[var(--cyan-dim)] text-[var(--cyan)]"
                : "border border-[var(--line)] text-[var(--ink-soft)] hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] hover:shadow-sm",
            ].join(" ")}
          >
            <BookOpen size={13} />
            <span className="hidden sm:inline">Sources</span>
            {visibleSelectedSourceCount >
              0 && (
              <span className="rounded-full bg-[var(--paper)] px-1.5 py-0.5 font-[var(--font-mono)] text-[8px]">
                {visibleSelectedSourceCount}
              </span>
            )}
          </button>
        </div>
      </header>

      <div
        className="min-h-0 flex-1 lg:flex"
      >
        <aside className="hidden w-64 shrink-0 flex-col border-r border-[var(--line)] bg-[var(--paper-dim)] lg:flex">
          <div className="border-b border-[var(--line-soft)] p-4">
            <button
              type="button"
              onClick={() => void createNewChat(chats, [], [], true)}
              className="flex w-full items-center justify-center gap-2 rounded-lg bg-[var(--ink)] px-3 py-2.5 text-xs font-medium text-[var(--paper)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-sm"
            >
              <Plus size={13} />
              New chat
            </button>
          </div>

          <div className="px-4 pb-2 pt-4">
            <p className="font-[var(--font-mono)] text-[9px] uppercase tracking-[0.13em] text-[var(--muted)]">
              Recent chats
            </p>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-4">
            {chats.map((chat) => (
              <div
                key={chat.id}
                className={[
                  "group mb-1 flex items-center gap-1 rounded-lg px-2 py-2 transition-colors",
                  chat.id === activeChatId
                    ? "bg-[var(--paper)] shadow-sm"
                    : "hover:bg-[var(--paper)]",
                ].join(" ")}
              >
                <button
                  type="button"
                  onClick={() => void loadChat(chat.id)}
                  className="min-w-0 flex-1 text-left"
                >
                  <p className="truncate text-xs font-medium text-[var(--ink)]">
                    {chat.title || "New Chat"}
                  </p>
                  <p className="mt-0.5 font-[var(--font-mono)] text-[8px] uppercase tracking-[0.08em] text-[var(--muted)]">
                    {formatChatDate(chat.updated_at ?? chat.created_at)}
                  </p>
                </button>

                <button
                  type="button"
                  onClick={() => void handleDeleteChat(chat.id)}
                  className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-[var(--muted)] opacity-0 transition-all hover:bg-[var(--accent-dim)] hover:text-[var(--accent)] group-hover:opacity-100"
                  title="Delete chat"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}

            {chats.length === 0 && (
              <p className="px-2 py-5 text-center text-[10px] text-[var(--muted)]">
                No recent chats.
              </p>
            )}
          </div>
        </aside>

        <div className="min-w-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-4xl px-5 pb-44 pt-10 sm:px-8 lg:px-10">
            {error && (
              <div className="mb-5 flex items-start gap-3 rounded-xl border border-[var(--line)] bg-[var(--accent-dim)] px-4 py-3 text-xs text-[var(--accent)]">
                <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-[var(--accent)]" />
                <span className="flex-1">{error}</span>
                <button
                  type="button"
                  onClick={() => setError(null)}
                >
                  <X size={13} />
                </button>
              </div>
            )}

            {loadingChat ? (
              <div className="flex min-h-[50vh] items-center justify-center">
                <Loader2
                  size={18}
                  className="animate-spin text-[var(--cyan)]"
                />
              </div>
            ) : messages.length === 0 ? (
              <ChatEmptyState
                workspace={workspace}
                onSuggestion={(prompt) => void sendMessage(prompt)}
              />
            ) : (
              <div className="space-y-8">
                {messages.map((message) => (
                  <MessageBubble
                    key={message.id}
                    message={message}
                    evidenceOpen={Boolean(
                      expandedEvidence[message.id],
                    )}
                    onToggleEvidence={() =>
                      setExpandedEvidence((current) => ({
                        ...current,
                        [message.id]: !current[message.id],
                      }))
                    }
                  />
                ))}

                {sending &&
                  messages[messages.length - 1]?.content === "" && (
                    <div className="flex items-start gap-3">
                      <AssistantMark />
                      <div className="rounded-2xl border border-[var(--line-soft)] bg-[var(--paper-dim)] px-4 py-3">
                        <span className="flex items-center gap-2 text-xs text-[var(--muted)]">
                          <Loader2
                            size={13}
                            className="animate-spin"
                          />
                          Retrieving evidence and thinking…
                        </span>
                      </div>
                    </div>
                  )}

                <div ref={bottomRef} />
              </div>
            )}
          </div>
        </div>
      </div>

      {showSources && (
        <aside className="absolute inset-y-16 right-0 z-30 flex w-full max-w-sm flex-col border-l border-[var(--line)] bg-[var(--paper)] shadow-xl">
          <div className="flex items-center justify-between border-b border-[var(--line-soft)] px-5 py-4">
            <div>
              <p className="text-sm font-semibold">
                Chat sources
              </p>
              <p className="mt-1 text-[10px] text-[var(--muted)]">
                Choose the papers, documents, and GitHub repositories this chat should use.
              </p>
            </div>

            <button
              type="button"
              onClick={() => setShowSources(false)}
              className="flex h-8 w-8 items-center justify-center rounded-md text-[var(--muted)] hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
            >
              <X size={15} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {githubSources.length > 0 && (
              <section className="mb-5">
                <div className="mb-2 flex items-center gap-2 px-1">
                  <GitBranch
                    size={12}
                    className="text-[var(--cyan)]"
                  />
                  <p className="font-[var(--font-mono)] text-[9px] uppercase tracking-[0.13em] text-[var(--muted)]">
                    GitHub repositories
                  </p>
                  <span className="ml-auto font-[var(--font-mono)] text-[8px] text-[var(--muted)]">
                    {selectedGithubSourceIds.size}/{githubSources.length}
                  </span>
                </div>

                <div className="space-y-2">
                  {githubSources.map((source) => {
                    const sourceId = String(
                      source.url || source.id || "",
                    ).trim();

                    if (!sourceId) {
                      return null;
                    }

                    const selected =
                      selectedGithubSourceIds.has(sourceId);

                    return (
                      <button
                        key={source.id || sourceId}
                        type="button"
                        disabled={!activeChatId || sending}
                        onClick={() => void toggleGithub(sourceId)}
                        className={[
                          "flex w-full items-start gap-3 rounded-xl border p-3 text-left transition-all duration-200 disabled:cursor-not-allowed disabled:opacity-50",
                          selected
                            ? "border-[var(--cyan)] bg-[var(--cyan-dim)]"
                            : "border-[var(--line)] hover:bg-[var(--paper-dim)]",
                        ].join(" ")}
                      >
                        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--paper)] text-[var(--cyan)]">
                          <GitBranch size={14} />
                        </div>

                        <div className="min-w-0 flex-1">
                          <p className="truncate text-xs font-medium">
                            {source.title || "GitHub repository"}
                          </p>
                          <p className="mt-1 truncate font-[var(--font-mono)] text-[9px] text-[var(--muted)]">
                            {source.url || sourceId}
                          </p>
                        </div>

                        <span
                          className={[
                            "mt-1 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border",
                            selected
                              ? "border-[var(--cyan)] bg-[var(--cyan)] text-[var(--paper)]"
                              : "border-[var(--line)]",
                          ].join(" ")}
                        >
                          {selected && (
                            <span className="h-1.5 w-1.5 rounded-full bg-current" />
                          )}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </section>
            )}

            {arxivSources.length > 0 && (
              <section className="mb-5">
                <div className="mb-2 flex items-center gap-2 px-1">
                  <BookOpen size={12} className="text-[var(--cyan)]" />
                  <p className="font-[var(--font-mono)] text-[9px] uppercase tracking-[0.13em] text-[var(--muted)]">
                    Papers
                  </p>
                  <span className="ml-auto font-[var(--font-mono)] text-[8px] text-[var(--muted)]">
                    {selectedArxivSourceIds.size}/{arxivSources.length}
                  </span>
                </div>
                <div className="space-y-2">
                  {arxivSources.map((source) => {
                    const documentId = String(
                      (source.metadata ?? {})["document_id"] ?? "",
                    ).trim();
                    if (!documentId) return null;
                    const selected = selectedArxivSourceIds.has(documentId);
                    const arxivId = String(
                      (source.metadata ?? {})["arxiv_id"] ?? "",
                    ).trim();
                    return (
                      <button
                        key={source.id}
                        type="button"
                        disabled={!activeChatId || sending}
                        onClick={() => void toggleArxiv(source)}
                        className={[
                          "flex w-full items-start gap-3 rounded-xl border p-3 text-left transition-all duration-200 disabled:cursor-not-allowed disabled:opacity-50",
                          selected
                            ? "border-[var(--cyan)] bg-[var(--cyan-dim)]"
                            : "border-[var(--line)] hover:bg-[var(--paper-dim)]",
                        ].join(" ")}
                      >
                        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--paper)] text-[var(--cyan)]">
                          <BookOpen size={14} />
                        </div>
                        <div className="min-w-0 flex-1">
                          <p className="line-clamp-2 text-xs font-medium">
                            {source.title || "Untitled paper"}
                          </p>
                          <p className="mt-1 truncate font-[var(--font-mono)] text-[9px] uppercase tracking-[0.08em] text-[var(--muted)]">
                            {arxivId ? `arXiv · ${arxivId}` : "arXiv paper"}
                          </p>
                        </div>
                        <span
                          className={[
                            "mt-1 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border",
                            selected
                              ? "border-[var(--cyan)] bg-[var(--cyan)] text-[var(--paper)]"
                              : "border-[var(--line)]",
                          ].join(" ")}
                        >
                          {selected && <span className="h-1.5 w-1.5 rounded-full bg-current" />}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </section>
            )}

            {documents.length === 0 && githubSources.length === 0 && arxivSources.length === 0 ? (
              <div className="rounded-xl border border-dashed border-[var(--line)] bg-[var(--paper-dim)] px-4 py-8 text-center">
                <FileText
                  size={17}
                  className="mx-auto text-[var(--muted)]"
                />
                <p className="mt-3 text-xs font-medium">
                  No documents available
                </p>
                <p className="mt-1 text-[10px] leading-4 text-[var(--muted)]">
                  Add documents from Sources to use document-grounded chat.
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                <button
                  type="button"
                  disabled={!activeChatId || sending}
                  onClick={() =>
                    void (async () => {
                      if (!activeChatId || sending) {
                        return;
                      }

                      setError(null);

                      const shouldSelectAll =
                        selectedDocumentIds.size !== documents.length;

                      const previousIds = new Set(
                        selectedDocumentIds,
                      );

                      const targetIds = documents.map(
                        (document) => document.document_id,
                      );

                      // Update the UI immediately.
                      setSelectedDocumentIds(
                        shouldSelectAll
                          ? new Set(targetIds)
                          : new Set(),
                      );

                      try {
                        if (shouldSelectAll) {
                          for (const documentId of targetIds) {
                            if (previousIds.has(documentId)) {
                              continue;
                            }

                            await addChatSource(activeChatId, {
                              source_type: "document",
                              source_id: documentId,
                            });
                          }
                        } else {
                          for (const documentId of targetIds) {
                            if (!previousIds.has(documentId)) {
                              continue;
                            }

                            await removeChatSource(
                              activeChatId,
                              "document",
                              documentId,
                            );
                          }
                        }
                      } catch (chatError) {
                        setSelectedDocumentIds(previousIds);

                        setError(
                          chatError instanceof Error
                            ? chatError.message
                            : "Unable to update chat sources.",
                        );
                      }
                    })()
                  }
                  className="mb-2 flex w-full items-center justify-between rounded-lg border border-[var(--line)] px-3 py-2 text-[10px] font-medium text-[var(--ink-soft)] hover:bg-[var(--paper-dim)]"
                >
                  <span>
                    {selectedDocumentIds.size === documents.length
                      ? "Clear all"
                      : "Select all"}
                  </span>
                  <span className="font-[var(--font-mono)] text-[9px] text-[var(--muted)]">
                    {selectedDocumentIds.size}/{documents.length}
                  </span>
                </button>

                {documents.map((document) => {
                  const selected = selectedDocumentIds.has(
                    document.document_id,
                  );

                  return (
                    <button
                      key={document.document_id}
                      type="button"
                      onClick={() =>
                        toggleDocument(document.document_id)
                      }
                      className={[
                        "flex w-full items-start gap-3 rounded-xl border p-3 text-left transition-all duration-200",
                        selected
                          ? "border-[var(--cyan)] bg-[var(--cyan-dim)]"
                          : "border-[var(--line)] hover:bg-[var(--paper-dim)]",
                      ].join(" ")}
                    >
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--paper)] text-[var(--cyan)]">
                        <FileText size={14} />
                      </div>

                      <div className="min-w-0 flex-1">
                        <p className="truncate text-xs font-medium">
                          {document.filename}
                        </p>
                        <p className="mt-1 font-[var(--font-mono)] text-[9px] uppercase tracking-[0.09em] text-[var(--muted)]">
                          {document.pages
                            ? `${document.pages} pages`
                            : document.content_type || "Document"}
                        </p>
                      </div>

                      <span
                        className={[
                          "mt-1 flex h-4 w-4 shrink-0 items-center justify-center rounded-full border",
                          selected
                            ? "border-[var(--cyan)] bg-[var(--cyan)] text-[var(--paper)]"
                            : "border-[var(--line)]",
                        ].join(" ")}
                      >
                        {selected && (
                          <span className="h-1.5 w-1.5 rounded-full bg-current" />
                        )}
                      </span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          <div className="border-t border-[var(--line-soft)] p-4">
            <div className="rounded-xl bg-[var(--paper-dim)] p-3">
              <div className="flex gap-2">
                <Settings2
                  size={14}
                  className="mt-0.5 text-[var(--cyan)]"
                />
                <div>
                  <p className="text-[10px] font-semibold">
                    Grounded chat
                  </p>
                  <p className="mt-1 text-[10px] leading-4 text-[var(--muted)]">
                    Selected documents and GitHub repositories become retrieval targets.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </aside>
      )}

      <div className="pointer-events-none absolute inset-x-0 bottom-0 z-20">
        <div className="pointer-events-auto border-t border-[var(--line)] bg-[var(--paper)]/95 px-4 py-4 backdrop-blur sm:px-8">
          <div className="mx-auto max-w-4xl">
            <div className="rounded-2xl border border-[var(--line)] bg-[var(--paper)] p-2 shadow-lg">
              {(selectedDocumentIds.size > 0 ||
                selectedGithubSourceIds.size > 0) && (
                <div className="flex flex-wrap gap-1.5 px-2 pb-2">
                  {arxivSources
                    .filter((source) => {
                      const documentId = String(
                        (source.metadata ?? {})["document_id"] ?? "",
                      ).trim();
                      return documentId && selectedArxivSourceIds.has(documentId);
                    })
                    .slice(0, 4)
                    .map((source) => (
                      <span
                        key={`arxiv-${source.id}`}
                        className="inline-flex max-w-52 items-center gap-1.5 rounded-full border border-[var(--line-soft)] bg-[var(--paper-dim)] px-2.5 py-1 font-[var(--font-mono)] text-[9px] text-[var(--ink-soft)]"
                      >
                        <BookOpen size={10} />
                        <span className="truncate">{source.title || "Untitled paper"}</span>
                      </span>
                    ))}

                  {documents
                    .filter((document) =>
                      selectedDocumentIds.has(document.document_id) &&
                      !arxivSources.some(
                        (source) =>
                          String((source.metadata ?? {})["document_id"] ?? "").trim() ===
                          String(document.document_id),
                      ),
                    )
                    .slice(0, 4)
                    .map((document) => (
                      <span
                        key={document.document_id}
                        className="inline-flex max-w-48 items-center gap-1.5 rounded-full border border-[var(--line-soft)] bg-[var(--paper-dim)] px-2.5 py-1 font-[var(--font-mono)] text-[9px] text-[var(--ink-soft)]"
                      >
                        <FileText size={10} />
                        <span className="truncate">
                          {document.filename}
                        </span>
                      </span>
                    ))}

                  {Array.from(selectedGithubSourceIds)
                    .slice(
                      0,
                      Math.max(0, 4 - visibleSelectedSourceCount),
                    )
                    .map((sourceId) => {
                      const source = githubSources.find(
                        (item) =>
                          String(
                            item.url || item.id || "",
                          ).trim() === sourceId,
                      );

                      return (
                        <span
                          key={sourceId}
                          className="inline-flex max-w-52 items-center gap-1.5 rounded-full border border-[var(--line-soft)] bg-[var(--paper-dim)] px-2.5 py-1 font-[var(--font-mono)] text-[9px] text-[var(--ink-soft)]"
                        >
                          <GitBranch size={10} />
                          <span className="truncate">
                            {source?.title || "GitHub repository"}
                          </span>
                        </span>
                      );
                    })}

                  {visibleSelectedSourceCount > 4 && (
                    <span className="rounded-full border border-[var(--line-soft)] bg-[var(--paper-dim)] px-2.5 py-1 font-[var(--font-mono)] text-[9px] text-[var(--muted)]">
                      +
                      {visibleSelectedSourceCount - 4}{" "}
                      more
                    </span>
                  )}
                </div>
              )}

              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                onChange={(event) => void handleAttachFile(event)}
              />

              <div className="flex items-end gap-2">
                <button
                  type="button"
                  disabled={sending}
                  onClick={() => fileInputRef.current?.click()}
                  className="mb-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-[var(--line)] text-[var(--muted)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] hover:shadow-sm disabled:cursor-not-allowed disabled:opacity-35"
                  title="Attach document"
                >
                  <Paperclip size={16} />
                </button>

                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={(event) =>
                    setInput(event.target.value)
                  }
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      !event.shiftKey
                    ) {
                      event.preventDefault();
                      void sendMessage();
                    }
                  }}
                  rows={1}
                  placeholder={
                    selectedDocumentIds.size > 0 ||
                    selectedGithubSourceIds.size > 0
                      ? "Ask about your research…"
                      : "Message Smart Research AI…"
                  }
                  className="max-h-36 min-h-12 flex-1 resize-none bg-transparent px-3 py-3 text-sm leading-6 text-[var(--ink)] outline-none placeholder:text-[var(--muted)]"
                />

                <button
                  type="button"
                  disabled={sending || !input.trim()}
                  onClick={() => void sendMessage()}
                  className="mb-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-[var(--ink)] text-[var(--paper)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-md disabled:cursor-not-allowed disabled:opacity-35"
                >
                  {sending ? (
                    <Loader2
                      size={16}
                      className="animate-spin"
                    />
                  ) : (
                    <Send size={16} />
                  )}
                </button>
              </div>

              <div className="flex items-center justify-between px-2 pt-1">
                <p className="font-[var(--font-mono)] text-[8px] uppercase tracking-[0.1em] text-[var(--muted)]">
                  Enter to send · Shift + Enter for newline
                </p>

                <p className="hidden font-[var(--font-mono)] text-[8px] uppercase tracking-[0.1em] text-[var(--muted)] sm:block">
                  {visibleSelectedSourceCount > 0
                    ? `${visibleSelectedSourceCount} source${
                        visibleSelectedSourceCount ===
                        1
                          ? ""
                          : "s"
                      } selected`
                    : "General chat"}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ChatEmptyState({
  workspace,
  onSuggestion,
}: {
  workspace: Workspace;
  onSuggestion: (prompt: string) => void;
}) {
  return (
    <div className="flex min-h-[65vh] flex-col items-center justify-center text-center">
      <div className="relative">
        <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[var(--ink)]">
          <Sparkles size={21} className="text-[var(--paper)]" />
        </div>
        <span className="absolute -right-1 -top-1 h-3 w-3 rounded-full border-2 border-[var(--paper)] bg-[var(--accent)]" />
      </div>

      <p className="mt-6 font-[var(--font-mono)] text-[9px] uppercase tracking-[0.15em] text-[var(--muted)]">
        {workspace.name}
      </p>

      <h1 className="mt-2 font-[var(--font-display)] text-3xl font-semibold tracking-[-0.03em] sm:text-4xl">
        What can I help you investigate?
      </h1>

      <p className="mt-3 max-w-xl text-sm leading-6 text-[var(--ink-soft)]">
        Ask questions, synthesize documents, understand technical work,
        or follow an idea deeper with grounded evidence.
      </p>

      <div className="mt-8 grid w-full max-w-2xl gap-2 sm:grid-cols-2">
        {SUGGESTIONS.map((suggestion) => (
          <button
            key={suggestion}
            type="button"
            onClick={() => onSuggestion(suggestion)}
            className="group rounded-xl border border-[var(--line)] bg-[var(--paper)] p-3.5 text-left transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:shadow-sm"
          >
            <div className="flex items-center justify-between gap-3">
              <span className="text-xs font-medium text-[var(--ink-soft)] group-hover:text-[var(--ink)]">
                {suggestion}
              </span>
              <ArrowLeft
                size={13}
                className="rotate-180 text-[var(--muted)] transition-transform group-hover:translate-x-0.5"
              />
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({
  message,
  evidenceOpen,
  onToggleEvidence,
}: {
  message: LocalMessage;
  evidenceOpen: boolean;
  onToggleEvidence: () => void;
}) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-[var(--ink)] px-4 py-3 text-sm leading-6 text-[var(--paper)]">
          {message.content}
        </div>
      </div>
    );
  }

  const sources = message.metadata?.sources ?? [];

  return (
    <div className="flex items-start gap-3">
      <AssistantMark />

      <div className="min-w-0 max-w-3xl flex-1">
        <div className="whitespace-pre-wrap text-sm leading-7 text-[var(--ink-soft)]">
          {message.content}
        </div>

        {message.metadata && (
          <div className="mt-4">
            <button
              type="button"
              onClick={onToggleEvidence}
              className="flex w-full items-center gap-3 rounded-xl border border-[var(--line)] bg-[var(--paper-dim)] px-3.5 py-3 text-left"
            >
              <BookOpen
                size={13}
                className="text-[var(--cyan)]"
              />

              <div className="min-w-0 flex-1">
                <p className="text-[10px] font-semibold">
                  {sources.length > 0
                    ? `${sources.length} source${
                        sources.length === 1 ? "" : "s"
                      } used`
                    : message.metadata.context_origin ===
                        "general_chat"
                      ? "General knowledge"
                      : "Retrieved evidence"}
                </p>

                <p className="mt-0.5 text-[9px] text-[var(--muted)]">
                  {message.metadata.model ?? "Smart Research AI"}
                  {message.metadata.chunks_retrieved
                    ? ` · ${message.metadata.chunks_retrieved} evidence chunks`
                    : ""}
                </p>
              </div>

              <ChevronDown
                size={13}
                className={[
                  "text-[var(--muted)] transition-transform",
                  evidenceOpen ? "rotate-180" : "",
                ].join(" ")}
              />
            </button>

            {evidenceOpen && sources.length > 0 && (
              <div className="mt-2 space-y-2">
                {sources.map((source, index) => (
                  <div
                    key={
                      source.chunk_id ??
                      `${source.document_id}-${index}`
                    }
                    className="rounded-xl border border-[var(--line)] bg-[var(--paper)] p-3.5"
                  >
                    <p className="font-[var(--font-mono)] text-[8px] uppercase tracking-[0.12em] text-[var(--muted)]">
                      Evidence {index + 1}
                    </p>

                    <p className="mt-2 text-[10px] font-semibold">
                      {source.filename ||
                        source.path ||
                        source.source ||
                        "Workspace source"}
                    </p>

                    {source.section && (
                      <p className="mt-1 text-[9px] text-[var(--muted)]">
                        {source.section}
                        {source.page
                          ? ` · page ${source.page}`
                          : ""}
                      </p>
                    )}

                    {source.content && (
                      <p className="mt-3 whitespace-pre-wrap text-xs leading-5 text-[var(--ink-soft)]">
                        {source.content}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function AssistantMark() {
  return (
    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl border border-[var(--line)] bg-[var(--paper-dim)] text-[var(--cyan)]">
      <Sparkles size={14} />
    </div>
  );
}

function compactTitle(value: string) {
  const text = value.trim().replace(/\s+/g, " ");
  return text.length <= 48
    ? text
    : `${text.slice(0, 45).trimEnd()}...`;
}

function buildChatTitle(
  question: string,
  selectedDocuments: WorkspaceDocument[],
) {
  const base = compactTitle(question);

  if (selectedDocuments.length === 0) {
    return base;
  }

  const filename = selectedDocuments[0].filename
    .replace(/\.[^/.]+$/, "")
    .trim();

  if (!filename) {
    return base;
  }

  const available = Math.max(12, 60 - filename.length - 3);
  const questionPart = question
    .trim()
    .replace(/\s+/g, " ")
    .slice(0, available)
    .trimEnd();

  return `${questionPart} · ${filename}`.slice(0, 60);
}

function formatChatDate(value: string) {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "";
  }

  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function parseMetadata(
  message: ChatMessage,
): AskResponseMetadata | undefined {
  if (
    message.metadata &&
    typeof message.metadata === "object"
  ) {
    return message.metadata as AskResponseMetadata;
  }

  return undefined;
}