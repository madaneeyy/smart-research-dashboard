const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ??
  "https://smart-research-dashboard.onrender.com";

/* ============================================================
   Types
   ============================================================ */

export interface CreateWorkspacePayload {
  name: string;
  description?: string | null;
}

export interface Workspace {
  id: string;
  name: string;
  description: string | null;
  owner_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface DeleteWorkspaceResponse {
  message: string;
}

export interface WorkspaceSource {
  id: string;
  workspace_id: string;
  source_type: string;
  title: string;
  url: string | null;
  metadata: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface WorkspaceDocument {
  id: string;
  workspace_id: string;
  document_id: string;
  filename: string;
  content_type: string | null;
  pages: number | null;
  characters: number | null;
  size_bytes: number | null;
  status: string;
  created_at?: string;
  updated_at?: string;
}

export interface DeleteSourceResponse {
  message?: string;
}

export interface DeleteDocumentResponse {
  message?: string;
}

export interface DocumentPreview {
  document_id: string;
  filename: string;
  content_type: string | null;
  pages: number | null;
  characters: number | null;
  content: string;
}

export type ActivityType =
  | "document_added"
  | "paper_added"
  | "model_added"
  | "repository_added"
  | "chat_started"
  | "research_performed";

export interface WorkspaceActivity {
  id: string;
  workspace_id: string;
  activity_type: ActivityType;
  title: string;
  description: string | null;
  reference_id: string | null;
  reference_type: string | null;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

export interface CreateWorkspaceActivityPayload {
  activity_type: ActivityType;
  title: string;
  description?: string | null;
  reference_id?: string | null;
  reference_type?: string | null;
  metadata?: Record<string, unknown> | null;
}

/* ============================================================
   Error handling
   ============================================================ */

async function getErrorMessage(
  response: Response,
  fallback: string,
): Promise<string> {
  try {
    const data = await response.json();

    if (typeof data?.detail === "string") {
      return data.detail;
    }

    if (
      data?.detail &&
      typeof data.detail.message === "string"
    ) {
      return data.detail.message;
    }

    if (typeof data?.message === "string") {
      return data.message;
    }
  } catch {
    // Keep the fallback.
  }

  return fallback;
}

async function requestJson<T>(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  fallback: string,
): Promise<T> {
  const response = await fetch(input, init);

  if (!response.ok) {
    throw new Error(
      await getErrorMessage(response, fallback),
    );
  }

  return response.json() as Promise<T>;
}

/* ============================================================
   Workspaces
   ============================================================ */

export async function createWorkspace(
  payload: CreateWorkspacePayload,
): Promise<Workspace> {
  return requestJson<Workspace>(
    `${API_BASE_URL}/workspaces`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
    "Unable to create workspace.",
  );
}

export async function getWorkspace(
  workspaceId: string,
): Promise<Workspace> {
  return requestJson<Workspace>(
    `${API_BASE_URL}/workspaces/${workspaceId}`,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    },
    "Workspace not found.",
  );
}
export async function listWorkspaces(): Promise<Workspace[]> {
  return requestJson<Workspace[]>(
    `${API_BASE_URL}/workspaces`,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    },
    "Unable to load workspaces.",
  );
}

export async function deleteWorkspace(
  workspaceId: string,
): Promise<DeleteWorkspaceResponse> {
  return requestJson<DeleteWorkspaceResponse>(
    `${API_BASE_URL}/workspaces/${workspaceId}`,
    {
      method: "DELETE",
      headers: {
        Accept: "application/json",
      },
    },
    "Unable to delete workspace.",
  );
}

/* ============================================================
   Recent activity
   ============================================================ */

export async function getWorkspaceActivity(
  workspaceId: string,
  limit: number = 8,
): Promise<WorkspaceActivity[]> {
  return requestJson<WorkspaceActivity[]>(
    `${API_BASE_URL}/workspaces/${workspaceId}/activity?limit=${encodeURIComponent(String(limit))}`,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    },
    "Unable to load recent activity.",
  );
}

export async function recordWorkspaceActivity(
  workspaceId: string,
  payload: CreateWorkspaceActivityPayload,
): Promise<WorkspaceActivity> {
  return requestJson<WorkspaceActivity>(
    `${API_BASE_URL}/workspaces/${workspaceId}/activity`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    },
    "Unable to record activity.",
  );
}

export async function tryRecordWorkspaceActivity(
  workspaceId: string,
  payload: CreateWorkspaceActivityPayload,
): Promise<void> {
  try {
    await recordWorkspaceActivity(workspaceId, payload);
  } catch {
    // Activity tracking is non-critical. A failed activity write must never
    // make the user's primary action fail.
  }
}

/* ============================================================
   Research sources
   ============================================================ */

export async function getWorkspaceSources(
  workspaceId: string,
): Promise<WorkspaceSource[]> {
  return requestJson<WorkspaceSource[]>(
    `${API_BASE_URL}/workspaces/${workspaceId}/sources`,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    },
    "Unable to load workspace sources.",
  );
}

export async function deleteWorkspaceSource(
  workspaceId: string,
  sourceId: string,
): Promise<DeleteSourceResponse> {
  return requestJson<DeleteSourceResponse>(
    `${API_BASE_URL}/workspaces/${workspaceId}/sources/${sourceId}`,
    {
      method: "DELETE",
      headers: {
        Accept: "application/json",
      },
    },
    "Unable to remove source.",
  );
}

/* ============================================================
   Documents
   ============================================================ */

export async function getWorkspaceDocuments(
  workspaceId: string,
): Promise<WorkspaceDocument[]> {
  return requestJson<WorkspaceDocument[]>(
    `${API_BASE_URL}/workspaces/${workspaceId}/documents`,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    },
    "Unable to load workspace documents.",
  );
}

export async function uploadWorkspaceDocument(
  workspaceId: string,
  file: File,
): Promise<WorkspaceDocument & {
  already_exists?: boolean;
}> {
  const formData = new FormData();
  formData.append("file", file);

  return requestJson(
    `${API_BASE_URL}/workspaces/${workspaceId}/documents`,
    {
      method: "POST",
      body: formData,
    },
    `Could not add document — ${file.name}`,
  );
}

export async function deleteWorkspaceDocument(
  workspaceId: string,
  documentId: string,
): Promise<DeleteDocumentResponse> {
  return requestJson<DeleteDocumentResponse>(
    `${API_BASE_URL}/workspaces/${workspaceId}/documents/${documentId}`,
    {
      method: "DELETE",
      headers: {
        Accept: "application/json",
      },
    },
    "Unable to delete document.",
  );
}

/* ============================================================
   Document preview
   ============================================================ */

export async function getDocumentPreview(
  workspaceId: string,
  documentId: string,
): Promise<DocumentPreview> {
  return requestJson<DocumentPreview>(
    `${API_BASE_URL}/workspaces/${workspaceId}/documents/${documentId}/preview`,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    },
    "Unable to preview document.",
  );
}

// ============================================================
// RESEARCH DISCOVERY
// ============================================================

export interface ResearchItem {
  id: string;
  title: string;
  description: string | null;
  authors: string[];
  source: string;
  url: string;

  published: string | null;
  updated: string | null;

  tags: string[];

  stars: number | null;
  forks: number | null;
  language: string | null;

  downloads: number | null;
  likes: number | null;
  library: string | null;
  pipeline_tag: string | null;

  tasks: string[];
  conference: string | null;

  metadata: Record<string, unknown>;
}

export interface ResearchSearchPayload {
  query: string;
  sources: string[];
  sortBy:
    | "relevance"
    | "published"
    | "updated";
  searchMode:
    | "keyword"
    | "semantic"
    | "hybrid";
}

export interface CreateWorkspaceSourcePayload {
  source_type: string;
  title: string;
  url?: string | null;
  metadata?: Record<
    string,
    unknown
  > | null;
}

export async function searchResearch(
  payload: ResearchSearchPayload,
): Promise<ResearchItem[]> {
  const response =
    await fetch(
      `${API_BASE_URL}/research/search`,
      {
        method: "POST",
        headers: {
          "Content-Type":
            "application/json",
          Accept:
            "application/json",
        },
        body: JSON.stringify({
          query:
            payload.query,
          sources:
            payload.sources,
          sort_by:
            payload.sortBy,
          search_mode:
            payload.searchMode,
        }),
      },
    );

  if (!response.ok) {
    let message =
      "Research search failed.";

    try {
      const data =
        await response.json();

      if (
        typeof data.detail ===
        "string"
      ) {
        message =
          data.detail;
      } else if (
        data.detail &&
        typeof data
          .detail
          .message ===
          "string"
      ) {
        message =
          data.detail
            .message;
      }
    } catch {
      // Keep default message.
    }

    throw new Error(
      message,
    );
  }

  return response.json();
}


export async function addResearchSourceToWorkspace(
  workspaceId: string,
  payload: CreateWorkspaceSourcePayload,
): Promise<WorkspaceSource> {
  const response =
    await fetch(
      `${API_BASE_URL}/workspaces/${workspaceId}/sources`,
      {
        method: "POST",
        headers: {
          "Content-Type":
            "application/json",
          Accept:
            "application/json",
        },
        body: JSON.stringify(
          payload,
        ),
      },
    );

  if (!response.ok) {
    let message =
      "Could not add source to workspace.";

    try {
      const data =
        await response.json();

      if (
        typeof data.detail ===
        "string"
      ) {
        message =
          data.detail;
      } else if (
        data.detail &&
        typeof data
          .detail
          .message ===
          "string"
      ) {
        message =
          data.detail
            .message;
      }
    } catch {
      // Keep default message.
    }

    throw new Error(
      message,
    );
  }

  return response.json();
}

/* ============================================================
   Chat / RAG
   ============================================================ */

export interface Chat {
  id: string;
  workspace_id: string;
  title: string;
  created_at: string;
  updated_at?: string;
}

export interface ChatMessage {
  id?: string;
  chat_id?: string;
  role: "user" | "assistant";
  content: string;
  created_at?: string;
  metadata?: Record<string, unknown> | null;
}

export interface ChatSource {
  id?: string;
  chat_id?: string;
  source_type: string;
  source_id: string;
  title?: string | null;
  created_at?: string;
  metadata?: Record<string, unknown> | null;
}

export interface CreateChatPayload {
  workspace_id: string;
  title?: string;
  source_type?: string;
  source_id?: string;

  /** Canonical document IDs to attach atomically when creating the chat. */
  document_ids?: string[];

  /** Additional chat sources to attach atomically (for example GitHub). */
  sources?: Array<{
    source_type: string;
    source_id: string;
  }>;
}

export interface AddChatMessagePayload {
  role: "user" | "assistant";
  content: string;
}

export interface AddChatSourcePayload {
  source_type: string;
  source_id: string;
}

export interface AskResponse {
  question?: string;
  answer: string;
  model?: string;
  retriever?: string;
  chunker?: string;
  context_origin?: string;
  github_url?: string | null;
  chunks_created?: number;
  chunks_retrieved?: number;
  performance?: Record<string, unknown>;
  sources?: Array<{
    source?: string;
    section?: string;
    content?: string;
    document_id?: string;
    filename?: string;
    path?: string;
    page?: number | null;
    chunk_id?: string;
    relevance_score?: number;
    evidence_role?: string;
    [key: string]: unknown;
  }>;
  retrieval?: {
    top_k?: number;
    query_type?: string;
    candidate_pool_size?: number;
    post_filter_pool_size?: number;
  };
}

export async function getWorkspaceChats(
  workspaceId: string,
): Promise<Chat[]> {
  return requestJson<Chat[]>(
    `${API_BASE_URL}/chats/workspace/${workspaceId}`,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    },
    "Unable to load conversations.",
  );
}

export async function createChat(
  payload: CreateChatPayload,
): Promise<{
  chat: Chat;
  source: ChatSource | null;
  sources?: ChatSource[];
}> {
  return requestJson<{
    chat: Chat;
    source: ChatSource | null;
    sources?: ChatSource[];
  }>(
    `${API_BASE_URL}/chats`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    },
    "Unable to create conversation.",
  );
}

export async function updateChatTitle(
  chatId: string,
  title: string,
): Promise<Chat> {
  return requestJson<Chat>(
    `${API_BASE_URL}/chats/${chatId}`,
    {
      method: "PATCH",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        title,
      }),
    },
    "Unable to update conversation title.",
  );
}

export async function getChatMessages(
  chatId: string,
): Promise<ChatMessage[]> {
  return requestJson<ChatMessage[]>(
    `${API_BASE_URL}/chats/${chatId}/messages`,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    },
    "Unable to load conversation messages.",
  );
}

export async function addChatMessage(
  chatId: string,
  payload: AddChatMessagePayload,
): Promise<ChatMessage> {
  return requestJson<ChatMessage>(
    `${API_BASE_URL}/chats/${chatId}/messages`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    },
    "Unable to save chat message.",
  );
}

export async function getChatSources(
  chatId: string,
): Promise<ChatSource[]> {
  return requestJson<ChatSource[]>(
    `${API_BASE_URL}/chats/${chatId}/sources`,
    {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    },
    "Unable to load conversation sources.",
  );
}

export async function addChatSources(
  chatId: string,
  sourceType: string,
  sourceIds: string[],
): Promise<ChatSource[]> {
  return requestJson<ChatSource[]>(
    `${API_BASE_URL}/chats/${chatId}/sources/batch`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        source_type: sourceType,
        source_ids: sourceIds,
      }),
    },
    "Unable to attach sources to conversation.",
  );
}


export async function addChatSource(
  chatId: string,
  payload: AddChatSourcePayload,
): Promise<ChatSource> {
  return requestJson<ChatSource>(
    `${API_BASE_URL}/chats/${chatId}/sources`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify(payload),
    },
    "Unable to attach source to conversation.",
  );
}

export async function removeChatSource(
  chatId: string,
  sourceType: string,
  sourceId: string,
): Promise<{
  message: string;
}> {
  return requestJson<{
    message: string;
  }>(
    `${API_BASE_URL}/chats/${chatId}/sources/${encodeURIComponent(
      sourceType,
    )}/${encodeURIComponent(sourceId)}`,
    {
      method: "DELETE",
      headers: {
        Accept: "application/json",
      },
    },
    "Unable to remove source from conversation.",
  );
}

export async function deleteChat(
  chatId: string,
): Promise<{
  message: string;
}> {
  return requestJson<{
    message: string;
  }>(
    `${API_BASE_URL}/chats/${chatId}`,
    {
      method: "DELETE",
      headers: {
        Accept: "application/json",
      },
    },
    "Unable to delete conversation.",
  );
}

export interface StreamAskPayload {
  question: string;
  history: Array<{
    role: "user" | "assistant";
    content: string;
  }>;
  chatId: string;
  documentIds?: string[];

  /** Selected GitHub repository URLs for this question. */
  githubUrls?: string[];

  onToken?: (token: string) => void;
}

export async function streamAsk(
  payload: StreamAskPayload,
): Promise<AskResponse> {
  const response = await fetch(
    `${API_BASE_URL}/ask?stream=true`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({
        question: payload.question,
        history: payload.history,
        top_k: 5,
        chat_id: payload.chatId,
        document_ids: payload.documentIds ?? [],
        github_urls: payload.githubUrls ?? [],
      }),
    },
  );

  if (!response.ok) {
    throw new Error(
      await getErrorMessage(
        response,
        "The AI request failed.",
      ),
    );
  }

  if (!response.body) {
    throw new Error(
      "The AI backend returned no streaming response.",
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";
  let finalData: AskResponse | null = null;

  while (true) {
    const { value, done } = await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(value, {
      stream: true,
    });

    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const rawLine of lines) {
      const line = rawLine.trim();

      if (!line.startsWith("data: ")) {
        continue;
      }

      let event: {
        type?: string;
        content?: string;
        error?: string;
        data?: AskResponse;
      };

      try {
        event = JSON.parse(line.slice(6));
      } catch {
        throw new Error(
          "Received malformed streaming data from the AI backend.",
        );
      }

      if (event.type === "token") {
        const token = String(
          event.content ?? "",
        );

        if (token) {
          payload.onToken?.(token);
        }
      } else if (event.type === "done") {
        if (
          event.data &&
          typeof event.data === "object"
        ) {
          finalData = event.data;
        }
      } else if (event.type === "error") {
        throw new Error(
          event.error ||
            "Streaming generation failed.",
        );
      }
    }
  }

  if (!finalData) {
    throw new Error(
      "The AI backend finished without returning a final response.",
    );
  }

  return finalData;
}