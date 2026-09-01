import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  BookOpen,
  Check,
  Clock3,
  ExternalLink,
  FileText,
  Loader2,
  MoreHorizontal,
  Plus,
  Search,
  Trash2,
  Upload,
  WandSparkles,
  X,
  Code2,
} from "lucide-react";

import {
  deleteWorkspaceDocument,
  deleteWorkspaceSource,
  getDocumentPreview,
  getWorkspaceDocuments,
  getWorkspaceSources,
  uploadWorkspaceDocument,
  type Workspace,
  type WorkspaceDocument,
  type WorkspaceSource,
  type DocumentPreview,
} from "../lib/api";

type Filter =
  | "all"
  | "documents"
  | "papers"
  | "repositories"
  | "models";

interface SourcesPageProps {
  workspace: Workspace;
  onAskAI?: (documentIds: string[]) => void;
}

export function SourcesPage({
  workspace,
  onAskAI,
}: SourcesPageProps) {
  const [sources, setSources] = useState<
    WorkspaceSource[]
  >([]);
  const [documents, setDocuments] = useState<
    WorkspaceDocument[]
  >([]);

  const [selectedDocumentIds, setSelectedDocumentIds] =
    useState<Set<string>>(new Set());

  const [selectedSourceIds, setSelectedSourceIds] =
    useState<Set<string>>(new Set());

  const [activeFilter, setActiveFilter] =
    useState<Filter>("all");

  const [searchQuery, setSearchQuery] =
    useState("");

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState<string | null>(null);

  const [
    uploading,
    setUploading,
  ] = useState(false);

  const fileInputRef =
    useRef<HTMLInputElement | null>(null);

  const [
    openMenuId,
    setOpenMenuId,
  ] = useState<string | null>(null);

  const [
    deleteTarget,
    setDeleteTarget,
  ] = useState<{
    kind: "document" | "source";
    id: string;
    name: string;
  } | null>(null);

  const [
    deleting,
    setDeleting,
  ] = useState(false);

  const [
    previewTarget,
    setPreviewTarget,
  ] = useState<WorkspaceDocument | null>(
    null,
  );

  const [
    preview,
    setPreview,
  ] = useState<DocumentPreview | null>(
    null,
  );

  const [
    previewLoading,
    setPreviewLoading,
  ] = useState(false);

  const [
    previewError,
    setPreviewError,
  ] = useState<string | null>(null);

  const loadSources = async () => {
    setLoading(true);
    setError(null);

    try {
      const [
        loadedSources,
        loadedDocuments,
      ] = await Promise.all([
        getWorkspaceSources(
          workspace.id,
        ),
        getWorkspaceDocuments(
          workspace.id,
        ),
      ]);

      setSources(
        Array.isArray(loadedSources)
          ? loadedSources
          : [],
      );

      setDocuments(
        Array.isArray(loadedDocuments)
          ? loadedDocuments
          : [],
      );
    } catch (loadError) {
      setError(
        loadError instanceof Error
          ? loadError.message
          : "Unable to load sources.",
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadSources();
  }, [workspace.id]);

  const filteredItems = useMemo(() => {
    const query =
      searchQuery.trim().toLowerCase();

    // arXiv papers are represented by a user-facing workspace source,
    // while their downloaded PDF is stored internally as a workspace document.
    // Never expose that internal document as a second source.
    const arxivDocumentIds = new Set(
      sources
        .filter(
          (source) =>
            source.source_type.toLowerCase() === "arxiv",
        )
        .map((source) => getLinkedDocumentId(source))
        .filter((documentId): documentId is string => Boolean(documentId)),
    );

    const documentItems =
      documents
        .filter(
          (document) =>
            !arxivDocumentIds.has(document.document_id),
        )
        .map((document) => ({
        kind: "document" as const,
        id: document.document_id,
        title:
          document.filename ||
          "Untitled document",
        subtitle:
          document.content_type ||
          "Document",
        searchText: [
          document.filename,
          document.content_type,
          document.status,
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase(),
        document,
      }));

    const sourceItems =
      sources.map((source) => ({
        kind: "source" as const,
        id: source.id,
        title:
          source.title ||
          "Untitled source",
        subtitle: source.source_type,
        searchText: [
          source.title,
          source.source_type,
          source.url,
          JSON.stringify(
            source.metadata ?? {},
          ),
        ]
          .filter(Boolean)
          .join(" ")
          .toLowerCase(),
        source,
      }));

    const items = [
      ...documentItems,
      ...sourceItems,
    ];

    return items.filter((item) => {
      const matchesFilter =
        activeFilter === "all" ||
        (activeFilter === "documents" &&
          item.kind === "document") ||
        (activeFilter === "papers" &&
          item.kind === "source" &&
          item.source.source_type.toLowerCase() ===
            "arxiv") ||
        (activeFilter ===
          "repositories" &&
          item.kind === "source" &&
          item.source.source_type.toLowerCase() ===
            "github") ||
        (activeFilter === "models" &&
          item.kind === "source" &&
          item.source.source_type
            .toLowerCase() ===
            "huggingface");

      if (!matchesFilter) {
        return false;
      }

      if (!query) {
        return true;
      }

      return item.searchText.includes(
        query,
      );
    });
  }, [
    documents,
    sources,
    activeFilter,
    searchQuery,
  ]);

  const handleFileSelection =
    async (
      event: React.ChangeEvent<HTMLInputElement>,
    ) => {
      const file =
        event.target.files?.[0];

      event.target.value = "";

      if (!file) {
        return;
      }

      setUploading(true);
      setError(null);

      try {
        const created =
          await uploadWorkspaceDocument(
            workspace.id,
            file,
          );

        setDocuments(
          (current) => {
            const existingIndex =
              current.findIndex(
                (item) =>
                  item.id ===
                  created.id,
              );

            if (existingIndex >= 0) {
              const next = [
                ...current,
              ];

              next[existingIndex] =
                created;

              return next;
            }

            return [
              created,
              ...current,
            ];
          },
        );
      } catch (uploadError) {
        setError(
          uploadError instanceof Error
            ? uploadError.message
            : `Could not add document — ${file.name}`,
        );
      } finally {
        setUploading(false);
      }
    };

  const openPreview = async (
    document: WorkspaceDocument,
  ) => {
    setOpenMenuId(null);
    setPreviewTarget(document);
    setPreview(null);
    setPreviewError(null);
    setPreviewLoading(true);

    try {
      const result =
        await getDocumentPreview(
          workspace.id,
          document.document_id,
        );

      setPreview(result);
    } catch (previewLoadError) {
      setPreviewError(
        previewLoadError instanceof Error
          ? previewLoadError.message
          : "Unable to preview document.",
      );
    } finally {
      setPreviewLoading(false);
    }
  };

  const confirmDelete =
    async () => {
      if (!deleteTarget) {
        return;
      }

      setDeleting(true);
      setError(null);

      try {
        if (
          deleteTarget.kind ===
          "document"
        ) {
          await deleteWorkspaceDocument(
            workspace.id,
            deleteTarget.id,
          );

          setDocuments(
            (current) =>
              current.filter(
                (document) =>
                  document.id !==
                  deleteTarget.id,
              ),
          );
          setSelectedDocumentIds((current) => {
            const next = new Set(current);
            const deleted = documents.find(
              (document) =>
                document.document_id ===
                deleteTarget.id,
            );
            if (deleted) {
              next.delete(deleted.document_id);
            }
            return next;
          });
        } else {
          await deleteWorkspaceSource(
            workspace.id,
            deleteTarget.id,
          );

          setSources(
            (current) =>
              current.filter(
                (source) =>
                  source.id !==
                  deleteTarget.id,
              ),
          );
          setSelectedSourceIds((current) => {
            const next = new Set(current);
            next.delete(deleteTarget.id);
            return next;
          });
        }

        setDeleteTarget(null);
      } catch (deleteError) {
        setError(
          deleteError instanceof Error
            ? deleteError.message
            : "Unable to remove item.",
        );
      } finally {
        setDeleting(false);
      }
    };

  const toggleDocumentSelection = (documentId: string) => {
    setSelectedDocumentIds((current) => {
      const next = new Set(current);
      if (next.has(documentId)) {
        next.delete(documentId);
      } else {
        next.add(documentId);
      }
      return next;
    });
  };

  const clearDocumentSelection = () => {
    setSelectedDocumentIds(new Set());
  };

  const toggleSourceSelection = (sourceId: string) => {
    setSelectedSourceIds((current) => {
      const next = new Set(current);

      if (next.has(sourceId)) {
        next.delete(sourceId);
      } else {
        next.add(sourceId);
      }

      return next;
    });
  };

  const clearSourceSelection = () => {
    setSelectedSourceIds(new Set());
  };

  const handleAskAI = () => {
    if (
      !selectedDocumentIds.size &&
      !selectedSourceIds.size
    ) {
      return;
    }

    // A paper is selected through workspace_sources, but its actual content
    // is retrieved through the linked document_id. Resolve that relationship
    // here without exposing the internal PDF as a separate source.
    const selectedArxivDocumentIds = sources
      .filter(
        (item) =>
          selectedSourceIds.has(item.id) &&
          item.source_type.toLowerCase() === "arxiv",
      )
      .map((item) => getLinkedDocumentId(item))
      .filter((documentId): documentId is string => Boolean(documentId));

    const selectedDocuments = Array.from(
      new Set([
        ...selectedDocumentIds,
        ...selectedArxivDocumentIds,
      ]),
    );

    const selectedGithub = sources
      .filter(
        (item) =>
          selectedSourceIds.has(item.id) &&
          (
            item.source_type.toLowerCase() === "github" ||
            item.source_type.toLowerCase() === "github_repository"
          ) &&
          Boolean(item.url),
      )
      .map(
        (item) => `github:${item.url}`,
      );

    onAskAI?.([
      ...selectedDocuments,
      ...selectedGithub,
    ]);
  };

  const arxivDocumentIds = useMemo(
    () =>
      new Set(
        sources
          .filter(
            (source) =>
              source.source_type.toLowerCase() === "arxiv",
          )
          .map((source) => getLinkedDocumentId(source))
          .filter((documentId): documentId is string => Boolean(documentId)),
      ),
    [sources],
  );

  const visibleDocumentCount =
    documents.filter(
      (document) =>
        !arxivDocumentIds.has(document.document_id),
    ).length;

  const totalCount =
    sources.length + visibleDocumentCount;

  const documentCount =
    visibleDocumentCount;

  const paperCount =
    sources.filter(
      (source) =>
        source.source_type
          .toLowerCase() === "arxiv",
    ).length;

  const repositoryCount =
    sources.filter(
      (source) =>
        source.source_type
          .toLowerCase() === "github",
    ).length;

  return (
    <>
      <div
        className="min-h-full bg-[var(--paper)] text-[var(--ink)]"
        onClick={() =>
          setOpenMenuId(null)
        }
      >
        {/* ====================================================
            HEADER
        ==================================================== */}

        <div className="border-b border-[var(--line)]">
          <div className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
            <div className="flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <div className="flex items-center gap-2 font-[var(--font-mono)] text-[10px] uppercase tracking-[0.14em] text-[var(--muted)]">
                  <span className="h-1.5 w-1.5 rounded-full bg-[var(--cyan)]" />
                  Research sources
                </div>

                <h1 className="mt-2 font-[var(--font-display)] text-3xl font-semibold tracking-[-0.025em] sm:text-4xl">
                  Sources
                </h1>

                <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--ink-soft)]">
                  Everything collected for{" "}
                  <span className="font-medium text-[var(--ink)]">
                    {workspace.name}
                  </span>{" "}
                  in one place.
                </p>
              </div>

              <div>
                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  onChange={
                    handleFileSelection
                  }
                />

                <button
                  type="button"
                  disabled={uploading}
                  onClick={() =>
                    fileInputRef.current?.click()
                  }
                  className="group inline-flex items-center gap-2 rounded-md bg-[var(--ink)] px-4 py-2.5 text-sm font-medium text-[var(--paper)] shadow-sm transition-all duration-200 hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-md disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {uploading ? (
                    <Loader2
                      size={15}
                      className="animate-spin"
                    />
                  ) : (
                    <Upload size={15} />
                  )}

                  {uploading
                    ? "Adding..."
                    : "Add document"}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* ====================================================
            TOOLBAR
        ==================================================== */}

        <div className="border-b border-[var(--line-soft)]">
          <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 py-4 lg:px-8 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex flex-wrap gap-1.5">
              <FilterButton
                label={`All ${totalCount}`}
                active={
                  activeFilter === "all"
                }
                onClick={() =>
                  setActiveFilter("all")
                }
              />

              <FilterButton
                label={`Documents ${documentCount}`}
                active={
                  activeFilter ===
                  "documents"
                }
                onClick={() =>
                  setActiveFilter(
                    "documents",
                  )
                }
              />

              <FilterButton
                label={`Papers ${paperCount}`}
                active={
                  activeFilter ===
                  "papers"
                }
                onClick={() =>
                  setActiveFilter(
                    "papers",
                  )
                }
              />

              <FilterButton
                label={`Repositories ${repositoryCount}`}
                active={
                  activeFilter ===
                  "repositories"
                }
                onClick={() =>
                  setActiveFilter(
                    "repositories",
                  )
                }
              />

              <FilterButton
                label="Models"
                active={
                  activeFilter ===
                  "models"
                }
                onClick={() =>
                  setActiveFilter(
                    "models",
                  )
                }
              />
            </div>

            <div className="relative w-full sm:w-64">
              <Search
                size={14}
                className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted)]"
              />

              <input
                value={searchQuery}
                onChange={(event) =>
                  setSearchQuery(
                    event.target.value,
                  )
                }
                placeholder="Search sources"
                className="w-full rounded-md border border-[var(--line)] bg-[var(--paper)] py-2.5 pl-9 pr-3 text-xs text-[var(--ink)] outline-none transition-all duration-200 placeholder:text-[var(--muted)] focus:border-[var(--ink-soft)] focus:ring-2 focus:ring-[var(--line-soft)]"
              />
            </div>
          </div>
        </div>

        {/* ====================================================
            MAIN
        ==================================================== */}

        <main className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
          {error && (
            <div className="mb-5 flex items-start gap-3 rounded-xl border border-[var(--line)] bg-[var(--accent-dim)] px-4 py-3 text-xs text-[var(--accent)]">
              <AlertCircle
                size={15}
                className="mt-0.5 shrink-0"
              />

              <div className="min-w-0 flex-1">
                <p className="font-semibold">
                  Something went wrong
                </p>

                <p className="mt-1 leading-5">
                  {error}
                </p>
              </div>

              <button
                type="button"
                onClick={() =>
                  setError(null)
                }
                className="rounded-md p-1 transition-colors hover:bg-white/40"
                aria-label="Dismiss"
              >
                <X size={14} />
              </button>
            </div>
          )}

          {loading ? (
            <LoadingState />
          ) : filteredItems.length ===
            0 ? (
            <EmptyState
              hasQuery={
                searchQuery.trim()
                  .length > 0
              }
              onAddDocument={() =>
                fileInputRef.current?.click()
              }
            />
          ) : (
            <div className="space-y-2">
              {filteredItems.map(
                (item) => {
                  if (
                    item.kind ===
                    "document"
                  ) {
                    return (
                      <DocumentRow
                        key={`document-${item.id}`}
                        document={
                          item.document
                        }
                        menuOpen={
                          openMenuId ===
                          `document-${item.id}`
                        }
                        onToggleMenu={() =>
                          setOpenMenuId(
                            openMenuId ===
                              `document-${item.id}`
                              ? null
                              : `document-${item.id}`,
                          )
                        }
                        onPreview={() =>
                          void openPreview(
                            item.document,
                          )
                        }
                        onDelete={() =>
                          setDeleteTarget(
                            {
                              kind: "document",
                              id: item.document.document_id,
                              name: item.document.filename,
                            },
                          )
                        }
                        selected={selectedDocumentIds.has(
                          item.document.document_id,
                        )}
                        onToggleSelection={() =>
                          toggleDocumentSelection(
                            item.document.document_id,
                          )
                        }
                      />
                    );
                  }

                  return (
                    <SourceRow
                      key={`source-${item.id}`}
                      source={item.source}
                      selected={
                        (
                          item.source.source_type.toLowerCase() === "arxiv" ||
                          item.source.source_type.toLowerCase() === "github" ||
                          item.source.source_type.toLowerCase() === "github_repository"
                        ) &&
                        selectedSourceIds.has(item.source.id)
                      }
                      onToggleSelection={
                        (
                          item.source.source_type.toLowerCase() === "arxiv" ||
                          item.source.source_type.toLowerCase() === "github" ||
                          item.source.source_type.toLowerCase() === "github_repository"
                        )
                          ? () =>
                              toggleSourceSelection(
                                item.source.id,
                              )
                          : undefined
                      }
                      menuOpen={
                        openMenuId ===
                        `source-${item.id}`
                      }
                      onToggleMenu={() =>
                        setOpenMenuId(
                          openMenuId ===
                            `source-${item.id}`
                            ? null
                            : `source-${item.id}`,
                        )
                      }
                      onOpen={() => {
                        if (
                          item.source.url
                        ) {
                          window.open(
                            item.source.url,
                            "_blank",
                            "noopener,noreferrer",
                          );
                        }
                      }}
                      onDelete={() =>
                        setDeleteTarget(
                          {
                            kind: "source",
                            id: item.source.id,
                            name: item.source.title,
                          },
                        )
                      }
                    />
                  );
                },
              )}
            </div>
          )}
        </main>
      </div>

      {(selectedDocumentIds.size > 0 ||
        selectedSourceIds.size > 0) && (
        <div className="pointer-events-none fixed inset-x-0 bottom-5 z-40 flex justify-center px-4">
          <div className="pointer-events-auto flex w-full max-w-2xl items-center justify-between gap-4 rounded-2xl border border-[var(--line)] bg-[var(--ink)] px-4 py-3 text-[var(--paper)] shadow-xl sm:px-5">
            <div className="min-w-0">
              <p className="font-[var(--font-display)] text-sm font-semibold">
                {selectedDocumentIds.size +
                  selectedSourceIds.size}{" "}
                source
                {selectedDocumentIds.size +
                  selectedSourceIds.size === 1
                  ? ""
                  : "s"}{" "}
                selected
              </p>
              <p className="mt-0.5 text-[10px] text-[var(--paper)]/60">
                Start a new AI conversation with the selected sources.
              </p>
            </div>

            <div className="flex shrink-0 items-center gap-2">
              <button
                type="button"
                onClick={() => {
                  clearDocumentSelection();
                  clearSourceSelection();
                }}
                className="rounded-md px-3 py-2 text-[10px] font-medium text-[var(--paper)]/70 transition-colors hover:bg-white/10 hover:text-[var(--paper)]"
              >
                Clear
              </button>

              <button
                type="button"
                onClick={handleAskAI}
                disabled={!onAskAI}
                className="inline-flex items-center gap-2 rounded-md bg-[var(--paper)] px-3.5 py-2.5 text-[11px] font-semibold text-[var(--ink)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--cyan-dim)] hover:shadow-md disabled:cursor-not-allowed disabled:opacity-50"
              >
                <WandSparkles size={13} />
                Ask AI
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ======================================================
          PREVIEW
      ====================================================== */}

      {previewTarget && (
        <PreviewModal
          document={previewTarget}
          preview={preview}
          loading={previewLoading}
          error={previewError}
          onClose={() => {
            setPreviewTarget(null);
            setPreview(null);
            setPreviewError(null);
          }}
        />
      )}

      {/* ======================================================
          DELETE
      ====================================================== */}

      {deleteTarget && (
        <DeleteDialog
          target={deleteTarget}
          deleting={deleting}
          onCancel={() => {
            if (!deleting) {
              setDeleteTarget(null);
            }
          }}
          onConfirm={() =>
            void confirmDelete()
          }
        />
      )}
    </>
  );
}

/* ============================================================
   FILTER
   ============================================================ */

function FilterButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "rounded-md border px-3 py-2 text-xs font-medium transition-all duration-200",
        active
          ? "border-[var(--ink)] bg-[var(--ink)] text-[var(--paper)] shadow-sm"
          : "border-[var(--line)] bg-[var(--paper)] text-[var(--ink-soft)] hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]",
      ].join(" ")}
    >
      {label}
    </button>
  );
}

/* ============================================================
   DOCUMENT ROW
   ============================================================ */

function DocumentRow({
  document,
  menuOpen,
  onToggleMenu,
  onPreview,
  onDelete,
  selected,
  onToggleSelection,
}: {
  document: WorkspaceDocument;
  menuOpen: boolean;
  onToggleMenu: () => void;
  onPreview: () => void;
  onDelete: () => void;
  selected: boolean;
  onToggleSelection: () => void;
}) {
  return (
    <div
      className={[
        "group flex flex-col gap-4 rounded-xl border bg-[var(--paper)] p-4 transition-all duration-200 hover:-translate-y-px hover:shadow-sm sm:flex-row sm:items-center",
        selected
          ? "border-[var(--cyan)] ring-1 ring-[var(--cyan-dim)]"
          : "border-[var(--line)]",
      ].join(" ")}
    >
      <button
        type="button"
        onClick={onToggleSelection}
        className={[
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border transition-all duration-200",
          selected
            ? "border-[var(--cyan)] bg-[var(--cyan)] text-[var(--paper)]"
            : "border-[var(--line)] bg-[var(--paper)] text-transparent hover:bg-[var(--paper-dim)]",
        ].join(" ")}
        aria-label={
          selected
            ? `Remove ${document.filename} from AI selection`
            : `Select ${document.filename} for AI`
        }
      >
        <Check size={14} />
      </button>

      <div className="flex min-w-0 flex-1 items-center gap-3">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-[var(--line)] bg-[var(--paper-dim)] text-[var(--ink-soft)]">
          <FileText size={18} />
        </div>

        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-[var(--ink)]">
            {document.filename}
          </p>

          <div className="mt-1 flex flex-wrap items-center gap-2 text-[10px] text-[var(--muted)]">
            <span className="rounded-md bg-[var(--paper-dim)] px-2 py-1 font-[var(--font-mono)] uppercase tracking-[0.05em]">
              {formatContentType(
                document.content_type,
              )}
            </span>

            {document.pages !==
              null &&
              document.pages !==
                undefined && (
                <span>
                  {document.pages}{" "}
                  {document.pages === 1
                    ? "page"
                    : "pages"}
                </span>
              )}

            <span className="h-1 w-1 rounded-full bg-[var(--line)]" />

            <span className="inline-flex items-center gap-1">
              <Check
                size={11}
                className="text-[var(--cyan)]"
              />
              {capitalize(
                document.status ||
                  "ready",
              )}
            </span>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 sm:shrink-0">
        <button
          type="button"
          onClick={onPreview}
          className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] px-3 py-2 text-xs font-medium text-[var(--ink-soft)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] hover:shadow-sm"
        >
          <BookOpen size={13} />
          Preview
        </button>

        <div className="relative">
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onToggleMenu();
            }}
            className="flex h-8 w-8 items-center justify-center rounded-md border border-[var(--line)] text-[var(--muted)] transition-all duration-200 hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
            aria-label="Document actions"
          >
            <MoreHorizontal size={15} />
          </button>

          {menuOpen && (
            <ActionMenu
              onPrimary={onPreview}
              primaryLabel="Preview"
              onDelete={onDelete}
            />
          )}
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   SOURCE ROW
   ============================================================ */

function SourceRow({
  source,
  menuOpen,
  onToggleMenu,
  onOpen,
  onDelete,
  selected,
  onToggleSelection,
}: {
  source: WorkspaceSource;
  menuOpen: boolean;
  onToggleMenu: () => void;
  onOpen: () => void;
  onDelete: () => void;
  selected?: boolean;
  onToggleSelection?: () => void;
}) {
  const type =
    source.source_type.toLowerCase();

  return (
    <div
      className={[
        "group flex flex-col gap-4 rounded-xl border bg-[var(--paper)] p-4 transition-all duration-200 hover:-translate-y-px hover:shadow-sm sm:flex-row sm:items-center",
        selected
          ? "border-[var(--cyan)] ring-1 ring-[var(--cyan-dim)]"
          : "border-[var(--line)]",
      ].join(" ")}
    >
      {onToggleSelection && (
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            onToggleSelection();
          }}
          className={[
            "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border transition-all duration-200",
            selected
              ? "border-[var(--cyan)] bg-[var(--cyan)] text-[var(--paper)]"
              : "border-[var(--line)] bg-[var(--paper)] text-transparent hover:bg-[var(--paper-dim)]",
          ].join(" ")}
          aria-label={
            selected
              ? `Remove ${source.title} from AI selection`
              : `Select ${source.title} for AI`
          }
        >
          <Check size={14} />
        </button>
      )}

      <div className="flex min-w-0 flex-1 items-center gap-3">
        <SourceIcon type={type} />

        <div className="min-w-0">
          <p className="truncate text-sm font-semibold text-[var(--ink)]">
            {source.title}
          </p>

          <div className="mt-1 flex flex-wrap items-center gap-2">
            <span className="font-[var(--font-mono)] text-[10px] uppercase tracking-[0.08em] text-[var(--muted)]">
              {sourceTypeLabel(
                type,
              )}
            </span>

            {source.url && (
              <>
                <span className="h-1 w-1 rounded-full bg-[var(--line)]" />

                <span className="max-w-xs truncate text-[10px] text-[var(--muted)]">
                  {source.url}
                </span>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2 sm:shrink-0">
        {source.url && (
          <button
            type="button"
            onClick={onOpen}
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--line)] px-3 py-2 text-xs font-medium text-[var(--ink-soft)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] hover:shadow-sm"
          >
            <ExternalLink size={13} />
            Open
          </button>
        )}

        <div className="relative">
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation();
              onToggleMenu();
            }}
            className="flex h-8 w-8 items-center justify-center rounded-md border border-[var(--line)] text-[var(--muted)] transition-all duration-200 hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
            aria-label="Source actions"
          >
            <MoreHorizontal size={15} />
          </button>

          {menuOpen && (
            <ActionMenu
              onPrimary={onOpen}
              primaryLabel="Open source"
              onDelete={onDelete}
              primaryDisabled={
                !source.url
              }
            />
          )}
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   SOURCE ICON
   ============================================================ */

function SourceIcon({
  type,
}: {
  type: string;
}) {
  if (type === "github") {
    return (
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-[var(--line)] bg-[var(--paper-dim)] text-[var(--ink)]">
        <Code2 size={18} />
      </div>
    );
  }

  if (type === "arxiv") {
    return (
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-[var(--line)] bg-[var(--cyan-dim)] text-[var(--cyan)]">
        <BookOpen size={18} />
      </div>
    );
  }

  return (
    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-[var(--line)] bg-[var(--paper-dim)] text-[var(--ink-soft)]">
      <Search size={18} />
    </div>
  );
}

/* ============================================================
   MENU
   ============================================================ */

function ActionMenu({
  onPrimary,
  primaryLabel,
  onDelete,
  primaryDisabled = false,
}: {
  onPrimary: () => void;
  primaryLabel: string;
  onDelete: () => void;
  primaryDisabled?: boolean;
}) {
  return (
    <div
      className="absolute right-0 top-10 z-30 w-40 overflow-hidden rounded-lg border border-[var(--line)] bg-[var(--paper)] p-1.5 shadow-lg"
      onClick={(event) =>
        event.stopPropagation()
      }
    >
      <button
        type="button"
        disabled={primaryDisabled}
        onClick={onPrimary}
        className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-xs font-medium text-[var(--ink-soft)] transition-colors hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] disabled:cursor-not-allowed disabled:opacity-40"
      >
        <ExternalLink size={13} />
        {primaryLabel}
      </button>

      <button
        type="button"
        onClick={onDelete}
        className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-xs font-medium text-[var(--accent)] transition-colors hover:bg-[var(--accent-dim)]"
      >
        <Trash2 size={13} />
        Remove
      </button>
    </div>
  );
}

/* ============================================================
   PREVIEW MODAL
   ============================================================ */

function PreviewModal({
  document,
  preview,
  loading,
  error,
  onClose,
}: {
  document: WorkspaceDocument;
  preview: DocumentPreview | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
}) {
  useEffect(() => {
    const handleKeyDown = (
      event: KeyboardEvent,
    ) => {
      if (event.key === "Escape") {
        onClose();
      }
    };

    window.document.addEventListener(
      "keydown",
      handleKeyDown,
    );

    return () =>
      window.document.removeEventListener(
        "keydown",
        handleKeyDown,
      );
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-[90] flex items-center justify-center bg-[rgba(18,35,61,0.22)] p-4 backdrop-blur-sm sm:p-6"
      onMouseDown={(event) => {
        if (
          event.target ===
          event.currentTarget
        ) {
          onClose();
        }
      }}
    >
      <div
        className="flex h-[min(86vh,860px)] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-[var(--line)] bg-[var(--paper)] shadow-2xl"
        onMouseDown={(event) =>
          event.stopPropagation()
        }
      >
        <div className="flex items-center justify-between border-b border-[var(--line)] px-5 py-4">
          <div className="min-w-0">
            <p className="font-[var(--font-display)] text-sm font-semibold">
              {document.filename}
            </p>

            <p className="mt-1 font-[var(--font-mono)] text-[9px] uppercase tracking-[0.1em] text-[var(--muted)]">
              Readable preview
            </p>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-[var(--line)] text-[var(--muted)] transition-all duration-200 hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
            aria-label="Close preview"
          >
            <X size={15} />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-auto bg-[var(--paper-dim)] p-4 sm:p-7">
          {loading ? (
            <div className="flex h-full items-center justify-center">
              <div className="text-center">
                <Loader2
                  size={20}
                  className="mx-auto animate-spin text-[var(--ink-soft)]"
                />

                <p className="mt-3 text-sm font-medium">
                  Preparing preview
                </p>

                <p className="mt-1 text-xs text-[var(--muted)]">
                  Loading extracted document content
                </p>
              </div>
            </div>
          ) : error ? (
            <div className="flex h-full items-center justify-center">
              <div className="max-w-md rounded-xl border border-[var(--line)] bg-[var(--paper)] p-6 text-center">
                <AlertCircle
                  size={20}
                  className="mx-auto text-[var(--accent)]"
                />

                <h3 className="mt-3 text-sm font-semibold">
                  Preview unavailable
                </h3>

                <p className="mt-2 text-xs leading-5 text-[var(--ink-soft)]">
                  {error}
                </p>
              </div>
            </div>
          ) : (
            <article className="mx-auto min-h-full max-w-3xl rounded-xl border border-[var(--line-soft)] bg-[var(--paper)] px-6 py-7 shadow-sm sm:px-10 sm:py-9">
              <div className="mb-7 flex flex-wrap items-center gap-2 border-b border-[var(--line-soft)] pb-5">
                <span className="rounded-md bg-[var(--paper-dim)] px-2.5 py-1.5 font-[var(--font-mono)] text-[9px] uppercase tracking-[0.08em] text-[var(--muted)]">
                  {formatContentType(
                    document.content_type,
                  )}
                </span>

                {document.pages !==
                  null &&
                  document.pages !==
                    undefined && (
                    <span className="inline-flex items-center gap-1.5 text-[10px] text-[var(--muted)]">
                      <Clock3 size={11} />
                      {document.pages} pages
                    </span>
                  )}
              </div>

              <pre className="whitespace-pre-wrap break-words font-[var(--font-mono)] text-xs leading-6 text-[var(--ink-soft)]">
                {preview?.content ||
                  "No readable text is available for this document."}
              </pre>
            </article>
          )}
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   DELETE DIALOG
   ============================================================ */

function DeleteDialog({
  target,
  deleting,
  onCancel,
  onConfirm,
}: {
  target: {
    kind: "document" | "source";
    id: string;
    name: string;
  };
  deleting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-[rgba(18,35,61,0.22)] p-5 backdrop-blur-sm"
      onMouseDown={(event) => {
        if (
          event.target ===
          event.currentTarget
        ) {
          onCancel();
        }
      }}
    >
      <div
        className="w-full max-w-md rounded-2xl border border-[var(--line)] bg-[var(--paper)] p-7 shadow-2xl"
        onMouseDown={(event) =>
          event.stopPropagation()
        }
      >
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[var(--accent-dim)] text-[var(--accent)]">
          <Trash2 size={17} />
        </div>

        <p className="mt-6 font-[var(--font-mono)] text-[10px] uppercase tracking-[0.14em] text-[var(--muted)]">
          Remove {target.kind}
        </p>

        <h2 className="mt-2 font-[var(--font-display)] text-2xl font-semibold tracking-[-0.025em]">
          Remove this source?
        </h2>

        <p className="mt-3 text-sm leading-6 text-[var(--ink-soft)]">
          This will remove{" "}
          <span className="font-semibold text-[var(--ink)]">
            {target.name}
          </span>{" "}
          from the current workspace.
        </p>

        {target.kind ===
          "document" && (
          <p className="mt-2 text-xs leading-5 text-[var(--muted)]">
            If another workspace uses the same underlying
            document, its document data will remain available there.
          </p>
        )}

        <div className="mt-7 flex justify-end gap-2">
          <button
            type="button"
            disabled={deleting}
            onClick={onCancel}
            className="rounded-md border border-[var(--line)] px-4 py-2.5 text-sm font-medium text-[var(--ink-soft)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:shadow-sm disabled:opacity-40"
          >
            Cancel
          </button>

          <button
            type="button"
            disabled={deleting}
            onClick={onConfirm}
            className="inline-flex items-center gap-2 rounded-md bg-[var(--ink)] px-4 py-2.5 text-sm font-medium text-[var(--paper)] shadow-sm transition-all duration-200 hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-md disabled:cursor-not-allowed disabled:opacity-50"
          >
            {deleting ? (
              <Loader2
                size={14}
                className="animate-spin"
              />
            ) : (
              <Trash2 size={14} />
            )}

            {deleting
              ? "Removing..."
              : "Remove source"}
          </button>
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   EMPTY
   ============================================================ */

function EmptyState({
  hasQuery,
  onAddDocument,
}: {
  hasQuery: boolean;
  onAddDocument: () => void;
}) {
  return (
    <div className="rounded-2xl border border-dashed border-[var(--line)] bg-[var(--paper-dim)] px-6 py-16 text-center">
      <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl border border-[var(--line)] bg-[var(--paper)] text-[var(--ink-soft)]">
        <FileText size={18} />
      </div>

      <h3 className="mt-5 font-[var(--font-display)] text-lg font-semibold">
        {hasQuery
          ? "No matching sources"
          : "No sources yet"}
      </h3>

      <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-[var(--ink-soft)]">
        {hasQuery
          ? "Try another search or clear the filter."
          : "Add a document or discover research to start building this workspace."}
      </p>

      {!hasQuery && (
        <button
          type="button"
          onClick={onAddDocument}
          className="mt-6 inline-flex items-center gap-2 rounded-md bg-[var(--ink)] px-4 py-2.5 text-sm font-medium text-[var(--paper)] shadow-sm transition-all duration-200 hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-md"
        >
          <Plus size={14} />
          Add your first document
        </button>
      )}
    </div>
  );
}

/* ============================================================
   LOADING
   ============================================================ */

function LoadingState() {
  return (
    <div className="space-y-2">
      {[1, 2, 3].map((item) => (
        <div
          key={item}
          className="animate-pulse rounded-xl border border-[var(--line)] bg-[var(--paper)] p-5"
        >
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-lg bg-[var(--paper-dim)]" />

            <div className="min-w-0 flex-1">
              <div className="h-3 w-2/5 rounded bg-[var(--paper-dim)]" />
              <div className="mt-2 h-2.5 w-1/4 rounded bg-[var(--paper-dim)]" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ============================================================
   Helpers
   ============================================================ */

function getLinkedDocumentId(
  source: WorkspaceSource,
): string | null {
  const metadata = source.metadata;

  if (!metadata || typeof metadata !== "object") {
    return null;
  }

  const documentId = metadata["document_id"];

  return typeof documentId === "string" && documentId.trim()
    ? documentId.trim()
    : null;
}

function formatContentType(
  contentType: string | null,
) {
  if (!contentType) {
    return "FILE";
  }

  const type =
    contentType.split("/").pop() ??
    contentType;

  return type
    .replace("vnd.openxmlformats-officedocument.", "")
    .replace("application.", "")
    .toUpperCase();
}

function capitalize(
  value: string,
) {
  return (
    value.charAt(0).toUpperCase() +
    value.slice(1)
  );
}

function sourceTypeLabel(
  sourceType: string,
) {
  switch (sourceType) {
    case "arxiv":
      return "Paper";
    case "github":
      return "Repository";
    case "huggingface":
      return "Hugging Face";
    case "paperswithcode":
      return "PapersWithCode";
    default:
      return sourceType || "Source";
  }
}