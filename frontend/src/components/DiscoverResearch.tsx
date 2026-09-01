import { useEffect, useMemo, useState } from "react";
import {
  BookOpen,
  Check,
  ChevronDown,
  Clock3,
  ExternalLink,
  GitBranch,
  Heart,
  Loader2,
  Plus,
  Search,
  Sparkles,
  WandSparkles,
} from "lucide-react";

import {
  addResearchSourceToWorkspace,
  getWorkspaceSources,
  searchResearch,
  type ResearchItem,
  type Workspace,
} from "../lib/api";

interface DiscoverResearchProps {
  workspace: Workspace;
  onNavigateSources?: () => void;
}

const SOURCES = [
  { id: "arxiv", label: "arXiv" },
  { id: "github", label: "GitHub" },
  {
    id: "paperswithcode",
    label: "PapersWithCode",
  },
  {
    id: "huggingface",
    label: "Hugging Face",
  },
] as const;

type SearchMode =
  | "keyword"
  | "semantic"
  | "hybrid";

type SortMode =
  | "relevance"
  | "published"
  | "updated";

const SEARCH_MODES = [
  {
    id: "hybrid" as const,
    label: "Hybrid",
    description:
      "Keyword + semantic relevance.",
  },
  {
    id: "semantic" as const,
    label: "Semantic",
    description:
      "Find conceptually similar research.",
  },
  {
    id: "keyword" as const,
    label: "Keyword",
    description:
      "Match explicit terms and metadata.",
  },
];

const SORT_MODES = [
  {
    id: "relevance" as const,
    label: "Most relevant",
  },
  {
    id: "published" as const,
    label: "Newest",
  },
  {
    id: "updated" as const,
    label: "Recently updated",
  },
];

const RESULTS_PER_PAGE = 10;

export function DiscoverResearch({
  workspace,
  onNavigateSources,
}: DiscoverResearchProps) {
  const [query, setQuery] = useState("");

  const [selectedSources, setSelectedSources] =
    useState<string[]>(
      SOURCES.map(
        (source) => source.id,
      ),
    );

  const [searchMode, setSearchMode] =
    useState<SearchMode>("hybrid");

  const [sortBy, setSortBy] =
    useState<SortMode>("relevance");

  const [results, setResults] =
    useState<ResearchItem[]>([]);

  const [searched, setSearched] =
    useState(false);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState<string | null>(null);

  const [
    workspaceSourceKeys,
    setWorkspaceSourceKeys,
  ] = useState<Set<string>>(
    new Set(),
  );

  const [
    addingSourceKey,
    setAddingSourceKey,
  ] = useState<string | null>(
    null,
  );

  const [
    addedSourceKeys,
    setAddedSourceKeys,
  ] = useState<Set<string>>(
    new Set(),
  );

  const [
    activeFilter,
    setActiveFilter,
  ] = useState("all");

  const [
    sourceMenuOpen,
    setSourceMenuOpen,
  ] = useState(false);

  const [
    modeMenuOpen,
    setModeMenuOpen,
  ] = useState(false);

  const [
    sortMenuOpen,
    setSortMenuOpen,
  ] = useState(false);

  const [
    currentPage,
    setCurrentPage,
  ] = useState(1);

  useEffect(() => {
    void loadWorkspaceSources();
  }, [workspace.id]);

  async function loadWorkspaceSources() {
    try {
      const saved =
        await getWorkspaceSources(
          workspace.id,
        );

      setWorkspaceSourceKeys(
        new Set(
          saved
            .filter(
              (source) => source.url,
            )
            .map(
              (source) =>
                `${source.source_type.toLowerCase()}|${source.url?.trim()}`,
            ),
        ),
      );
    } catch {
      // Search remains usable even when
      // workspace-state detection fails.
    }
  }

  async function handleSearch() {
    const normalized =
      query.trim();

    if (!normalized) {
      setError(
        "Enter a research question, topic, paper, model, or method.",
      );
      setResults([]);
      setSearched(false);
      return;
    }

    if (
      selectedSources.length ===
      0
    ) {
      setError(
        "Select at least one research source.",
      );
      return;
    }

    setLoading(true);
    setError(null);
    setSearched(true);
    setActiveFilter("all");
    setCurrentPage(1);

    try {
      const found =
        await searchResearch({
          query: normalized,
          sources: selectedSources,
          sortBy,
          searchMode,
        });

      setResults(found);
    } catch (searchError) {
      setResults([]);

      setError(
        searchError instanceof Error
          ? searchError.message
          : "Research search failed.",
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleAdd(
    result: ResearchItem,
  ) {
    const key =
      makeSourceKey(result);

    if (
      workspaceSourceKeys.has(
        key,
      ) ||
      addedSourceKeys.has(key) ||
      addingSourceKey === key
    ) {
      return;
    }

    setAddingSourceKey(key);
    setError(null);

    try {
      await addResearchSourceToWorkspace(
        workspace.id,
        {
          source_type: result.source,
          title: result.title,
          url: result.url,
          metadata: {
            research_id:
              result.id,
            title: result.title,
            description:
              result.description,
            authors:
              result.authors,
            source:
              result.source,
            url: result.url,
            published:
              result.published,
            updated:
              result.updated,
            tags: result.tags,
            stars:
              result.stars,
            forks:
              result.forks,
            language:
              result.language,
            downloads:
              result.downloads,
            likes:
              result.likes,
            library:
              result.library,
            pipeline_tag:
              result.pipeline_tag,
            tasks:
              result.tasks,
            conference:
              result.conference,
            metadata:
              result.metadata,
          },
        },
      );

      setAddedSourceKeys(
        (current) => {
          const next =
            new Set(current);
          next.add(key);
          return next;
        },
      );

      setWorkspaceSourceKeys(
        (current) => {
          const next =
            new Set(current);
          next.add(key);
          return next;
        },
      );
    } catch (addError) {
      setError(
        addError instanceof Error
          ? addError.message
          : "Could not add source to workspace.",
      );
    } finally {
      setAddingSourceKey(
        null,
      );
    }
  }

  const counts =
    useMemo(() => {
      const next: Record<
        string,
        number
      > = {
        arxiv: 0,
        github: 0,
        paperswithcode: 0,
        huggingface: 0,
      };

      for (const result of results) {
        const key =
          result.source.toLowerCase();

        if (
          key in next
        ) {
          next[key] += 1;
        }
      }

      return next;
    }, [results]);

  const filteredResults =
    useMemo(() => {
      if (
        activeFilter === "all"
      ) {
        return results;
      }

      return results.filter(
        (result) =>
          result.source.toLowerCase() ===
          activeFilter,
      );
    }, [
      results,
      activeFilter,
    ]);

  const totalResults =
    filteredResults.length;

  const totalPages =
    Math.max(
      1,
      Math.ceil(
        totalResults /
          RESULTS_PER_PAGE,
      ),
    );

  const safePage = Math.min(
    currentPage,
    totalPages,
  );

  const startIndex =
    totalResults === 0
      ? 0
      : (safePage - 1) *
        RESULTS_PER_PAGE;

  const endIndex = Math.min(
    startIndex +
      RESULTS_PER_PAGE,
    totalResults,
  );

  const pageResults =
    filteredResults.slice(
      startIndex,
      endIndex,
    );

  const activeMode =
    SEARCH_MODES.find(
      (mode) =>
        mode.id ===
        searchMode,
    ) ??
    SEARCH_MODES[0];

  const activeSort =
    SORT_MODES.find(
      (mode) =>
        mode.id === sortBy,
    ) ??
    SORT_MODES[0];

  const sourceLabel =
    selectedSources.length ===
    SOURCES.length
      ? "All sources"
      : selectedSources.length ===
          1
        ? SOURCES.find(
            (source) =>
              source.id ===
              selectedSources[0],
          )?.label ??
          "1 source"
        : `${selectedSources.length} sources`;

  function toggleSource(
    sourceId: string,
  ) {
    setSelectedSources(
      (current) =>
        current.includes(
          sourceId,
        )
          ? current.filter(
              (item) =>
                item !== sourceId,
            )
          : [
              ...current,
              sourceId,
            ],
    );
  }

  return (
    <div
      className="min-h-full bg-[var(--paper)] text-[var(--ink)]"
      onClick={() => {
        setSourceMenuOpen(
          false,
        );
        setModeMenuOpen(false);
        setSortMenuOpen(false);
      }}
    >
      {/* =====================================================
          HEADER
      ====================================================== */}

      <section className="border-b border-[var(--line)]">
        <div className="mx-auto max-w-7xl px-6 py-10 lg:px-8">
          <div className="flex flex-col gap-7 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="flex items-center gap-2 font-[var(--font-mono)] text-[10px] uppercase tracking-[0.14em] text-[var(--muted)]">
                <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
                Research discovery
              </div>

              <h1 className="mt-3 max-w-2xl font-[var(--font-display)] text-4xl font-semibold leading-[1.02] tracking-[-0.035em] sm:text-5xl">
                Find the research behind the question.
              </h1>

              <p className="mt-4 max-w-2xl text-sm leading-6 text-[var(--ink-soft)]">
                Search papers, repositories,
                models, and research resources.
                Save the useful ones to this
                workspace and build your evidence base.
              </p>
            </div>

            {onNavigateSources && (
              <button
                type="button"
                onClick={
                  onNavigateSources
                }
                className="inline-flex items-center gap-2 self-start rounded-md border border-[var(--line)] px-4 py-2.5 text-xs font-medium text-[var(--ink-soft)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] hover:shadow-sm lg:self-auto"
              >
                <BookOpen size={14} />
                View saved sources
              </button>
            )}
          </div>
        </div>
      </section>

      {/* =====================================================
          SEARCH PANEL
      ====================================================== */}

      <section className="border-b border-[var(--line-soft)]">
        <div className="mx-auto max-w-7xl px-6 py-6 lg:px-8">
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void handleSearch();
            }}
            className="rounded-2xl border border-[var(--line)] bg-[var(--paper)] p-2 shadow-sm"
          >
            <div className="flex flex-col gap-2 lg:flex-row">
              <div className="relative min-w-0 flex-1">
                <Search
                  size={17}
                  className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-[var(--muted)]"
                />

                <input
                  value={query}
                  onChange={(event) =>
                    setQuery(
                      event.target
                        .value,
                    )
                  }
                  placeholder="What are you researching?"
                  className="w-full bg-transparent py-4 pl-11 pr-4 text-sm text-[var(--ink)] outline-none placeholder:text-[var(--muted)]"
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="inline-flex min-w-32 items-center justify-center gap-2 rounded-xl bg-[var(--ink)] px-5 py-3 text-sm font-medium text-[var(--paper)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-md disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <Loader2
                      size={15}
                      className="animate-spin"
                    />
                    Searching
                  </>
                ) : (
                  <>
                    <Sparkles
                      size={15}
                    />
                    Discover
                  </>
                )}
              </button>
            </div>

            <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-[var(--line-soft)] pt-2">
              {/* Source picker */}

              <div className="relative">
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();

                    setSourceMenuOpen(
                      (open) =>
                        !open,
                    );

                    setModeMenuOpen(
                      false,
                    );

                    setSortMenuOpen(
                      false,
                    );
                  }}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--line)] px-3 py-2 text-[11px] font-medium text-[var(--ink-soft)] transition-colors hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
                >
                  <Search size={12} />
                  {sourceLabel}
                  <ChevronDown
                    size={12}
                  />
                </button>

                {sourceMenuOpen && (
                  <div
                    className="absolute left-0 top-11 z-30 w-56 rounded-xl border border-[var(--line)] bg-[var(--paper)] p-2 shadow-xl"
                    onClick={(event) =>
                      event.stopPropagation()
                    }
                  >
                    <p className="px-2 py-1.5 font-[var(--font-mono)] text-[9px] uppercase tracking-[0.13em] text-[var(--muted)]">
                      Search across
                    </p>

                    {SOURCES.map(
                      (source) => (
                        <button
                          key={
                            source.id
                          }
                          type="button"
                          onClick={() =>
                            toggleSource(
                              source.id,
                            )
                          }
                          className="flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-left text-xs text-[var(--ink-soft)] transition-colors hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
                        >
                          <span>
                            {
                              source.label
                            }
                          </span>

                          {selectedSources.includes(
                            source.id,
                          ) && (
                            <Check
                              size={
                                13
                              }
                              className="text-[var(--cyan)]"
                            />
                          )}
                        </button>
                      ),
                    )}
                  </div>
                )}
              </div>

              {/* Search mode */}

              <Dropdown
                open={modeMenuOpen}
                onToggle={(event) => {
                  event.stopPropagation();

                  setModeMenuOpen(
                    (open) =>
                      !open,
                  );

                  setSourceMenuOpen(
                    false,
                  );

                  setSortMenuOpen(
                    false,
                  );
                }}
                label={
                  <>
                    <WandSparkles
                      size={12}
                    />
                    {
                      activeMode.label
                    }
                    <ChevronDown
                      size={12}
                    />
                  </>
                }
              >
                {SEARCH_MODES.map(
                  (mode) => (
                    <button
                      key={mode.id}
                      type="button"
                      onClick={() => {
                        setSearchMode(
                          mode.id,
                        );
                        setModeMenuOpen(
                          false,
                        );
                      }}
                      className="w-full rounded-lg px-2.5 py-2.5 text-left transition-colors hover:bg-[var(--paper-dim)]"
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold">
                          {
                            mode.label
                          }
                        </span>

                        {searchMode ===
                          mode.id && (
                          <Check
                            size={
                              13
                            }
                            className="text-[var(--cyan)]"
                          />
                        )}
                      </div>

                      <p className="mt-1 text-[10px] leading-4 text-[var(--muted)]">
                        {
                          mode.description
                        }
                      </p>
                    </button>
                  ),
                )}
              </Dropdown>

              {/* Sort */}

              <Dropdown
                open={sortMenuOpen}
                onToggle={(event) => {
                  event.stopPropagation();

                  setSortMenuOpen(
                    (open) =>
                      !open,
                  );

                  setSourceMenuOpen(
                    false,
                  );

                  setModeMenuOpen(
                    false,
                  );
                }}
                label={
                  <>
                    <Clock3
                      size={12}
                    />
                    {activeSort.label}
                    <ChevronDown
                      size={12}
                    />
                  </>
                }
              >
                {SORT_MODES.map(
                  (mode) => (
                    <button
                      key={mode.id}
                      type="button"
                      onClick={() => {
                        setSortBy(
                          mode.id,
                        );
                        setSortMenuOpen(
                          false,
                        );
                      }}
                      className="flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-left text-xs transition-colors hover:bg-[var(--paper-dim)]"
                    >
                      <span>
                        {
                          mode.label
                        }
                      </span>

                      {sortBy ===
                        mode.id && (
                        <Check
                          size={
                            13
                          }
                          className="text-[var(--cyan)]"
                        />
                      )}
                    </button>
                  ),
                )}
              </Dropdown>
            </div>
          </form>
        </div>
      </section>

      {/* =====================================================
          RESULTS
      ====================================================== */}

      <main className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
        {error && (
          <div className="mb-5 rounded-xl border border-[var(--line)] bg-[var(--accent-dim)] px-4 py-3 text-xs text-[var(--accent)]">
            {error}
          </div>
        )}

        {!searched ? (
          <DiscoveryIntro />
        ) : loading ? (
          <LoadingResults />
        ) : (
          <>
            <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-[var(--font-mono)] text-[10px] uppercase tracking-[0.13em] text-[var(--muted)]">
                    Discovery results
                  </span>

                  <span className="rounded-full bg-[var(--paper-dim)] px-2 py-0.5 font-[var(--font-mono)] text-[9px] text-[var(--muted)]">
                    {results.length}
                  </span>
                </div>

                <h2 className="mt-2 font-[var(--font-display)] text-xl font-semibold tracking-[-0.02em]">
                  {query.trim()}
                </h2>

                <p className="mt-1 text-xs text-[var(--muted)]">
                  {
                    activeMode.description
                  }
                </p>
              </div>

              <div className="flex flex-wrap gap-1.5">
                <FilterChip
                  label="All"
                  count={
                    results.length
                  }
                  active={
                    activeFilter ===
                    "all"
                  }
                  onClick={() => {
                    setActiveFilter("all");
                    setCurrentPage(1);
                  }}
                />

                {SOURCES.map(
                  (source) => (
                    <FilterChip
                      key={
                        source.id
                      }
                      label={
                        source.label
                      }
                      count={
                        counts[
                          source.id
                        ] ?? 0
                      }
                      active={
                        activeFilter ===
                        source.id
                      }
                      onClick={() => {
                        setActiveFilter(source.id);
                        setCurrentPage(1);
                      }}
                    />
                  ),
                )}
              </div>
            </div>

            {filteredResults.length ===
            0 ? (
              <div className="rounded-2xl border border-dashed border-[var(--line)] bg-[var(--paper-dim)] px-6 py-16 text-center">
                <Search
                  size={18}
                  className="mx-auto text-[var(--muted)]"
                />

                <h3 className="mt-4 font-[var(--font-display)] text-base font-semibold">
                  No results in this filter
                </h3>

                <p className="mt-1 text-xs text-[var(--muted)]">
                  Try another provider or
                  broaden the query.
                </p>
              </div>
            ) : (
              <div className="grid gap-3">
                {pageResults.map(
                  (result) => {
                    const key =
                      makeSourceKey(
                        result,
                      );

                    return (
                      <ResearchResultCard
                        key={`${result.source}-${result.id}-${result.url}`}
                        result={
                          result
                        }
                        saved={
                          workspaceSourceKeys.has(
                            key,
                          ) ||
                          addedSourceKeys.has(
                            key,
                          )
                        }
                        adding={
                          addingSourceKey ===
                          key
                        }
                        onAdd={() =>
                          void handleAdd(
                            result,
                          )
                        }
                      />
                    );
                  },
                )}
              </div>
            )}

            {totalPages > 1 && (
              <div className="mt-6 flex flex-col gap-3 border-t border-[var(--line-soft)] pt-5 sm:flex-row sm:items-center sm:justify-between">
                <p className="font-[var(--font-mono)] text-[9px] uppercase tracking-[0.1em] text-[var(--muted)]">
                  Showing{" "}
                  {startIndex + 1}
                  {"–"}
                  {endIndex}{" "}
                  of {totalResults}
                </p>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    disabled={safePage <= 1}
                    onClick={() =>
                      setCurrentPage(
                        (page) =>
                          Math.max(
                            1,
                            page - 1,
                          ),
                      )
                    }
                    className="rounded-md border border-[var(--line)] px-3 py-2 text-xs font-medium text-[var(--ink-soft)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    ← Previous
                  </button>

                  <div className="min-w-28 text-center font-[var(--font-mono)] text-[9px] uppercase tracking-[0.1em] text-[var(--muted)]">
                    Page {safePage} of{" "}
                    {totalPages}
                  </div>

                  <button
                    type="button"
                    disabled={
                      safePage >=
                      totalPages
                    }
                    onClick={() =>
                      setCurrentPage(
                        (page) =>
                          Math.min(
                            totalPages,
                            page + 1,
                          ),
                      )
                    }
                    className="rounded-md border border-[var(--line)] px-3 py-2 text-xs font-medium text-[var(--ink-soft)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Next →
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

/* ============================================================
   RESULT CARD
   ============================================================ */

function ResearchResultCard({
  result,
  saved,
  adding,
  onAdd,
}: {
  result: ResearchItem;
  saved: boolean;
  adding: boolean;
  onAdd: () => void;
}) {
  const source =
    result.source.toLowerCase();

  return (
    <article className="group rounded-2xl border border-[var(--line)] bg-[var(--paper)] p-5 transition-all duration-200 hover:-translate-y-px hover:shadow-md sm:p-6">
      <div className="flex flex-col gap-5 lg:flex-row lg:justify-between">
        <div className="min-w-0 flex-1">
          <div className="flex items-start gap-4">
            <ResearchIcon
              source={source}
            />

            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-[var(--font-mono)] text-[9px] uppercase tracking-[0.13em] text-[var(--muted)]">
                  {prettySource(
                    source,
                  )}
                </span>

                {result.published && (
                  <>
                    <span className="h-1 w-1 rounded-full bg-[var(--line)]" />

                    <span className="text-[10px] text-[var(--muted)]">
                      {formatDate(
                        result.published,
                      )}
                    </span>
                  </>
                )}
              </div>

              <h3 className="mt-2 max-w-4xl font-[var(--font-display)] text-lg font-semibold leading-6 tracking-[-0.015em]">
                {result.title}
              </h3>

              {result.description && (
                <p className="mt-2 max-w-4xl text-sm leading-6 text-[var(--ink-soft)]">
                  {
                    result.description
                  }
                </p>
              )}

              <ProviderMetadata
                result={result}
              />

              {result.authors
                .length > 0 && (
                <p className="mt-4 truncate text-xs text-[var(--muted)]">
                  {result.authors
                    .slice(0, 4)
                    .join(", ")}
                  {result.authors
                    .length >
                    4
                    ? ` +${result.authors.length - 4}`
                    : ""}
                </p>
              )}

              {result.tags
                .length > 0 && (
                <div className="mt-4 flex flex-wrap gap-1.5">
                  {result.tags
                    .slice(
                      0,
                      8,
                    )
                    .map(
                      (tag) => (
                        <span
                          key={tag}
                          className="rounded-md border border-[var(--line-soft)] bg-[var(--paper-dim)] px-2 py-1 text-[9px] text-[var(--ink-soft)]"
                        >
                          {tag}
                        </span>
                      ),
                    )}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex shrink-0 flex-row gap-2 lg:w-44 lg:flex-col">
          <button
            type="button"
            onClick={() =>
              window.open(
                result.url,
                "_blank",
                "noopener,noreferrer",
              )
            }
            className="flex flex-1 items-center justify-center gap-2 rounded-md border border-[var(--line)] px-3 py-2.5 text-xs font-medium text-[var(--ink-soft)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] hover:shadow-sm lg:flex-none"
          >
            <ExternalLink
              size={13}
            />
            Open source
          </button>

          <button
            type="button"
            disabled={
              saved || adding
            }
            onClick={onAdd}
            className={[
              "flex flex-1 items-center justify-center gap-2 rounded-md px-3 py-2.5 text-xs font-medium transition-all duration-200 lg:flex-none",
              saved
                ? "border border-[var(--line)] bg-[var(--paper-dim)] text-[var(--muted)]"
                : "bg-[var(--ink)] text-[var(--paper)] shadow-sm hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-md disabled:opacity-50",
            ].join(" ")}
          >
            {adding ? (
              <Loader2
                size={13}
                className="animate-spin"
              />
            ) : saved ? (
              <Check size={13} />
            ) : (
              <Plus size={13} />
            )}

            {adding
              ? "Adding..."
              : saved
                ? "In workspace"
                : "Add to workspace"}
          </button>
        </div>
      </div>

      <div className="mt-5 flex items-center justify-between gap-4 border-t border-[var(--line-soft)] pt-3">
        <span className="truncate text-[10px] text-[var(--muted)]">
          {result.url}
        </span>

        <span className="font-[var(--font-mono)] text-[9px] uppercase tracking-[0.12em] text-[var(--muted)]">
          Research result
        </span>
      </div>
    </article>
  );
}

/* ============================================================
   PROVIDER METADATA
   ============================================================ */

function ProviderMetadata({
  result,
}: {
  result: ResearchItem;
}) {
  const source =
    result.source.toLowerCase();

  if (source === "github") {
    const parts = [
      result.stars !== null &&
      result.stars !== undefined
        ? `${formatNumber(result.stars)} stars`
        : null,
      result.forks !== null &&
      result.forks !== undefined
        ? `${formatNumber(result.forks)} forks`
        : null,
      result.language ??
        null,
    ].filter(Boolean);

    return parts.length ? (
      <p className="mt-3 text-[10px] font-medium text-[var(--ink-soft)]">
        {parts.join("  ·  ")}
      </p>
    ) : null;
  }

  if (
    source === "huggingface"
  ) {
    const parts = [
      result.downloads !== null &&
      result.downloads !== undefined
        ? `${formatNumber(result.downloads)} downloads`
        : null,
      result.likes !== null &&
      result.likes !== undefined
        ? `${formatNumber(result.likes)} likes`
        : null,
      result.pipeline_tag ??
        null,
    ].filter(Boolean);

    return parts.length ? (
      <p className="mt-3 text-[10px] font-medium text-[var(--ink-soft)]">
        {parts.join("  ·  ")}
      </p>
    ) : null;
  }

  if (
    source ===
    "paperswithcode"
  ) {
    const parts = [
      ...result.tasks.slice(
        0,
        3,
      ),
      result.conference ??
        null,
    ].filter(Boolean);

    return parts.length ? (
      <p className="mt-3 text-[10px] font-medium text-[var(--ink-soft)]">
        {parts.join("  ·  ")}
      </p>
    ) : null;
  }

  return null;
}

/* ============================================================
   INTRO
   ============================================================ */

function DiscoveryIntro() {
  return (
    <div className="grid gap-4 lg:grid-cols-[1.4fr_0.8fr]">
      <div className="relative overflow-hidden rounded-2xl border border-[var(--line)] bg-[var(--paper-dim)] px-7 py-10 sm:px-10">
        <div className="absolute right-[-44px] top-[-44px] h-32 w-32 rounded-full border border-[var(--line)]" />
        <div className="absolute right-[-8px] top-[-8px] h-16 w-16 rounded-full border border-[var(--line-soft)]" />

        <div className="relative">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[var(--ink)]">
            <WandSparkles
              size={16}
              className="text-[var(--paper)]"
            />
          </div>

          <h2 className="mt-6 max-w-xl font-[var(--font-display)] text-2xl font-semibold tracking-[-0.025em]">
            Start with a question, not a source.
          </h2>

          <p className="mt-3 max-w-xl text-sm leading-6 text-[var(--ink-soft)]">
            Smart Research searches the providers
            you select, ranks the results, removes
            duplicates, and gives you a clean
            starting point for investigation.
          </p>

          <div className="mt-7 flex flex-wrap gap-2">
            {[
              "retrieval augmented generation",
              "vision transformers",
              "multimodal learning",
            ].map((item) => (
              <span
                key={item}
                className="rounded-full border border-[var(--line)] bg-[var(--paper)] px-3 py-1.5 text-[10px] text-[var(--ink-soft)]"
              >
                {item}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-[var(--line)] bg-[var(--paper)] p-6">
        <p className="font-[var(--font-mono)] text-[9px] uppercase tracking-[0.14em] text-[var(--muted)]">
          Search pipeline
        </p>

        <div className="mt-5 space-y-4">
          <PipelineStep
            index="01"
            title="Collect"
            description="Search the providers you select."
          />

          <PipelineStep
            index="02"
            title="Rank"
            description="Use lexical, semantic, or hybrid relevance."
          />

          <PipelineStep
            index="03"
            title="Curate"
            description="Save useful evidence to this workspace."
            last
          />
        </div>
      </div>
    </div>
  );
}

/* ============================================================
   PIPELINE STEP
   ============================================================ */

function PipelineStep({
  index,
  title,
  description,
  last = false,
}: {
  index: string;
  title: string;
  description: string;
  last?: boolean;
}) {
  return (
    <div className="flex gap-3">
      <div className="flex flex-col items-center">
        <span className="flex h-7 w-7 items-center justify-center rounded-full border border-[var(--line)] bg-[var(--paper-dim)] font-[var(--font-mono)] text-[9px] text-[var(--muted)]">
          {index}
        </span>

        {!last && (
          <span className="mt-1 h-7 w-px bg-[var(--line-soft)]" />
        )}
      </div>

      <div className="pt-0.5">
        <p className="text-sm font-semibold">
          {title}
        </p>

        <p className="mt-1 text-xs leading-5 text-[var(--muted)]">
          {description}
        </p>
      </div>
    </div>
  );
}

/* ============================================================
   FILTER CHIP
   ============================================================ */

function FilterChip({
  label,
  count,
  active,
  onClick,
}: {
  label: string;
  count: number;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "rounded-md border px-2.5 py-1.5 text-[10px] font-medium transition-all duration-200",
        active
          ? "border-[var(--ink)] bg-[var(--ink)] text-[var(--paper)]"
          : "border-[var(--line)] text-[var(--ink-soft)] hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]",
      ].join(" ")}
    >
      {label} {count}
    </button>
  );
}

/* ============================================================
   DROPDOWN
   ============================================================ */

function Dropdown({
  open,
  onToggle,
  label,
  children,
}: {
  open: boolean;
  onToggle: (
    event: React.MouseEvent<HTMLButtonElement>,
  ) => void;
  label: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="relative">
      <button
        type="button"
        onClick={onToggle}
        className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--line)] px-3 py-2 text-[11px] font-medium text-[var(--ink-soft)] transition-colors hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
      >
        {label}
      </button>

      {open && (
        <div
          className="absolute left-0 top-11 z-30 w-64 rounded-xl border border-[var(--line)] bg-[var(--paper)] p-2 shadow-xl"
          onClick={(event) =>
            event.stopPropagation()
          }
        >
          {children}
        </div>
      )}
    </div>
  );
}

/* ============================================================
   PROVIDER ICON
   ============================================================ */

function ResearchIcon({
  source,
}: {
  source: string;
}) {
  const base =
    "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-[var(--line)]";

  if (source === "github") {
    return (
      <div
        className={`${base} bg-[var(--paper-dim)] text-[var(--ink)]`}
      >
        <GitBranch size={18} />
      </div>
    );
  }

  if (
    source === "huggingface"
  ) {
    return (
      <div
        className={`${base} bg-[var(--accent-dim)] text-[var(--accent)]`}
      >
        <Heart size={17} />
      </div>
    );
  }

  if (source === "arxiv") {
    return (
      <div
        className={`${base} bg-[var(--cyan-dim)] text-[var(--cyan)]`}
      >
        <BookOpen size={18} />
      </div>
    );
  }

  return (
    <div
      className={`${base} bg-[var(--paper-dim)] text-[var(--cyan)]`}
    >
      <Sparkles size={17} />
    </div>
  );
}

/* ============================================================
   LOADING
   ============================================================ */

function LoadingResults() {
  return (
    <div className="space-y-3">
      {Array.from({
        length: 4,
      }).map((_, index) => (
        <div
          key={index}
          className="animate-pulse rounded-2xl border border-[var(--line)] bg-[var(--paper)] p-6"
        >
          <div className="flex gap-4">
            <div className="h-10 w-10 rounded-xl bg-[var(--paper-dim)]" />

            <div className="flex-1">
              <div className="h-2.5 w-20 rounded bg-[var(--paper-dim)]" />
              <div className="mt-3 h-4 w-3/5 rounded bg-[var(--paper-dim)]" />
              <div className="mt-3 h-2.5 w-full rounded bg-[var(--paper-dim)]" />
              <div className="mt-2 h-2.5 w-4/5 rounded bg-[var(--paper-dim)]" />
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ============================================================
   HELPERS
   ============================================================ */

function makeSourceKey(
  result: ResearchItem,
) {
  return `${result.source.toLowerCase()}|${result.url.trim()}`;
}

function prettySource(
  source: string,
) {
  switch (source) {
    case "arxiv":
      return "arXiv";

    case "github":
      return "GitHub";

    case "paperswithcode":
      return "PapersWithCode";

    case "huggingface":
      return "Hugging Face";

    default:
      return source;
  }
}

function formatDate(
  value: string,
) {
  const date = new Date(value);

  if (
    Number.isNaN(
      date.getTime(),
    )
  ) {
    return value;
  }

  return date.toLocaleDateString(
    undefined,
    {
      year: "numeric",
      month: "short",
      day: "numeric",
    },
  );
}

function formatNumber(
  value: number,
) {
  return new Intl.NumberFormat(
    undefined,
    {
      notation:
        value >= 1000
          ? "compact"
          : "standard",
      maximumFractionDigits: 1,
    },
  ).format(value);
}