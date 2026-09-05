import { useEffect, useState } from "react";
import {
  ArrowRight,
  BookOpen,
  Brain,
  FileText,
  Code2,
  MessageSquare,
  Plus,
  Search,
} from "lucide-react";

import {
  getWorkspaceActivity,
  getWorkspaceDocuments,
  getWorkspaceSources,
  type Workspace,
  type WorkspaceActivity,
  type WorkspaceDocument,
  type WorkspaceSource,
} from "../lib/api";
import { RecentActivity } from "./RecentActivity";

interface WorkspaceOverviewProps {
  workspace: Workspace;
  onNavigate: (
    destination:
      | "sources"
      | "discover"
      | "chat",
  ) => void;
}

export function WorkspaceOverview({
  workspace,
  onNavigate,
}: WorkspaceOverviewProps) {
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
  const [sources, setSources] = useState<WorkspaceSource[]>([]);
  const [activities, setActivities] = useState<WorkspaceActivity[]>([]);
  const [loadingStats, setLoadingStats] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function loadStats() {
      setLoadingStats(true);

      try {
        const [
          loadedDocuments,
          loadedSources,
          loadedActivity,
        ] = await Promise.all([
          getWorkspaceDocuments(workspace.id),
          getWorkspaceSources(workspace.id),
          getWorkspaceActivity(workspace.id, 8),
        ]);

        if (cancelled) {
          return;
        }

        setDocuments(
          Array.isArray(loadedDocuments) ? loadedDocuments : [],
        );
        setSources(
          Array.isArray(loadedSources) ? loadedSources : [],
        );
        setActivities(
          Array.isArray(loadedActivity) ? loadedActivity : [],
        );
      } catch {
        if (!cancelled) {
          setDocuments([]);
          setSources([]);
          setActivities([]);
        }
      } finally {
        if (!cancelled) {
          setLoadingStats(false);
        }
      }
    }

    void loadStats();

    return () => {
      cancelled = true;
    };
  }, [workspace.id]);

  const normalizedSources = sources.map((source) => ({
    ...source,
    sourceType: source.source_type.trim().toLowerCase(),
  }));

  const paperSources = normalizedSources.filter(
    (source) =>
      source.sourceType === "arxiv" ||
      source.sourceType === "paperswithcode",
  );

  const githubCount = normalizedSources.filter(
    (source) => source.sourceType === "github",
  ).length;

  const paperCount = paperSources.length;

  const huggingfaceCount = normalizedSources.filter(
    (source) => source.sourceType === "huggingface",
  ).length;

  const arxivDocumentIds = new Set(
    normalizedSources
      .filter((source) => source.sourceType === "arxiv")
      .map((source) => {
        const value = source.metadata?.["document_id"];
        return typeof value === "string" ? value.trim() : "";
      })
      .filter(Boolean),
  );

  const documentCount = documents.filter(
    (document) => !arxivDocumentIds.has(document.document_id),
  ).length;

  const statValue = (value: number) =>
    loadingStats ? "—" : String(value);

  return (
    <div className="min-h-full bg-[var(--paper)] text-[var(--ink)]">
      {/* ======================================================
          PAGE HEADER
      ====================================================== */}

      <div className="border-b border-[var(--line)]">
        <div className="mx-auto max-w-7xl px-6 py-8 lg:px-8">
          <p className="font-[var(--font-mono)] text-[10px] uppercase tracking-[0.14em] text-[var(--muted)]">
            Workspace
          </p>

          <div className="mt-2 flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h1 className="font-[var(--font-display)] text-3xl font-semibold tracking-[-0.025em] sm:text-4xl">
                {workspace.name}
              </h1>

              <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--ink-soft)]">
                {workspace.description ||
                  "A focused space for your research, sources, and questions."}
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() =>
                  onNavigate("sources")
                }
                className="inline-flex items-center gap-2 rounded-md border border-[var(--line)] bg-[var(--paper)] px-3.5 py-2.5 text-sm font-medium text-[var(--ink)] shadow-sm transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:shadow-md"
              >
                <Plus size={15} />
                Add source
              </button>

              <button
                type="button"
                onClick={() =>
                  onNavigate("chat")
                }
                className="group inline-flex items-center gap-2 rounded-md bg-[var(--ink)] px-3.5 py-2.5 text-sm font-medium text-[var(--paper)] shadow-sm transition-all duration-200 hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-md"
              >
                Ask AI

                <ArrowRight
                  size={15}
                  className="transition-transform duration-200 group-hover:translate-x-0.5"
                />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ======================================================
          CONTENT
      ====================================================== */}

      <main className="mx-auto max-w-7xl px-6 py-8 lg:px-8 lg:py-10">
        {/* ====================================================
            GET STARTED
        ==================================================== */}

        <section className="mt-12">
          <SectionLabel>
            Get started
          </SectionLabel>

          <div className="mt-3">
            <h2 className="font-[var(--font-display)] text-2xl font-semibold tracking-[-0.02em]">
              Build your research workspace.
            </h2>

            <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--ink-soft)]">
              Start with a source or explore what's already
              available. You can change your workflow at any
              time.
            </p>
          </div>

          <div className="mt-6 grid gap-3 md:grid-cols-3">
            <ActionCard
              icon={<Search size={18} />}
              step="01"
              title="Discover research"
              description="Find papers, repositories, and other useful research."
              action="Explore"
              onClick={() =>
                onNavigate("discover")
              }
            />

            <ActionCard
              icon={<FileText size={18} />}
              step="02"
              title="Add a source"
              description="Bring a document or repository into this workspace."
              action="Add source"
              onClick={() =>
                onNavigate("sources")
              }
            />

            <ActionCard
              icon={<MessageSquare size={18} />}
              step="03"
              title="Ask your research"
              description="Start asking questions once you've collected material."
              action="Open chat"
              onClick={() =>
                onNavigate("chat")
              }
            />
          </div>
        </section>


        {/* Workspace at a glance */}

        <section className="mt-12">
          <SectionLabel>
            Workspace at a glance
          </SectionLabel>

          <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              icon={<FileText size={16} />}
              label="Documents"
              value={statValue(documentCount)}
              description={
                documentCount === 1
                  ? "Uploaded document"
                  : "Uploaded documents"
              }
            />

            <StatCard
              icon={<Code2 size={16} />}
              label="GitHub Repos"
              value={statValue(githubCount)}
              description={
                githubCount === 1
                  ? "Repository connected"
                  : "Repositories connected"
              }
            />

            <StatCard
              icon={<BookOpen size={16} />}
              label="Papers"
              value={statValue(paperCount)}
              description={
                paperCount === 1
                  ? "Research paper added"
                  : "Research papers added"
              }
            />

            <StatCard
              icon={<Brain size={16} />}
              label="Hugging Face Models"
              value={statValue(huggingfaceCount)}
              description={
                huggingfaceCount === 1
                  ? "Model added"
                  : "Models added"
              }
            />
          </div>
        </section>


        <RecentActivity
          activities={activities}
          loading={loadingStats}
          onNavigate={onNavigate}
        />

        {/* ====================================================
            EMPTY RESEARCH STATE
        ==================================================== */}

        <section className="mt-12">
          <SectionLabel>
            Your research
          </SectionLabel>

          <div className="mt-3 rounded-2xl border border-[var(--line)] bg-[var(--paper-dim)]">
            <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-[var(--line)] bg-[var(--paper)] text-[var(--ink-soft)] shadow-sm">
                <BookOpen size={19} />
              </div>

              <h3 className="mt-5 font-[var(--font-display)] text-lg font-semibold">
                Your research will appear here
              </h3>

              <p className="mt-2 max-w-md text-sm leading-6 text-[var(--ink-soft)]">
                Add your first source and this space will become
                the center of your research activity.
              </p>

              <button
                type="button"
                onClick={() =>
                  onNavigate("sources")
                }
                className="group mt-6 inline-flex items-center gap-2 rounded-md border border-[var(--line)] bg-[var(--paper)] px-4 py-2.5 text-sm font-medium text-[var(--ink)] shadow-sm transition-all duration-200 hover:-translate-y-px hover:shadow-md"
              >
                Add your first source

                <ArrowRight
                  size={15}
                  className="transition-transform duration-200 group-hover:translate-x-0.5"
                />
              </button>
            </div>
          </div>
        </section>


        {/* ====================================================
            WORKSPACE META
        ==================================================== */}

        <section className="mt-10 border-t border-[var(--line-soft)] pt-6">
          <div className="flex flex-col gap-2 text-[var(--muted)] sm:flex-row sm:items-center sm:justify-between">
            <p className="font-[var(--font-mono)] text-[10px] uppercase tracking-[0.08em]">
              Workspace ID · {workspace.id}
            </p>


          </div>
        </section>
      
      </main>
    </div>
  );
}

/* ============================================================
   SECTION LABEL
   ============================================================ */

function SectionLabel({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <p className="flex items-center gap-2 font-[var(--font-mono)] text-[10px] font-semibold uppercase tracking-[0.15em] text-[var(--muted)]">
      <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
      {children}
    </p>
  );
}

/* ============================================================
   STAT CARD
   ============================================================ */

function StatCard({
  icon,
  label,
  value,
  description,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  description: string;
}) {
  return (
    <div className="rounded-xl border border-[var(--line)] bg-[var(--paper)] p-5 transition-all duration-200 hover:-translate-y-px hover:shadow-sm">
      <div className="flex items-start justify-between">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--paper-dim)] text-[var(--ink-soft)]">
          {icon}
        </div>

        <span className="font-[var(--font-mono)] text-[9px] uppercase tracking-[0.12em] text-[var(--muted)]">
          Live
        </span>
      </div>

      <p className="mt-6 font-[var(--font-display)] text-3xl font-semibold tracking-[-0.03em]">
        {value}
      </p>

      <p className="mt-1 text-sm font-medium text-[var(--ink)]">
        {label}
      </p>

      <p className="mt-1 text-xs text-[var(--muted)]">
        {description}
      </p>
    </div>
  );
}

/* ============================================================
   ACTION CARD
   ============================================================ */

function ActionCard({
  icon,
  step,
  title,
  description,
  action,
  onClick,
}: {
  icon: React.ReactNode;
  step: string;
  title: string;
  description: string;
  action: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="group rounded-xl border border-[var(--line)] bg-[var(--paper)] p-5 text-left transition-all duration-200 hover:-translate-y-px hover:border-[var(--ink-soft)] hover:shadow-md"
    >
      <div className="flex items-start justify-between">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--paper-dim)] text-[var(--ink-soft)] transition-colors duration-200 group-hover:bg-[var(--accent-dim)] group-hover:text-[var(--accent)]">
          {icon}
        </div>

        <span className="font-[var(--font-mono)] text-[9px] text-[var(--muted)]">
          {step}
        </span>
      </div>

      <h3 className="mt-5 font-[var(--font-display)] text-base font-semibold">
        {title}
      </h3>

      <p className="mt-2 min-h-[48px] text-xs leading-5 text-[var(--ink-soft)]">
        {description}
      </p>

      <div className="mt-5 flex items-center gap-1.5 text-xs font-medium text-[var(--ink-soft)] transition-colors duration-200 group-hover:text-[var(--accent)]">
        {action}

        <ArrowRight
          size={13}
          className="transition-transform duration-200 group-hover:translate-x-0.5"
        />
      </div>
    </button>
  );
}