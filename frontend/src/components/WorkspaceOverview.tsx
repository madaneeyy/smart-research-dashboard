import {
  ArrowRight,
  BookOpen,
  FileText,
  MessageSquare,
  Plus,
  Search,
  Sparkles,
} from "lucide-react";

import type { Workspace } from "../lib/api";

interface WorkspaceOverviewProps {
  workspace: Workspace;
  onNavigate: (
    destination:
      | "sources"
      | "discover"
      | "chat",
  ) => void;
  onCreateWorkspace: () => void;
}

export function WorkspaceOverview({
  workspace,
  onNavigate,
  onCreateWorkspace,
}: WorkspaceOverviewProps) {
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
        {/* Stats */}

        <section>
          <div className="grid gap-3 sm:grid-cols-3">
            <StatCard
              icon={<Sparkles size={16} />}
              label="Sources"
              value="0"
              description="Nothing collected yet"
            />

            <StatCard
              icon={<MessageSquare size={16} />}
              label="Chats"
              value="0"
              description="No conversations yet"
            />

            <StatCard
              icon={<FileText size={16} />}
              label="Documents"
              value="0"
              description="No documents added"
            />
          </div>
        </section>

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

            <button
              type="button"
              onClick={onCreateWorkspace}
              className="inline-flex items-center gap-1.5 self-start text-xs font-medium text-[var(--ink-soft)] transition-colors duration-200 hover:text-[var(--ink)]"
            >
              <Plus size={13} />
              New workspace
            </button>
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