import {
  ArrowRight,
  BookOpen,
  Brain,
  Code2,
  FileText,
  MessageSquare,
  Search,
} from "lucide-react";

import type {
  ActivityType,
  WorkspaceActivity,
} from "../lib/api";

interface RecentActivityProps {
  activities: WorkspaceActivity[];
  loading?: boolean;
  onNavigate?: (
    destination: "sources" | "discover" | "chat",
  ) => void;
}

const ACTIVITY_META: Record<
  ActivityType,
  { label: string; icon: typeof FileText }
> = {
  document_added: {
    label: "Document added",
    icon: FileText,
  },
  paper_added: {
    label: "Paper added",
    icon: BookOpen,
  },
  model_added: {
    label: "Model added",
    icon: Brain,
  },
  repository_added: {
    label: "Repository added",
    icon: Code2,
  },
  chat_started: {
    label: "Started a new chat",
    icon: MessageSquare,
  },
  research_performed: {
    label: "Research performed",
    icon: Search,
  },
};

function destinationForActivity(
  activityType: ActivityType,
): "sources" | "discover" | "chat" {
  switch (activityType) {
    case "chat_started":
      return "chat";
    case "research_performed":
      return "discover";
    default:
      return "sources";
  }
}

function formatRelativeTime(value: string): string {
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) {
    return "Recently";
  }

  const diffSeconds = Math.max(
    0,
    Math.floor((Date.now() - timestamp) / 1000),
  );

  if (diffSeconds < 10) return "Just now";
  if (diffSeconds < 60) return `${diffSeconds}s ago`;

  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) return `${diffMinutes}m ago`;

  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 7) return `${diffDays}d ago`;

  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
  }).format(timestamp);
}

export function RecentActivity({
  activities,
  loading = false,
  onNavigate,
}: RecentActivityProps) {
  return (
    <section className="mt-12">
      <div className="flex items-end justify-between gap-4">
        <div>
          <p className="flex items-center gap-2 font-[var(--font-mono)] text-[10px] font-semibold uppercase tracking-[0.15em] text-[var(--muted)]">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
            Recent activity
          </p>
          <h2 className="mt-3 font-[var(--font-display)] text-2xl font-semibold tracking-[-0.02em]">
            What you've been working on.
          </h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-[var(--ink-soft)]">
            A quick view of the latest research actions in this workspace.
          </p>
        </div>
      </div>

      <div className="mt-6 overflow-hidden rounded-2xl border border-[var(--line)] bg-[var(--paper)]">
        {loading ? (
          <div className="px-6 py-14 text-center">
            <p className="font-[var(--font-mono)] text-[10px] uppercase tracking-[0.12em] text-[var(--muted)]">
              Loading recent activity
            </p>
          </div>
        ) : activities.length === 0 ? (
          <div className="px-6 py-14 text-center">
            <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl border border-[var(--line)] bg-[var(--paper-dim)] text-[var(--ink-soft)]">
              <Search size={18} />
            </div>
            <h3 className="mt-4 font-[var(--font-display)] text-base font-semibold">
              No activity yet
            </h3>
            <p className="mx-auto mt-2 max-w-sm text-sm leading-6 text-[var(--muted)]">
              Add a source, start a chat, or run a research search to build your workspace history.
            </p>
          </div>
        ) : (
          <div>
            {activities.map((activity) => {
              const meta = ACTIVITY_META[activity.activity_type];
              const Icon = meta.icon;
              const destination = destinationForActivity(
                activity.activity_type,
              );

              const clickable = Boolean(onNavigate);

              return (
                <button
                  key={activity.id}
                  type="button"
                  disabled={!clickable}
                  onClick={() => {
                    if (onNavigate) {
                      onNavigate(destination);
                    }
                  }}
                  className="group flex w-full items-center gap-4 border-b border-[var(--line-soft)] px-5 py-4 text-left transition-colors last:border-b-0 hover:bg-[var(--paper-dim)] disabled:cursor-default"
                >
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-[var(--line)] bg-[var(--paper-dim)] text-[var(--ink-soft)]">
                    <Icon size={16} />
                  </div>

                  <div className="min-w-0 flex-1">
                    <div className="flex items-center justify-between gap-3">
                      <p className="truncate text-sm font-semibold text-[var(--ink)]">
                        {meta.label}
                      </p>
                      <span className="shrink-0 font-[var(--font-mono)] text-[9px] uppercase tracking-[0.08em] text-[var(--muted)]">
                        {formatRelativeTime(activity.created_at)}
                      </span>
                    </div>

                    <p className="mt-1 truncate text-sm text-[var(--ink-soft)]">
                      {activity.description || activity.title}
                    </p>
                  </div>

                  {clickable && (
                    <ArrowRight
                      size={14}
                      className="shrink-0 text-[var(--muted)] transition-transform duration-200 group-hover:translate-x-0.5 group-hover:text-[var(--ink)]"
                    />
                  )}
                </button>
              );
            })}
          </div>
        )}
      </div>
    </section>
  );
}
