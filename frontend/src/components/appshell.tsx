import { useState } from "react";
import type { ReactNode } from "react";
import {
  AlertTriangle,
  FileText,
  LayoutDashboard,
  MessageSquare,
  Moon,
  Plus,
  Search,
  Sparkles,
  Sun,
  Trash2,
  X,
} from "lucide-react";

import { useTheme } from "../context/ThemeContext";
import type { Workspace } from "../lib/api";

type AppSection =
  | "overview"
  | "sources"
  | "discover"
  | "chat";

interface AppShellProps {
  workspace: Workspace;
  activeSection: AppSection;
  onNavigate: (section: AppSection) => void;
  onCreateWorkspace: () => void;
  onDeleteWorkspace: () => void;
  isDeletingWorkspace: boolean;
  workspaceError: string | null;
  children: ReactNode;
}

export function AppShell({
  workspace,
  activeSection,
  onNavigate,
  onDeleteWorkspace,
  isDeletingWorkspace,
  workspaceError,
  children,
}: AppShellProps) {
  const {
    theme,
    toggleTheme,
  } = useTheme();

  const [
    accountGateOpen,
    setAccountGateOpen,
  ] = useState(false);

  const [
    deleteDialogOpen,
    setDeleteDialogOpen,
  ] = useState(false);

  return (
    <div className="flex min-h-screen bg-[var(--paper)] text-[var(--ink)]">
      <aside className="hidden w-60 shrink-0 flex-col border-r border-[var(--line)] bg-[var(--paper)] lg:flex">
        <div className="flex h-16 items-center border-b border-[var(--line-soft)] px-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-[var(--ink)]">
              <Sparkles
                size={14}
                className="text-[var(--paper)]"
              />
            </div>

            <span className="font-[var(--font-display)] text-sm font-semibold tracking-tight">
              Smart Research AI
            </span>
          </div>
        </div>

        <div className="px-3 py-4">
          <div className="flex items-center justify-between px-2">
            <p className="font-[var(--font-mono)] text-[9px] uppercase tracking-[0.14em] text-[var(--muted)]">
              Workspace
            </p>

            <button
              type="button"
              onClick={() =>
                setDeleteDialogOpen(
                  true,
                )
              }
              disabled={
                isDeletingWorkspace
              }
              className="flex h-7 w-7 items-center justify-center rounded-md text-[var(--muted)] transition-all duration-200 hover:bg-[var(--accent-dim)] hover:text-[var(--accent)] disabled:opacity-40"
              aria-label="Delete workspace"
              title="Delete workspace"
            >
              <Trash2 size={13} />
            </button>
          </div>

          <div className="mt-2 rounded-lg border border-[var(--line)] bg-[var(--paper-dim)] p-3">
            <p className="truncate font-[var(--font-display)] text-xs font-semibold">
              {workspace.name}
            </p>

            <p className="mt-1 truncate font-[var(--font-mono)] text-[9px] text-[var(--muted)]">
              Demo workspace
            </p>
          </div>
        </div>

        <nav className="px-3">
          <p className="px-2 font-[var(--font-mono)] text-[9px] uppercase tracking-[0.14em] text-[var(--muted)]">
            Research
          </p>

          <div className="mt-2 space-y-1">
            <NavItem
              icon={
                <LayoutDashboard size={15} />
              }
              label="Overview"
              active={
                activeSection ===
                "overview"
              }
              onClick={() =>
                onNavigate(
                  "overview",
                )
              }
            />

            <NavItem
              icon={
                <FileText size={15} />
              }
              label="Sources"
              active={
                activeSection ===
                "sources"
              }
              onClick={() =>
                onNavigate(
                  "sources",
                )
              }
            />

            <NavItem
              icon={<Search size={15} />}
              label="Discover"
              active={
                activeSection ===
                "discover"
              }
              onClick={() =>
                onNavigate(
                  "discover",
                )
              }
            />

            <NavItem
              icon={
                <MessageSquare
                  size={15}
                />
              }
              label="Chat"
              active={
                activeSection ===
                "chat"
              }
              onClick={() =>
                onNavigate("chat")
              }
            />
          </div>
        </nav>

        <div className="mt-8 px-3">
          <p className="px-2 font-[var(--font-mono)] text-[9px] uppercase tracking-[0.14em] text-[var(--muted)]">
            Recent chats
          </p>

          <div className="mt-3 rounded-lg border border-dashed border-[var(--line)] px-3 py-4 text-center">
            <MessageSquare
              size={15}
              className="mx-auto text-[var(--muted)]"
            />

            <p className="mt-2 text-[10px] text-[var(--muted)]">
              No conversations yet
            </p>
          </div>
        </div>

        <div className="mt-auto border-t border-[var(--line-soft)] p-3">
          <button
            type="button"
            onClick={() =>
              setAccountGateOpen(
                true,
              )
            }
            className="flex w-full items-center gap-2 rounded-md px-3 py-2.5 text-xs font-medium text-[var(--ink-soft)] transition-all duration-200 hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
          >
            <Plus size={14} />
            New workspace
          </button>

          <button
            type="button"
            onClick={toggleTheme}
            className="mt-1 flex w-full items-center gap-2 rounded-md px-3 py-2.5 text-xs font-medium text-[var(--ink-soft)] transition-all duration-200 hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
          >
            {theme === "light" ? (
              <Moon size={14} />
            ) : (
              <Sun size={14} />
            )}

            {theme === "light"
              ? "Dark mode"
              : "Light mode"}
          </button>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-40 flex h-16 shrink-0 items-center justify-between border-b border-[var(--line)] bg-[var(--paper)]/95 px-6 backdrop-blur-xl lg:px-8">
          <div className="min-w-0">
            <p className="truncate font-[var(--font-display)] text-sm font-semibold">
              {workspace.name}
            </p>

            <p className="hidden font-[var(--font-mono)] text-[9px] uppercase tracking-[0.12em] text-[var(--muted)] sm:block">
              Research workspace
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() =>
                setAccountGateOpen(
                  true,
                )
              }
              className="hidden items-center gap-1.5 rounded-md border border-[var(--line)] px-3 py-2 text-xs font-medium text-[var(--ink-soft)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] hover:shadow-sm sm:inline-flex"
            >
              <Plus size={13} />
              New workspace
            </button>

            <button
              type="button"
              onClick={toggleTheme}
              className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-[var(--line)] text-[var(--ink-soft)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] hover:shadow-sm"
              aria-label="Toggle theme"
            >
              {theme === "light" ? (
                <Moon size={15} />
              ) : (
                <Sun size={15} />
              )}
            </button>

            <div className="hidden h-8 w-px bg-[var(--line-soft)] sm:block" />

            <div className="flex h-8 w-8 items-center justify-center rounded-full border border-[var(--line)] bg-[var(--paper-dim)] text-xs font-semibold">
              D
            </div>
          </div>
        </header>

        {workspaceError && (
          <div className="border-b border-[var(--line)] bg-[var(--accent-dim)] px-6 py-3 lg:px-8">
            <div className="mx-auto flex max-w-7xl items-center gap-2 text-xs text-[var(--accent)]">
              <AlertTriangle size={14} />

              <span>{workspaceError}</span>
            </div>
          </div>
        )}

        <div className="min-w-0 flex-1">
          {children}
        </div>
      </div>

      {accountGateOpen && (
        <ModalOverlay
          onClose={() =>
            setAccountGateOpen(
              false,
            )
          }
        >
          <div className="relative w-full max-w-md rounded-2xl border border-[var(--line)] bg-[var(--paper)] shadow-2xl">
            <button
              type="button"
              onClick={() =>
                setAccountGateOpen(
                  false,
                )
              }
              className="absolute right-4 top-4 flex h-8 w-8 items-center justify-center rounded-md text-[var(--muted)] hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
              aria-label="Close"
            >
              <X size={16} />
            </button>

            <div className="p-7">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[var(--ink)]">
                <Sparkles
                  size={17}
                  className="text-[var(--paper)]"
                />
              </div>

              <p className="mt-6 font-[var(--font-mono)] text-[10px] uppercase tracking-[0.14em] text-[var(--muted)]">
                Demo workspace
              </p>

              <h2 className="mt-2 font-[var(--font-display)] text-2xl font-semibold tracking-[-0.025em]">
                Ready for more?
              </h2>

              <p className="mt-3 text-sm leading-6 text-[var(--ink-soft)]">
                The demo lets you explore one research
                workspace. Create an account to manage
                multiple workspaces and keep your research
                organized.
              </p>

              <div className="mt-7 flex justify-end gap-2">
                <button
                  type="button"
                  onClick={() =>
                    setAccountGateOpen(
                      false,
                    )
                  }
                  className="rounded-md border border-[var(--line)] px-4 py-2.5 text-sm font-medium text-[var(--ink-soft)] transition-all duration-200 hover:bg-[var(--paper-dim)]"
                >
                  Not now
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setAccountGateOpen(
                      false,
                    );
                    console.log(
                      "Sign up clicked",
                    );
                  }}
                  className="rounded-md bg-[var(--ink)] px-4 py-2.5 text-sm font-medium text-[var(--paper)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-md"
                >
                  Create account
                </button>
              </div>
            </div>
          </div>
        </ModalOverlay>
      )}

      {deleteDialogOpen && (
        <ModalOverlay
          onClose={() => {
            if (
              !isDeletingWorkspace
            ) {
              setDeleteDialogOpen(
                false,
              );
            }
          }}
        >
          <div className="relative w-full max-w-md rounded-2xl border border-[var(--line)] bg-[var(--paper)] shadow-2xl">
            <button
              type="button"
              disabled={
                isDeletingWorkspace
              }
              onClick={() =>
                setDeleteDialogOpen(
                  false,
                )
              }
              className="absolute right-4 top-4 flex h-8 w-8 items-center justify-center rounded-md text-[var(--muted)] hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] disabled:opacity-40"
              aria-label="Close"
            >
              <X size={16} />
            </button>

            <div className="p-7">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[var(--accent-dim)] text-[var(--accent)]">
                <Trash2 size={17} />
              </div>

              <p className="mt-6 font-[var(--font-mono)] text-[10px] uppercase tracking-[0.14em] text-[var(--muted)]">
                Permanent action
              </p>

              <h2 className="mt-2 font-[var(--font-display)] text-2xl font-semibold tracking-[-0.025em]">
                Delete workspace?
              </h2>

              <p className="mt-3 text-sm leading-6 text-[var(--ink-soft)]">
                This permanently removes{" "}
                <span className="font-semibold text-[var(--ink)]">
                  {workspace.name}
                </span>{" "}
                and its sources, documents, chats, and research data.
              </p>

              <div className="mt-7 flex justify-end gap-2">
                <button
                  type="button"
                  disabled={
                    isDeletingWorkspace
                  }
                  onClick={() =>
                    setDeleteDialogOpen(
                      false,
                    )
                  }
                  className="rounded-md border border-[var(--line)] px-4 py-2.5 text-sm font-medium text-[var(--ink-soft)] hover:bg-[var(--paper-dim)] disabled:opacity-40"
                >
                  Cancel
                </button>

                <button
                  type="button"
                  disabled={
                    isDeletingWorkspace
                  }
                  onClick={async () => {
                    await onDeleteWorkspace();
                    setDeleteDialogOpen(
                      false,
                    );
                  }}
                  className="inline-flex items-center gap-2 rounded-md bg-[var(--ink)] px-4 py-2.5 text-sm font-medium text-[var(--paper)] shadow-sm transition-all duration-200 hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-md disabled:opacity-50"
                >
                  {isDeletingWorkspace ? (
                    "Deleting..."
                  ) : (
                    <>
                      <Trash2 size={14} />
                      Delete workspace
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </ModalOverlay>
      )}
    </div>
  );
}

/* ============================================================
   NAV ITEM
   ============================================================ */

function NavItem({
  icon,
  label,
  active,
  onClick,
}: {
  icon: ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "flex w-full items-center gap-2.5 rounded-md px-3 py-2.5 text-left text-xs font-medium",
        "transition-all duration-150",
        active
          ? "bg-[var(--paper-dim)] text-[var(--ink)]"
          : "text-[var(--ink-soft)] hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]",
      ].join(" ")}
    >
      <span
        className={
          active
            ? "text-[var(--accent)]"
            : "text-[var(--muted)]"
        }
      >
        {icon}
      </span>

      {label}
    </button>
  );
}

/* ============================================================
   MODAL OVERLAY
   ============================================================ */

function ModalOverlay({
  children,
  onClose,
}: {
  children: ReactNode;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-[rgba(18,35,61,0.22)] p-5 backdrop-blur-sm"
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
        className="w-full animate-[modalIn_180ms_ease-out_both]"
        onMouseDown={(event) =>
          event.stopPropagation()
        }
      >
        {children}
      </div>

      <style>{`
        @keyframes modalIn {
          from {
            opacity: 0;
            transform: translateY(8px) scale(0.985);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }
      `}</style>
    </div>
  );
}