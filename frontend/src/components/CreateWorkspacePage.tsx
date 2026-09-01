import { useState } from "react";
import type { ReactNode } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Check,
  FileText,
  Search,
  Sparkles,
} from "lucide-react";

type StartMode = "blank" | "topic" | "file";

interface CreateWorkspacePageProps {
  onBack: () => void;
  onCreate: (workspace: {
    name: string;
    description: string;
    startMode: StartMode;
  }) => void;
  isCreating: boolean;
  error: string | null;
}

export function CreateWorkspacePage({
  onBack,
  onCreate,
  isCreating,
  error,
}: CreateWorkspacePageProps) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [startMode, setStartMode] =
    useState<StartMode>("blank");

  const canCreate = name.trim().length >= 2;

  const handleCreate = () => {
    if (!canCreate) {
      return;
    }

    onCreate({
      name: name.trim(),
      description: description.trim(),
      startMode,
    });
  };

  return (
    <div className="min-h-screen animate-[fadeIn_350ms_ease-out_both] bg-[var(--paper)] text-[var(--ink)]">
      {/* Header */}

      <header className="border-b border-[var(--line)] bg-[var(--paper)]">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6 lg:px-8">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-md bg-[var(--ink)]">
              <Sparkles
                size={15}
                className="text-[var(--paper)]"
              />
            </div>

            <span className="font-[var(--font-display)] text-sm font-semibold">
              Smart Research AI
            </span>
          </div>

          <button
            type="button"
            onClick={onBack}
            className="group inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium text-[var(--ink-soft)] transition-all duration-200 hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
          >
            <ArrowLeft
              size={15}
              className="transition-transform duration-200 group-hover:-translate-x-0.5"
            />
            Back
          </button>
        </div>
      </header>

      {/* Main */}

      <main className="relative min-h-[calc(100vh-64px)] overflow-hidden">
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 opacity-70"
          style={{
            backgroundImage:
              "linear-gradient(var(--line-soft) 1px, transparent 1px), linear-gradient(90deg, var(--line-soft) 1px, transparent 1px)",
            backgroundSize: "36px 36px",
            WebkitMaskImage:
              "radial-gradient(ellipse 65% 55% at 50% 20%, black 20%, transparent 78%)",
            maskImage:
              "radial-gradient(ellipse 65% 55% at 50% 20%, black 20%, transparent 78%)",
          }}
        />

        <div className="relative mx-auto max-w-5xl px-6 py-16 lg:px-8 lg:py-24">
          {/* Intro */}

          <div className="mx-auto max-w-2xl text-center">
            <div className="inline-flex items-center gap-2 rounded-full border border-[var(--line)] bg-[var(--paper)] px-3 py-1.5 font-[var(--font-mono)] text-[10px] uppercase tracking-[0.12em] text-[var(--muted)]">
              <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
              Your first workspace
            </div>

            <h1 className="mt-6 font-[var(--font-display)] text-4xl font-semibold tracking-[-0.035em] sm:text-5xl">
              Give your research
              <br />
              <span className="text-[var(--accent)]">
                a home.
              </span>
            </h1>

            <p className="mx-auto mt-5 max-w-xl text-sm leading-6 text-[var(--ink-soft)] sm:text-base">
              Create a workspace for a topic, project, or
              question. You can add sources and continue building
              your research later.
            </p>
          </div>

          {/* Steps */}

          <div className="mx-auto mt-10 flex max-w-xl items-center justify-center gap-3">
            <StepIndicator
              number="01"
              label="Setup"
              active
            />

            <div className="h-px w-10 bg-[var(--line)]" />

            <StepIndicator
              number="02"
              label="Sources"
            />

            <div className="h-px w-10 bg-[var(--line)]" />

            <StepIndicator
              number="03"
              label="Research"
            />
          </div>

          {/* Form */}

          <div className="mx-auto mt-6 max-w-2xl">
            <div className="relative rounded-2xl border border-[var(--line)] bg-[var(--paper)] shadow-[var(--shadow-lg)]">
              <CornerBrackets
                color="var(--accent)"
                size={14}
                inset={0}
              />

              <div className="p-6 sm:p-8 lg:p-10">
                {error && (
                  <div className="mb-6 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                    <p className="font-medium">We couldn't create your workspace.</p>
                    <p className="mt-1 text-xs leading-5 text-red-600">{error}</p>
                  </div>
                )}

                <div>
                  <p className="font-[var(--font-mono)] text-[10px] uppercase tracking-[0.14em] text-[var(--muted)]">
                    Step 01
                  </p>

                  <h2 className="mt-2 font-[var(--font-display)] text-xl font-semibold">
                    Start with the basics
                  </h2>

                  <p className="mt-1.5 text-sm leading-6 text-[var(--ink-soft)]">
                    Give your workspace a name so you can find
                    it easily later.
                  </p>
                </div>

                {/* Name */}

                <div className="mt-8">
                  <label
                    htmlFor="workspace-name"
                    className="text-xs font-semibold"
                  >
                    Workspace name
                  </label>

                  <div className="relative mt-2">
                    <input
                      id="workspace-name"
                      value={name}
                      onChange={(event) =>
                        setName(event.target.value)
                      }
                      placeholder="e.g. Vision Mamba Research"
                      maxLength={80}
                      autoFocus
                      className="w-full rounded-lg border border-[var(--line)] bg-[var(--paper)] px-4 py-3 text-sm outline-none transition-all duration-200 placeholder:text-[var(--muted)] focus:border-[var(--ink-soft)] focus:ring-2 focus:ring-[var(--line-soft)]"
                    />

                    <span className="pointer-events-none absolute bottom-3 right-3 font-[var(--font-mono)] text-[9px] text-[var(--muted)]">
                      {name.length}/80
                    </span>
                  </div>

                  {name.length > 0 &&
                    name.trim().length < 2 && (
                      <p className="mt-2 text-xs text-[var(--accent)]">
                        Use at least 2 characters.
                      </p>
                    )}
                </div>

                {/* Description */}

                <div className="mt-6">
                  <div className="flex items-center justify-between">
                    <label
                      htmlFor="workspace-description"
                      className="text-xs font-semibold"
                    >
                      Description
                    </label>

                    <span className="font-[var(--font-mono)] text-[9px] text-[var(--muted)]">
                      Optional
                    </span>
                  </div>

                  <textarea
                    id="workspace-description"
                    value={description}
                    onChange={(event) =>
                      setDescription(event.target.value)
                    }
                    placeholder="What are you investigating?"
                    rows={4}
                    maxLength={300}
                    className="mt-2 w-full resize-none rounded-lg border border-[var(--line)] bg-[var(--paper)] px-4 py-3 text-sm leading-6 outline-none transition-all duration-200 placeholder:text-[var(--muted)] focus:border-[var(--ink-soft)] focus:ring-2 focus:ring-[var(--line-soft)]"
                  />

                  <div className="mt-1.5 text-right font-[var(--font-mono)] text-[9px] text-[var(--muted)]">
                    {description.length}/300
                  </div>
                </div>

                {/* Starting point */}

                <div className="my-8 h-px bg-[var(--line-soft)]" />

                <div>
                  <p className="font-[var(--font-mono)] text-[10px] uppercase tracking-[0.14em] text-[var(--muted)]">
                    Starting point
                  </p>

                  <div className="mt-2 flex flex-col justify-between gap-1 sm:flex-row sm:items-end">
                    <h3 className="font-[var(--font-display)] text-base font-semibold">
                      How would you like to begin?
                    </h3>

                    <span className="text-xs text-[var(--muted)]">
                      You can add more later.
                    </span>
                  </div>

                  <div className="mt-5 grid gap-3 md:grid-cols-3">
                    <StartOption
                      icon={<FileText size={18} />}
                      title="Blank"
                      description="Start from scratch."
                      active={startMode === "blank"}
                      onClick={() =>
                        setStartMode("blank")
                      }
                    />

                    <StartOption
                      icon={<Search size={18} />}
                      title="Research topic"
                      description="Begin with a question."
                      active={startMode === "topic"}
                      onClick={() =>
                        setStartMode("topic")
                      }
                    />

                    <StartOption
                      icon={<BookOpen size={18} />}
                      title="Upload files"
                      description="Start with your documents."
                      active={startMode === "file"}
                      onClick={() =>
                        setStartMode("file")
                      }
                    />
                  </div>
                </div>

                {/* Footer */}

                <div className="mt-8 flex flex-col gap-5 border-t border-[var(--line-soft)] pt-6 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-center gap-2 text-xs text-[var(--muted)]">
                    <span className="flex h-5 w-5 items-center justify-center rounded-full border border-[var(--line)]">
                      <Check size={10} />
                    </span>

                    No account required for the demo
                  </div>

                  <button
                    type="button"
                    onClick={handleCreate}
                    disabled={!canCreate || isCreating}
                    className="group inline-flex items-center justify-center gap-2 rounded-md bg-[var(--ink)] px-5 py-3 text-sm font-medium text-[var(--paper)] shadow-sm transition-all duration-200 hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-md disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:translate-y-0 disabled:hover:bg-[var(--ink)] disabled:hover:shadow-sm"
                  >
                    {isCreating ? "Creating workspace..." : "Create workspace"}

                    <ArrowRight
                      size={16}
                      className="transition-transform duration-200 group-hover:translate-x-0.5"
                    />
                  </button>
                </div>
              </div>
            </div>

            <p className="mt-5 text-center font-[var(--font-mono)] text-[10px] uppercase tracking-[0.08em] text-[var(--muted)]">
              Your workspace is private by default
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}

/* ============================================================
   STEP INDICATOR
   ============================================================ */

function StepIndicator({
  number,
  label,
  active = false,
}: {
  number: string;
  label: string;
  active?: boolean;
}) {
  return (
    <div
      className={[
        "flex items-center gap-2 transition-colors duration-200",
        active
          ? "text-[var(--ink)]"
          : "text-[var(--muted)]",
      ].join(" ")}
    >
      <span
        className={[
          "flex h-7 w-7 items-center justify-center rounded-md border font-[var(--font-mono)] text-[9px] font-semibold",
          active
            ? "border-[var(--ink)] bg-[var(--ink)] text-[var(--paper)]"
            : "border-[var(--line)] bg-[var(--paper)]",
        ].join(" ")}
      >
        {number}
      </span>

      <span className="hidden text-[10px] font-medium sm:block">
        {label}
      </span>
    </div>
  );
}

/* ============================================================
   START OPTION
   ============================================================ */

function StartOption({
  icon,
  title,
  description,
  active,
  onClick,
}: {
  icon: ReactNode;
  title: string;
  description: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        "group rounded-xl border p-4 text-left",
        "transition-all duration-200",
        active
          ? "border-[var(--accent)] bg-[var(--accent-dim)] shadow-sm"
          : "border-[var(--line)] bg-[var(--paper)] hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:shadow-sm",
      ].join(" ")}
    >
      <div className="flex items-start justify-between gap-3">
        <div
          className={[
            "flex h-9 w-9 items-center justify-center rounded-lg",
            "transition-colors duration-200",
            active
              ? "bg-[var(--accent)] text-white"
              : "bg-[var(--paper-dim)] text-[var(--ink-soft)]",
          ].join(" ")}
        >
          {icon}
        </div>

        <span
          className={[
            "flex h-4 w-4 items-center justify-center rounded-full border",
            active
              ? "border-[var(--accent)] bg-[var(--accent)]"
              : "border-[var(--line)]",
          ].join(" ")}
        >
          {active && (
            <span className="h-1.5 w-1.5 rounded-full bg-white" />
          )}
        </span>
      </div>

      <p className="mt-4 font-[var(--font-display)] text-sm font-semibold">
        {title}
      </p>

      <p className="mt-1 text-xs leading-5 text-[var(--ink-soft)]">
        {description}
      </p>
    </button>
  );
}

/* ============================================================
   CORNER BRACKETS
   ============================================================ */

function CornerBrackets({
  color,
  size = 16,
  inset = 0,
}: {
  color: string;
  size?: number;
  inset?: number;
}) {
  const base =
    "pointer-events-none absolute border-current";

  return (
    <div
      style={{ color }}
      className="pointer-events-none absolute inset-0 z-10"
    >
      <span
        className={base}
        style={{
          top: inset,
          left: inset,
          width: size,
          height: size,
          borderTop: "2px solid",
          borderLeft: "2px solid",
        }}
      />

      <span
        className={base}
        style={{
          top: inset,
          right: inset,
          width: size,
          height: size,
          borderTop: "2px solid",
          borderRight: "2px solid",
        }}
      />

      <span
        className={base}
        style={{
          bottom: inset,
          left: inset,
          width: size,
          height: size,
          borderBottom: "2px solid",
          borderLeft: "2px solid",
        }}
      />

      <span
        className={base}
        style={{
          bottom: inset,
          right: inset,
          width: size,
          height: size,
          borderBottom: "2px solid",
          borderRight: "2px solid",
        }}
      />
    
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
      `}</style>
</div>
  );
}