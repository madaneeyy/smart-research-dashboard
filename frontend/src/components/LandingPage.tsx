import type { ReactNode } from "react";
import {
  ArrowRight,
  BookOpen,
  Check,
  Code2,
  FileText,
  Search,
  Sparkles,
} from "lucide-react";

interface LandingPageProps {
  onTryDemo: () => void;
  onSignIn: () => void;
}

export function LandingPage({
  onTryDemo,
  onSignIn,
}: LandingPageProps) {
  return (
    <div className="min-h-screen bg-white text-zinc-950">
      {/* ======================================================
          NAVIGATION
      ====================================================== */}

      <header className="sticky top-0 z-50 border-b border-zinc-100/80 bg-white/90 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6 lg:px-8">
          {/* Brand */}

          <button
            onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
            className="flex items-center gap-2.5"
          >
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-zinc-950">
              <Sparkles
                size={15}
                strokeWidth={2}
                className="text-white"
              />
            </span>

            <span className="text-sm font-semibold tracking-[-0.01em]">
              Smart Research AI
            </span>
          </button>

          {/* Navigation */}

          <nav className="hidden items-center gap-7 md:flex">
            <a
              href="#product"
              className="text-sm text-zinc-500 transition hover:text-zinc-950"
            >
              Product
            </a>

            <a
              href="#workflow"
              className="text-sm text-zinc-500 transition hover:text-zinc-950"
            >
              How it works
            </a>

            <a
              href="#capabilities"
              className="text-sm text-zinc-500 transition hover:text-zinc-950"
            >
              Capabilities
            </a>
          </nav>

          {/* Actions */}

          <div className="flex items-center gap-2">
            <button
              onClick={onSignIn}
              className="hidden rounded-lg px-3.5 py-2 text-sm font-medium text-zinc-600 transition hover:bg-zinc-50 hover:text-zinc-950 sm:inline-flex"
            >
              Sign in
            </button>

            <button
              onClick={onTryDemo}
              className="inline-flex items-center gap-2 rounded-lg bg-zinc-950 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-zinc-800"
            >
              Try demo
              <ArrowRight size={15} />
            </button>
          </div>
        </div>
      </header>

      <main>
        {/* ======================================================
            HERO
        ====================================================== */}

        <section
          id="product"
          className="border-b border-zinc-100"
        >
          <div className="mx-auto max-w-7xl px-6 pb-24 pt-20 lg:px-8 lg:pb-32 lg:pt-28">
            <div className="mx-auto max-w-4xl text-center">
              {/* Eyebrow */}

              <div className="inline-flex items-center gap-2 rounded-full border border-zinc-200 bg-zinc-50 px-3.5 py-1.5 text-xs font-medium text-zinc-600">
                <Sparkles size={13} />
                A workspace for modern research
              </div>

              {/* Heading */}

              <h1 className="mt-7 text-5xl font-semibold leading-[0.98] tracking-[-0.055em] text-zinc-950 sm:text-6xl lg:text-[76px]">
                Research smarter.
                <br />
                Understand deeper.
              </h1>

              {/* Description */}

              <p className="mx-auto mt-7 max-w-2xl text-base leading-7 text-zinc-500 sm:text-lg">
                Discover research, collect the sources that matter,
                and ask intelligent questions across your documents
                and code.
              </p>

              {/* CTA */}

              <div className="mt-9 flex flex-col justify-center gap-3 sm:flex-row">
                <button
                  onClick={onTryDemo}
                  className="inline-flex items-center justify-center gap-2 rounded-lg bg-zinc-950 px-5 py-3 text-sm font-medium text-white shadow-sm transition hover:bg-zinc-800"
                >
                  Try the demo
                  <ArrowRight size={16} />
                </button>

                <button
                  onClick={onSignIn}
                  className="inline-flex items-center justify-center rounded-lg border border-zinc-200 px-5 py-3 text-sm font-medium text-zinc-700 transition hover:bg-zinc-50"
                >
                  Sign in
                </button>
              </div>

              <p className="mt-4 text-xs text-zinc-400">
                No account required for the demo.
              </p>
            </div>

            {/* ==================================================
                PRODUCT SHOWCASE
            ================================================== */}

            <div className="mx-auto mt-20 max-w-6xl">
              <div className="overflow-hidden rounded-2xl border border-zinc-200 bg-white shadow-[0_24px_80px_rgba(0,0,0,0.07)]">
                {/* browser top */}

                <div className="flex h-11 items-center border-b border-zinc-100 px-4">
                  <div className="flex items-center gap-1.5">
                    <span className="h-2.5 w-2.5 rounded-full bg-zinc-200" />
                    <span className="h-2.5 w-2.5 rounded-full bg-zinc-200" />
                    <span className="h-2.5 w-2.5 rounded-full bg-zinc-200" />
                  </div>

                  <div className="mx-auto hidden rounded-md border border-zinc-100 bg-zinc-50 px-16 py-1.5 text-[10px] text-zinc-400 sm:block">
                    app.smartresearch.ai
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-[190px_1fr]">
                  {/* App sidebar */}

                  <div className="hidden border-r border-zinc-100 bg-zinc-50/70 p-4 md:block">
                    <div className="flex items-center gap-2">
                      <div className="flex h-7 w-7 items-center justify-center rounded-md bg-zinc-950">
                        <Sparkles
                          size={13}
                          className="text-white"
                        />
                      </div>

                      <span className="text-[11px] font-semibold">
                        Smart Research
                      </span>
                    </div>

                    <div className="mt-8">
                      <p className="px-2 text-[9px] font-semibold uppercase tracking-[0.16em] text-zinc-400">
                        Workspace
                      </p>

                      <div className="mt-2 space-y-1">
                        <MiniNav
                          icon={<div className="h-3 w-3 rounded-sm bg-zinc-800" />}
                          label="Overview"
                          active
                        />

                        <MiniNav
                          icon={<FileText size={12} />}
                          label="Sources"
                        />

                        <MiniNav
                          icon={<Search size={12} />}
                          label="Discover"
                        />

                        <MiniNav
                          icon={<BookOpen size={12} />}
                          label="Chat"
                        />
                      </div>
                    </div>

                    <div className="mt-10">
                      <p className="px-2 text-[9px] font-semibold uppercase tracking-[0.16em] text-zinc-400">
                        Workspace
                      </p>

                      <div className="mt-3 rounded-lg border border-zinc-200 bg-white p-3">
                        <p className="truncate text-[10px] font-medium">
                          Vision Mamba Research
                        </p>

                        <p className="mt-1 text-[9px] text-zinc-400">
                          21 sources
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* App content */}

                  <div className="min-h-[460px] p-5 sm:p-7">
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <p className="text-[10px] font-medium uppercase tracking-[0.14em] text-zinc-400">
                          Research workspace
                        </p>

                        <h2 className="mt-1.5 text-xl font-semibold tracking-tight">
                          Vision Mamba Research
                        </h2>

                        <p className="mt-1 text-xs text-zinc-500">
                          Understanding Vision Mamba, ViT, and state
                          space models.
                        </p>
                      </div>

                      <button className="hidden rounded-md border border-zinc-200 px-3 py-1.5 text-[10px] font-medium text-zinc-600 sm:block">
                        View sources
                      </button>
                    </div>

                    {/* Metrics */}

                    <div className="mt-7 grid grid-cols-3 divide-x divide-zinc-100 rounded-xl border border-zinc-200">
                      <Metric
                        value="21"
                        label="Sources"
                      />

                      <Metric
                        value="12"
                        label="Papers"
                      />

                      <Metric
                        value="4"
                        label="Repositories"
                      />
                    </div>

                    {/* Source list */}

                    <div className="mt-5 overflow-hidden rounded-xl border border-zinc-200">
                      <div className="flex items-center justify-between border-b border-zinc-100 px-4 py-3">
                        <span className="text-[11px] font-semibold">
                          Recently added
                        </span>

                        <span className="text-[10px] text-zinc-400">
                          View all
                        </span>
                      </div>

                      <SourceRow
                        icon={<FileText size={14} />}
                        title="Vision Mamba: Efficient Visual Representation"
                        meta="arXiv · 2024"
                      />

                      <SourceRow
                        icon={<Code2 size={14} />}
                        title="vision-mamba"
                        meta="Code2 · microsoft/vision-mamba"
                      />

                      <SourceRow
                        icon={<Code2 size={14} />}
                        title="experiment_results.py"
                        meta="Document · Python"
                      />
                    </div>

                    {/* AI interaction */}

                    <div className="mt-5 rounded-xl border border-zinc-200 bg-zinc-50/70 p-4">
                      <div className="flex gap-3">
                        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-white shadow-sm">
                          <Sparkles
                            size={14}
                            className="text-zinc-700"
                          />
                        </div>

                        <div className="min-w-0">
                          <p className="text-[11px] font-semibold">
                            Ask your research
                          </p>

                          <p className="mt-1.5 text-xs leading-5 text-zinc-500">
                            How does the repository implementation
                            differ from the approach described in
                            the paper?
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ======================================================
            TRUST / POSITIONING
        ====================================================== */}

        <section className="border-b border-zinc-100">
          <div className="mx-auto max-w-5xl px-6 py-14 text-center lg:px-8">
            <p className="text-xs font-medium text-zinc-400">
              Designed around how researchers actually work
            </p>

            <div className="mt-7 flex flex-wrap justify-center gap-x-10 gap-y-4 text-sm font-medium text-zinc-400">
              <span>Discover</span>
              <span>Collect</span>
              <span>Compare</span>
              <span>Understand</span>
              <span>Investigate</span>
            </div>
          </div>
        </section>

        {/* ======================================================
            WORKFLOW
        ====================================================== */}

        <section
          id="workflow"
          className="border-b border-zinc-100"
        >
          <div className="mx-auto max-w-7xl px-6 py-24 lg:px-8">
            <div className="max-w-xl">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-zinc-400">
                How it works
              </p>

              <h2 className="mt-3 text-3xl font-semibold tracking-[-0.03em] sm:text-4xl">
                A simpler way to research.
              </h2>

              <p className="mt-4 text-sm leading-6 text-zinc-500">
                Keep discovery, sources, and questions together
                instead of spreading your research across different
                tools.
              </p>
            </div>

            <div className="mt-14 grid gap-8 md:grid-cols-4">
              <WorkflowStep
                number="01"
                title="Discover"
                description="Find papers, repositories, models, and datasets relevant to your topic."
              />

              <WorkflowStep
                number="02"
                title="Collect"
                description="Bring the sources that matter into one research workspace."
              />

              <WorkflowStep
                number="03"
                title="Ask"
                description="Ask questions about your documents and repositories."
              />

              <WorkflowStep
                number="04"
                title="Connect"
                description="Compare evidence and understand relationships across sources."
              />
            </div>
          </div>
        </section>

        {/* ======================================================
            CAPABILITIES
        ====================================================== */}

        <section
          id="capabilities"
          className="border-b border-zinc-100"
        >
          <div className="mx-auto max-w-7xl px-6 py-24 lg:px-8">
            <div className="max-w-xl">
              <p className="text-xs font-semibold uppercase tracking-[0.16em] text-zinc-400">
                Capabilities
              </p>

              <h2 className="mt-3 text-3xl font-semibold tracking-[-0.03em] sm:text-4xl">
                Everything your research needs.
              </h2>
            </div>

            <div className="mt-14 grid gap-4 md:grid-cols-3">
              <CapabilityCard
                icon={<Search size={17} />}
                title="Research discovery"
                description="Search across the research sources you care about and bring useful results into your workspace."
              />

              <CapabilityCard
                icon={<BookOpen size={17} />}
                title="Source-grounded AI"
                description="Ask questions about your collected documents and repositories with evidence from your sources."
              />

              <CapabilityCard
                icon={<Code2 size={17} />}
                title="Cross-source thinking"
                description="Connect research with implementation and compare information across multiple sources."
              />
            </div>
          </div>
        </section>

        {/* ======================================================
            SUPPORTED SOURCES
        ====================================================== */}

        <section className="border-b border-zinc-100">
          <div className="mx-auto max-w-7xl px-6 py-20 lg:px-8">
            <div className="rounded-2xl border border-zinc-200 bg-zinc-50/60 p-8 sm:p-10">
              <div className="flex flex-col gap-8 md:flex-row md:items-center md:justify-between">
                <div className="max-w-lg">
                  <p className="text-xs font-semibold uppercase tracking-[0.16em] text-zinc-400">
                    Your sources
                  </p>

                  <h2 className="mt-3 text-2xl font-semibold tracking-tight">
                    Keep everything together.
                  </h2>

                  <p className="mt-3 text-sm leading-6 text-zinc-500">
                    Your workspace can bring together documents,
                    repositories, and research from different
                    sources as your project grows.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  <SourceBadge
                    icon={<FileText size={15} />}
                    label="Documents"
                  />

                  <SourceBadge
                    icon={<Code2 size={15} />}
                    label="Github"
                  />

                  <SourceBadge
                    icon={<Search size={15} />}
                    label="arXiv"
                  />

                  <SourceBadge
                    icon={<Code2 size={15} />}
                    label="Models"
                  />
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ======================================================
            FINAL CTA
        ====================================================== */}

        <section>
          <div className="mx-auto max-w-3xl px-6 py-28 text-center lg:px-8">
            <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-xl bg-zinc-950">
              <Sparkles
                size={17}
                className="text-white"
              />
            </div>

            <h2 className="mt-6 text-3xl font-semibold tracking-[-0.035em] sm:text-4xl">
              Start with your research.
            </h2>

            <p className="mx-auto mt-4 max-w-xl text-sm leading-6 text-zinc-500">
              Explore the workspace first. Create an account when
              you're ready to keep going.
            </p>

            <button
              onClick={onTryDemo}
              className="mt-8 inline-flex items-center gap-2 rounded-lg bg-zinc-950 px-5 py-3 text-sm font-medium text-white shadow-sm transition hover:bg-zinc-800"
            >
              Try the demo
              <ArrowRight size={16} />
            </button>

            <div className="mt-5 flex items-center justify-center gap-2 text-xs text-zinc-400">
              <Check size={13} />
              No account required
            </div>
          </div>
        </section>
      </main>

      {/* ========================================================
          FOOTER
      ======================================================== */}

      <footer className="border-t border-zinc-100">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 py-7 sm:flex-row sm:items-center sm:justify-between lg:px-8">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-zinc-950">
              <Sparkles
                size={12}
                className="text-white"
              />
            </div>

            <span className="text-xs font-semibold">
              Smart Research AI
            </span>
          </div>

          <p className="text-xs text-zinc-400">
            Research, organized.
          </p>
        </div>
      </footer>
    </div>
  );
}

/* ============================================================
   Small reusable UI components
   ============================================================ */

interface MiniNavProps {
  icon: ReactNode;
  label: string;
  active?: boolean;
}

function MiniNav({
  icon,
  label,
  active = false,
}: MiniNavProps) {
  return (
    <div
      className={[
        "flex items-center gap-2 rounded-md px-2.5 py-2 text-[10px] font-medium",
        active
          ? "bg-white text-zinc-900 shadow-sm"
          : "text-zinc-400",
      ].join(" ")}
    >
      <span className="text-zinc-500">
        {icon}
      </span>

      {label}
    </div>
  );
}

interface MetricProps {
  value: string;
  label: string;
}

function Metric({
  value,
  label,
}: MetricProps) {
  return (
    <div className="px-4 py-4">
      <p className="text-lg font-semibold tracking-tight">
        {value}
      </p>

      <p className="mt-0.5 text-[10px] text-zinc-400">
        {label}
      </p>
    </div>
  );
}

interface SourceRowProps {
  icon: ReactNode;
  title: string;
  meta: string;
}

function SourceRow({
  icon,
  title,
  meta,
}: SourceRowProps) {
  return (
    <div className="flex items-center gap-3 border-b border-zinc-100 px-4 py-3 last:border-b-0">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-zinc-50 text-zinc-500">
        {icon}
      </div>

      <div className="min-w-0">
        <p className="truncate text-[10px] font-medium text-zinc-800">
          {title}
        </p>

        <p className="mt-0.5 text-[9px] text-zinc-400">
          {meta}
        </p>
      </div>
    </div>
  );
}

interface WorkflowStepProps {
  number: string;
  title: string;
  description: string;
}

function WorkflowStep({
  number,
  title,
  description,
}: WorkflowStepProps) {
  return (
    <div>
      <p className="text-xs font-semibold text-zinc-400">
        {number}
      </p>

      <h3 className="mt-5 text-sm font-semibold">
        {title}
      </h3>

      <p className="mt-2 text-sm leading-6 text-zinc-500">
        {description}
      </p>
    </div>
  );
}

interface CapabilityCardProps {
  icon: ReactNode;
  title: string;
  description: string;
}

function CapabilityCard({
  icon,
  title,
  description,
}: CapabilityCardProps) {
  return (
    <div className="rounded-xl border border-zinc-200 p-6 transition-shadow duration-200 hover:shadow-sm">
      <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-zinc-50 text-zinc-700">
        {icon}
      </div>

      <h3 className="mt-5 text-sm font-semibold">
        {title}
      </h3>

      <p className="mt-2 text-sm leading-6 text-zinc-500">
        {description}
      </p>
    </div>
  );
}

interface SourceBadgeProps {
  icon: ReactNode;
  label: string;
}

function SourceBadge({
  icon,
  label,
}: SourceBadgeProps) {
  return (
    <div className="flex min-w-[110px] items-center gap-2 rounded-lg border border-zinc-200 bg-white px-3 py-3">
      <span className="text-zinc-500">
        {icon}
      </span>

      <span className="text-xs font-medium text-zinc-700">
        {label}
      </span>
    </div>
  );
}