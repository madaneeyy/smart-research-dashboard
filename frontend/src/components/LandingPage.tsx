import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  ArrowRight,
  BookOpen,
  Check,
  Code2,
  Crosshair,
  FileText,
  MessageSquare,
  Moon,
  Search,
  Sun,
} from "lucide-react";
import { useTheme } from "../context/ThemeContext";
interface LandingPageProps {
  onTryDemo: () => void;
  onSignIn: () => void;
}




export function LandingPage({ onTryDemo, onSignIn }: LandingPageProps) {
  const workflowRef = useRef<HTMLElement | null>(null);
  const capabilitiesRef = useRef<HTMLElement | null>(null);

  const [workflowVisible, setWorkflowVisible] = useState(false);
  const [capabilitiesVisible, setCapabilitiesVisible] = useState(false);
  const [activeCapability, setActiveCapability] = useState(1);
  const { theme, toggleTheme } = useTheme();

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.target === workflowRef.current) {
            setWorkflowVisible(entry.isIntersecting);
          }
          if (entry.target === capabilitiesRef.current) {
            setCapabilitiesVisible(entry.isIntersecting);
          }
        });
      },
      { threshold: 0.18 },
    );

    if (workflowRef.current) observer.observe(workflowRef.current);
    if (capabilitiesRef.current) observer.observe(capabilitiesRef.current);

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!capabilitiesVisible) return;
    const interval = window.setInterval(() => {
      setActiveCapability((current) => (current + 1) % 3);
    }, 4600);
    return () => window.clearInterval(interval);
  }, [capabilitiesVisible]);

  return (
    <div
      className="min-h-screen overflow-x-hidden bg-[var(--paper)] text-[var(--ink)] transition-colors duration-300"
    >
      {/* ======================================================
          NAVIGATION
      ====================================================== */}

      <header className="sticky top-0 z-50 border-b border-[var(--line)] bg-[var(--paper)]/90 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6 lg:px-8">
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
            className="group flex items-center gap-0.5"
          >
            <img
              src="/bujhalogo.png"
              alt="Bujha AI"
              className="h-11 w-11 object-contain transition-transform duration-200 group-hover:scale-[1.05]"
            />
             <span className="font-[var(--font-display)] text-sm font-semibold tracking-[-0.02em]">
               Bujha AI
             </span>
          </button>

          <nav className="hidden items-center gap-7 md:flex">
            <a
              href="#product"
              className="font-[var(--font-mono)] text-[11px] uppercase tracking-[0.1em] text-[var(--ink-soft)] transition-colors duration-200 hover:text-[var(--ink)]"
            >
              Product
            </a>
            <a
              href="#workflow"
              className="font-[var(--font-mono)] text-[11px] uppercase tracking-[0.1em] text-[var(--ink-soft)] transition-colors duration-200 hover:text-[var(--ink)]"
            >
              How it works
            </a>
            <a
              href="#capabilities"
              className="font-[var(--font-mono)] text-[11px] uppercase tracking-[0.1em] text-[var(--ink-soft)] transition-colors duration-200 hover:text-[var(--ink)]"
            >
              Capabilities
            </a>
          </nav>

          <div className="flex items-center gap-2">
            <button
              onClick={toggleTheme}
              aria-label={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
              className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-[var(--line)] text-[var(--ink-soft)] transition-colors duration-200 hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
            >
              {theme === "light" ? <Moon size={15} /> : <Sun size={15} />}
            </button>

            <button
              onClick={onSignIn}
              className="hidden rounded-md px-3.5 py-2 text-sm font-medium text-[var(--ink-soft)] transition-colors duration-200 hover:bg-[var(--paper-dim)] hover:text-[var(--ink)] sm:inline-flex"
            >
              Sign in
            </button>
            <button
              onClick={onTryDemo}
              className="group inline-flex items-center gap-2 rounded-md bg-[var(--ink)] px-4 py-2.5 text-sm font-medium text-[var(--paper)] shadow-sm transition-all duration-200 hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-md"
            >
              Try demo
              <ArrowRight size={15} className="transition-transform duration-200 group-hover:translate-x-0.5" />
            </button>
          </div>
        </div>
      </header>

      <main>
        {/* ======================================================
            HERO
        ====================================================== */}

        <section id="product" className="relative border-b border-[var(--line)]">
          {/* Blueprint grid backdrop, faded at the edges */}
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0"
            style={{
              backgroundImage:
                "linear-gradient(var(--line-soft) 1px, transparent 1px), linear-gradient(90deg, var(--line-soft) 1px, transparent 1px)",
              backgroundSize: "34px 34px",
              WebkitMaskImage:
                "radial-gradient(ellipse 70% 55% at 50% 20%, black 40%, transparent 85%)",
              maskImage:
                "radial-gradient(ellipse 70% 55% at 50% 20%, black 40%, transparent 85%)",
            }}
          />

          <div className="relative mx-auto max-w-7xl px-6 pb-24 pt-20 lg:px-8 lg:pb-32 lg:pt-28">
            <div className="mx-auto max-w-4xl text-center">
              {/* Eyebrow */}
              <div className="inline-flex animate-[fadeUp_0.65s_ease-out_both] items-center gap-2 rounded-full border border-[var(--line)] bg-[var(--paper)] px-3.5 py-1.5 font-[var(--font-mono)] text-[11px] uppercase tracking-[0.1em] text-[var(--ink-soft)]">
                <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
                AI-powered research workspace
              </div>

              {/* Heading */}
              <h1 className="mt-7 animate-[fadeUp_0.75s_0.08s_ease-out_both] font-[var(--font-display)] text-5xl font-semibold leading-[0.98] tracking-[-0.03em] text-[var(--ink)] sm:text-6xl lg:text-[74px]">
                Trace the sources.
                <br />
                <span className="text-[var(--accent)]">Reach</span> the answer.
              </h1>

              {/* Description */}
              <p className="mx-auto mt-7 max-w-2xl animate-[fadeUp_0.75s_0.16s_ease-out_both] text-base leading-7 text-[var(--ink-soft)] sm:text-lg">
                Discover research, organize the sources that matter, and ask
                grounded questions across your documents and code — all in one
                focused workspace.
              </p>

              {/* CTA */}
              <div className="mt-9 flex animate-[fadeUp_0.75s_0.24s_ease-out_both] flex-col justify-center gap-3 sm:flex-row">
                <button
                  onClick={onTryDemo}
                  className="group inline-flex items-center justify-center gap-2 rounded-md bg-[var(--ink)] px-5 py-3 text-sm font-medium text-[var(--paper)] shadow-sm transition-all duration-200 hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-md"
                >
                  Try the demo
                  <ArrowRight size={16} className="transition-transform duration-200 group-hover:translate-x-0.5" />
                </button>
                <button
                  onClick={onSignIn}
                  className="inline-flex items-center justify-center rounded-md border border-[var(--line)] bg-[var(--paper)] px-5 py-3 text-sm font-medium text-[var(--ink)] transition-all duration-200 hover:-translate-y-px hover:bg-[var(--paper-dim)] hover:shadow-sm"
                >
                  Sign in
                </button>
              </div>

              <p className="mt-4 animate-[fadeIn_0.7s_0.4s_ease-out_both] font-[var(--font-mono)] text-[11px] text-[var(--muted)]">
                No account required for the demo.
              </p>
            </div>

            {/* ==================================================
                PRODUCT SHOWCASE — framed like a pinned spec sheet
            ================================================== */}

            <div className="relative mx-auto mt-20 max-w-6xl">
              <div className="relative animate-[previewIn_0.9s_0.25s_ease-out_both]">
                <CornerBrackets color="var(--accent)" size={18} />
                <div
                  className="overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--paper)]"
                  style={{ boxShadow: "var(--shadow-lg)" }}
                >
                  {/* Browser top */}
                  <div className="flex h-11 items-center border-b border-[var(--line-soft)] px-4">
                    <div className="flex items-center gap-1.5">
                      <span className="h-2.5 w-2.5 rounded-full border border-[var(--line)]" />
                      <span className="h-2.5 w-2.5 rounded-full border border-[var(--line)]" />
                      <span className="h-2.5 w-2.5 rounded-full border border-[var(--line)]" />
                    </div>
                    <div className="mx-auto hidden rounded-md border border-[var(--line-soft)] bg-[var(--paper-dim)] px-16 py-1.5 font-[var(--font-mono)] text-[10px] text-[var(--muted)] sm:block">
                      app.bujha.ai
                    </div>
                    <span className="ml-auto hidden font-[var(--font-mono)] text-[9px] uppercase tracking-[0.12em] text-[var(--muted)] sm:block">
                      REF—014
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-[190px_1fr]">
                    {/* App sidebar */}
                    <div className="hidden border-r border-[var(--line-soft)] bg-[var(--paper-dim)] p-4 md:block">
                      <div className="flex items-center gap-2">
                        <div className="flex h-7 w-7 items-center justify-center rounded-md bg-[var(--ink)]">
                          <Crosshair size={13} className="text-[var(--paper)]" />
                        </div>
                        <span className="font-[var(--font-display)] text-[11px] font-semibold">
                          Bujha AI
                        </span>
                      </div>

                      <div className="mt-8">
                        <p className="px-2 font-[var(--font-mono)] text-[9px] uppercase tracking-[0.16em] text-[var(--muted)]">
                          Workspace
                        </p>
                        <div className="mt-2 space-y-1">
                          <MiniNav icon={<div className="h-3 w-3 rounded-sm bg-[var(--ink)]" />} label="Overview" active />
                          <MiniNav icon={<FileText size={12} />} label="Sources" />
                          <MiniNav icon={<Search size={12} />} label="Discover" />
                          <MiniNav icon={<BookOpen size={12} />} label="Chat" />
                        </div>
                      </div>

                      <div className="mt-10">
                        <p className="px-2 font-[var(--font-mono)] text-[9px] uppercase tracking-[0.16em] text-[var(--muted)]">
                          Active
                        </p>
                        <div className="mt-3 rounded-lg border border-[var(--line)] bg-[var(--paper)] p-3">
                          <p className="truncate text-[10px] font-medium">Vision Mamba Research</p>
                          <p className="mt-1 font-[var(--font-mono)] text-[9px] text-[var(--muted)]">21 sources</p>
                        </div>
                      </div>
                    </div>

                    {/* App content */}
                    <div className="min-h-[460px] p-5 sm:p-7">
                      <div className="flex items-start justify-between gap-4">
                        <div>
                          <p className="font-[var(--font-mono)] text-[10px] uppercase tracking-[0.14em] text-[var(--muted)]">
                            Research workspace
                          </p>
                          <h2 className="mt-1.5 font-[var(--font-display)] text-xl font-semibold tracking-tight">
                            Vision Mamba Research
                          </h2>
                          <p className="mt-1 text-xs text-[var(--ink-soft)]">
                            Understanding Vision Mamba, ViT, and state space models.
                          </p>
                        </div>
                        <button className="hidden rounded-md border border-[var(--line)] px-3 py-1.5 text-[10px] font-medium text-[var(--ink-soft)] transition hover:bg-[var(--paper-dim)] sm:block">
                          View sources
                        </button>
                      </div>

                      {/* Metrics */}
                      <div className="mt-7 grid grid-cols-3 divide-x divide-[var(--line-soft)] rounded-xl border border-[var(--line)]">
                        <Metric value="21" label="Sources" />
                        <Metric value="12" label="Papers" />
                        <Metric value="4" label="Repositories" />
                      </div>

                      {/* Source list */}
                      <div className="mt-5 overflow-hidden rounded-xl border border-[var(--line)]">
                        <div className="flex items-center justify-between border-b border-[var(--line-soft)] px-4 py-3">
                          <span className="text-[11px] font-semibold">Recently added</span>
                          <span className="font-[var(--font-mono)] text-[10px] text-[var(--muted)]">View all</span>
                        </div>
                        <SourceRow icon={<FileText size={14} />} title="Vision Mamba: Efficient Visual Representation" meta="arXiv · 2024" />
                        <SourceRow icon={<Code2 size={14} />} title="vision-mamba" meta="GitHub · microsoft/vision-mamba" />
                        <SourceRow icon={<Code2 size={14} />} title="experiment_results.py" meta="Document · Python" />
                      </div>

                      {/* AI interaction */}
                      <div className="mt-5 rounded-xl border border-[var(--line)] bg-[var(--paper-dim)] p-4 transition-colors duration-300 hover:bg-[var(--cyan-dim)]">
                        <div className="flex gap-3">
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--paper)] shadow-sm">
                            <MessageSquare size={14} className="text-[var(--cyan)]" />
                          </div>
                          <div className="min-w-0">
                            <p className="font-[var(--font-mono)] text-[10px] uppercase tracking-[0.1em] text-[var(--muted)]">
                              Ask your sources
                            </p>
                            <p className="mt-1.5 text-xs leading-5 text-[var(--ink-soft)]">
                              How does the repository implementation differ from
                              the approach described in the paper?
                            </p>
                          </div>
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
            WORKFLOW
        ====================================================== */}

        <section ref={workflowRef} id="workflow" className="border-b border-[var(--line)]">
          <div className="mx-auto max-w-7xl px-6 py-14 lg:px-8">
            <div className="max-w-xl">
              <SectionEyebrow visible={workflowVisible}>The process</SectionEyebrow>
              <h2
                className={[
                  "mt-3 font-[var(--font-display)] text-3xl font-semibold tracking-[-0.02em] sm:text-4xl",
                  "transition-all duration-700 delay-100",
                  workflowVisible ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0",
                ].join(" ")}
              >
                Four steps, one straight line.
              </h2>
              <p
                className={[
                  "mt-3 text-sm leading-6 text-[var(--ink-soft)]",
                  "transition-all duration-700 delay-150",
                  workflowVisible ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0",
                ].join(" ")}
              >
                Keep discovery, sources, and questions together instead of
                spreading your research across different tools.
              </p>
            </div>

            <div className="relative mt-8">
              {/* Connector line with a traveling pulse once visible */}
              <div className="pointer-events-none absolute left-[12.5%] right-[12.5%] top-5 hidden h-px overflow-hidden md:block">
                <div
                  className={[
                    "h-full origin-left bg-[var(--line)]",
                    "transition-transform duration-[1400ms] ease-out",
                    workflowVisible ? "scale-x-100" : "scale-x-0",
                  ].join(" ")}
                />
                {workflowVisible && (
                  <div
                    className="absolute top-0 h-px w-24 bg-gradient-to-r from-transparent via-[var(--accent)] to-transparent [animation:travel_3.2s_ease-in-out_1.2s_infinite]"
                  />
                )}
              </div>

              <div className="grid gap-6 md:grid-cols-4">
                <WorkflowStep number="01" title="Discover" description="Find papers, repositories, models, and datasets relevant to your topic." visible={workflowVisible} delay="0ms" />
                <WorkflowStep number="02" title="Collect" description="Bring the sources that matter into one research workspace." visible={workflowVisible} delay="120ms" />
                <WorkflowStep number="03" title="Ask" description="Ask questions about your documents and repositories." visible={workflowVisible} delay="240ms" />
                <WorkflowStep number="04" title="Connect" description="Compare evidence and understand relationships across sources." visible={workflowVisible} delay="360ms" />
              </div>
            </div>
          </div>
        </section>

        {/* ======================================================
            CAPABILITIES
        ====================================================== */}

        <section ref={capabilitiesRef} id="capabilities" className="border-b border-[var(--line)]">
          <div className="mx-auto max-w-7xl px-6 py-14 lg:px-8">
            <div className="max-w-xl">
              <SectionEyebrow visible={capabilitiesVisible}>Capabilities</SectionEyebrow>
              <h2
                className={[
                  "mt-3 font-[var(--font-display)] text-3xl font-semibold tracking-[-0.02em] sm:text-4xl",
                  "transition-all duration-700 delay-100",
                  capabilitiesVisible ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0",
                ].join(" ")}
              >
                Every source, one map.
              </h2>
              <p
                className={[
                  "mt-3 max-w-lg text-sm leading-6 text-[var(--ink-soft)]",
                  "transition-all duration-700 delay-150",
                  capabilitiesVisible ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0",
                ].join(" ")}
              >
                A focused set of tools for finding information, understanding
                it, and tracing the connections between different sources.
              </p>
            </div>

            <div
              className={[
                "mt-8 grid gap-4 md:grid-cols-[0.85fr_1.15fr]",
                "transition-all duration-700 delay-200",
                capabilitiesVisible ? "translate-y-0 opacity-100" : "translate-y-5 opacity-0",
              ].join(" ")}
            >
              {/* Capability navigation */}
              <div className="space-y-2">
                <CapabilitySelector icon={<Search size={17} />} title="Research discovery" description="Find what matters." active={activeCapability === 0} onClick={() => setActiveCapability(0)} />
                <CapabilitySelector icon={<BookOpen size={17} />} title="Source-grounded AI" description="Understand your material." active={activeCapability === 1} onClick={() => setActiveCapability(1)} />
                <CapabilitySelector icon={<Code2 size={17} />} title="Cross-source thinking" description="Connect research and implementation." active={activeCapability === 2} onClick={() => setActiveCapability(2)} />
              </div>

              {/* Capability preview — the signature connection diagram */}
              <div className="relative min-h-[300px] rounded-2xl border border-[var(--line)] bg-[var(--paper-dim)]">
                <CornerBrackets color="var(--cyan)" size={16} inset={0} />
                <div className="h-full overflow-hidden rounded-2xl p-6 sm:p-8">
                  <CapabilityPreview active={activeCapability} visible={capabilitiesVisible} />
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ======================================================
            SOURCES
        ====================================================== */}

        <section className="border-b border-[var(--line)]">
          <div className="mx-auto max-w-7xl px-6 py-12 lg:px-8">
            <div className="rounded-2xl border border-[var(--line)] bg-[var(--paper-dim)] p-8 transition-shadow duration-300 hover:shadow-sm sm:p-10">
              <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
                <div className="max-w-lg">
                  <p className="font-[var(--font-mono)] text-[11px] uppercase tracking-[0.16em] text-[var(--muted)]">
                    Your sources
                  </p>
                  <h2 className="mt-3 font-[var(--font-display)] text-2xl font-semibold tracking-tight">
                    Keep everything together.
                  </h2>
                  <p className="mt-3 text-sm leading-6 text-[var(--ink-soft)]">
                    Bring together the material that helps you understand your
                    problem and keep your research organized as it grows.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                  <SourceBadge icon={<FileText size={15} />} label="Documents" />
                  <SourceBadge icon={<Code2 size={15} />} label="GitHub" />
                  <SourceBadge icon={<Search size={15} />} label="arXiv" />
                  <SourceBadge icon={<Code2 size={15} />} label="Models" />
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ======================================================
            FINAL CTA
        ====================================================== */}

        <section>
          <div className="mx-auto max-w-3xl px-6 py-16 text-center lg:px-8">
            <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl border border-[var(--line)] bg-[var(--paper)] p-1 transition-transform duration-300 hover:scale-105">
              <img
                src="/bujhalogo.png"
                alt=""
                className="h-full w-full rounded-lg object-contain"
              />
            </div>

            <h2 className="mt-6 font-[var(--font-display)] text-3xl font-semibold tracking-[-0.02em] sm:text-4xl">
              Start with your research.
            </h2>

            <p className="mx-auto mt-4 max-w-xl text-sm leading-6 text-[var(--ink-soft)]">
              Explore the workspace first. Create an account when you're ready
              to keep going.
            </p>

            <button
              onClick={onTryDemo}
              className="group mt-8 inline-flex items-center gap-2 rounded-md bg-[var(--ink)] px-5 py-3 text-sm font-medium text-[var(--paper)] shadow-sm transition-all duration-200 hover:-translate-y-px hover:bg-[var(--accent)] hover:shadow-md"
            >
              Try the demo
              <ArrowRight size={16} className="transition-transform duration-200 group-hover:translate-x-0.5" />
            </button>

            <div className="mt-5 flex items-center justify-center gap-2 font-[var(--font-mono)] text-[11px] text-[var(--muted)]">
              <Check size={13} />
              No account required
            </div>
          </div>
        </section>
      </main>

      {/* ========================================================
          FOOTER
      ======================================================== */}

      <footer className="border-t border-[var(--line)]">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 py-6 sm:flex-row sm:items-center sm:justify-between lg:px-8">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-[var(--ink)]">
              <Crosshair size={12} className="text-[var(--paper)]" />
            </div>
            <span className="font-[var(--font-display)] text-xs font-semibold">Bujha AI</span>
          </div>
          <p className="font-[var(--font-mono)] text-[11px] text-[var(--muted)]">Research, understood.</p>
        </div>
      </footer>

      {/* ========================================================
          GLOBAL STYLES
      ======================================================== */}

      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(14px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes previewIn {
          from { opacity: 0; transform: translateY(22px) scale(0.985); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        @keyframes travel {
          0% { left: -10%; }
          100% { left: 110%; }
        }
        @keyframes dashFlow {
          to { stroke-dashoffset: -24; }
        }
        @keyframes pulseRing {
          0%, 100% { opacity: 0.35; transform: scale(1); }
          50% { opacity: 0.9; transform: scale(1.12); }
        }

        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after {
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
          }
        }
      `}</style>
    </div>
  );
}

/* ============================================================
   SMALL UI COMPONENTS
   ============================================================ */

function CornerBrackets({
  color,
  size = 16,
  inset = -1,
}: {
  color: string;
  size?: number;
  /** Distance the bracket sits from the parent's edge. Use 0+ when the
   *  parent (or a sibling) clips with overflow-hidden + rounded corners,
   *  so the bracket isn't cut off by the curve. */
  inset?: number;
}) {
  const base = "pointer-events-none absolute border-current";
  return (
    <div style={{ color }} className="absolute inset-0 z-10">
      <span className={base} style={{ top: inset, left: inset, width: size, height: size, borderTop: "2px solid", borderLeft: "2px solid" }} />
      <span className={base} style={{ top: inset, right: inset, width: size, height: size, borderTop: "2px solid", borderRight: "2px solid" }} />
      <span className={base} style={{ bottom: inset, left: inset, width: size, height: size, borderBottom: "2px solid", borderLeft: "2px solid" }} />
      <span className={base} style={{ bottom: inset, right: inset, width: size, height: size, borderBottom: "2px solid", borderRight: "2px solid" }} />
    </div>
  );
}

function SectionEyebrow({ children, visible }: { children: ReactNode; visible: boolean }) {
  return (
    <p
      className={[
        "flex items-center gap-2 font-[var(--font-mono)] text-[11px] uppercase tracking-[0.16em] text-[var(--muted)]",
        "transition-all duration-700",
        visible ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0",
      ].join(" ")}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)]" />
      {children}
    </p>
  );
}

interface MiniNavProps {
  icon: ReactNode;
  label: string;
  active?: boolean;
}

function MiniNav({ icon, label, active = false }: MiniNavProps) {
  return (
    <div
      className={[
        "flex items-center gap-2 rounded-md px-2.5 py-2",
        "text-[10px] font-medium transition-colors duration-200",
        active ? "bg-[var(--paper)] text-[var(--ink)] shadow-sm" : "text-[var(--muted)]",
      ].join(" ")}
    >
      <span className="text-[var(--ink-soft)]">{icon}</span>
      {label}
    </div>
  );
}

interface MetricProps {
  value: string;
  label: string;
}

function Metric({ value, label }: MetricProps) {
  return (
    <div className="px-4 py-4">
      <p className="font-[var(--font-display)] text-lg font-semibold tracking-tight">{value}</p>
      <p className="mt-0.5 font-[var(--font-mono)] text-[10px] text-[var(--muted)]">{label}</p>
    </div>
  );
}

interface SourceRowProps {
  icon: ReactNode;
  title: string;
  meta: string;
}

function SourceRow({ icon, title, meta }: SourceRowProps) {
  return (
    <div className="group flex items-center gap-3 border-b border-[var(--line-soft)] px-4 py-3 last:border-b-0">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-[var(--paper-dim)] text-[var(--ink-soft)] transition-transform duration-200 group-hover:-translate-y-px">
        {icon}
      </div>
      <div className="min-w-0">
        <p className="truncate text-[10px] font-medium text-[var(--ink)]">{title}</p>
        <p className="mt-0.5 font-[var(--font-mono)] text-[9px] text-[var(--muted)]">{meta}</p>
      </div>
    </div>
  );
}

interface WorkflowStepProps {
  number: string;
  title: string;
  description: string;
  visible: boolean;
  delay: string;
}

function WorkflowStep({ number, title, description, visible, delay }: WorkflowStepProps) {
  return (
    <div
      style={{ transitionDelay: delay }}
      className={[
        "relative transition-all duration-700 ease-out",
        visible ? "translate-y-0 opacity-100" : "translate-y-6 opacity-0",
      ].join(" ")}
    >
      <div className="relative z-10 flex h-9 w-9 items-center justify-center rounded-md border border-[var(--line)] bg-[var(--paper)] font-[var(--font-mono)] text-[11px] font-semibold text-[var(--ink-soft)] shadow-sm">
        {number}
      </div>
      <h3 className="mt-5 font-[var(--font-display)] text-sm font-semibold">{title}</h3>
      <p className="mt-2 max-w-xs text-sm leading-6 text-[var(--ink-soft)]">{description}</p>
    </div>
  );
}

interface CapabilitySelectorProps {
  icon: ReactNode;
  title: string;
  description: string;
  active: boolean;
  onClick: () => void;
}

function CapabilitySelector({ icon, title, description, active, onClick }: CapabilitySelectorProps) {
  return (
    <button
      onClick={onClick}
      className={[
        "group w-full rounded-xl border p-4 text-left transition-all duration-300",
        active ? "border-[var(--accent)]/40 bg-[var(--paper)] shadow-sm" : "border-transparent bg-transparent hover:border-[var(--line)] hover:bg-[var(--paper-dim)]",
      ].join(" ")}
    >
      <div className="flex items-start gap-3">
        <div
          className={[
            "flex h-9 w-9 shrink-0 items-center justify-center rounded-lg transition-all duration-300",
            active ? "bg-[var(--accent)] text-white" : "bg-[var(--paper-dim)] text-[var(--ink-soft)] group-hover:bg-[var(--paper)]",
          ].join(" ")}
        >
          {icon}
        </div>
        <div>
          <p className="text-sm font-semibold">{title}</p>
          <p className="mt-1 text-xs leading-5 text-[var(--ink-soft)]">{description}</p>
        </div>
      </div>
    </button>
  );
}

interface CapabilityPreviewProps {
  active: number;
  visible: boolean;
}

// All positions share one 0–100 coordinate space that maps 1:1 to percentages,
// so the SVG paths and the HTML icon overlays always line up regardless of
// the container's actual pixel aspect ratio (see preserveAspectRatio="none" below).
const ICON_LEFT = 10; // % — left edge of each source icon
const ICON_SIZE = 15; // % — icon box width/height, used to start the line at its edge
const HUB_X = 84; // % — x position of the central "answer" node
const HUB_Y = 50; // % — y position of the central "answer" node

const NODES = [
  { key: "papers", label: "Papers", icon: FileText, top: 15 },
  { key: "repos", label: "Repos", icon: Code2, top: 50 },
  { key: "docs", label: "Docs", icon: BookOpen, top: 85 },
];

function CapabilityPreview({ active, visible }: CapabilityPreviewProps) {
  const previews = [
    { eyebrow: "DISCOVER", title: "Find the sources that matter.", description: "Search across research and collect useful material into one focused workspace.", glow: 0 },
    { eyebrow: "UNDERSTAND", title: "Ask questions grounded in your sources.", description: "Get answers based on the documents and repositories you've actually collected.", glow: 2 },
    { eyebrow: "CONNECT", title: "Understand how ideas connect.", description: "Compare research with implementations and reason across multiple sources.", glow: -1 },
  ];
  const preview = previews[active];

  return (
    <div
      className={[
        "flex h-full flex-col gap-8 sm:flex-row sm:items-center",
        "transition-all duration-700",
        visible ? "translate-y-0 opacity-100" : "translate-y-5 opacity-0",
      ].join(" ")}
    >
      {/* Diagram — a single 0–100 coordinate space shared by the SVG lines
          and the HTML icon overlays, stretched to fill the box exactly
          (preserveAspectRatio="none") so nothing drifts out of alignment. */}
      <div className="relative h-44 w-full shrink-0 sm:h-full sm:w-44">
        <svg
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          className="absolute inset-0 h-full w-full overflow-visible"
        >
          {NODES.map((node, i) => {
            const isGlowing = preview.glow === -1 || preview.glow === i;
            const startX = ICON_LEFT + ICON_SIZE;
            const midX = (startX + HUB_X) / 2;
            return (
              <path
                key={node.key}
                d={`M ${startX} ${node.top} C ${midX} ${node.top}, ${midX} ${HUB_Y}, ${HUB_X} ${HUB_Y}`}
                fill="none"
                stroke={isGlowing ? "var(--accent)" : "var(--line)"}
                strokeWidth={isGlowing ? 1.4 : 1}
                strokeLinecap="round"
                strokeDasharray="3 4"
                vectorEffect="non-scaling-stroke"
                style={{
                  animation: isGlowing ? "dashFlow 0.9s linear infinite" : undefined,
                  transition: "stroke 0.4s ease, stroke-width 0.4s ease",
                }}
              />
            );
          })}

          {/* Connect view also traces the source-to-source relationships,
              since that capability is specifically about cross-source links. */}
          {preview.glow === -1 && (
            <path
              d={`M ${ICON_LEFT + ICON_SIZE} ${NODES[0].top} L ${ICON_LEFT + ICON_SIZE} ${NODES[2].top}`}
              fill="none"
              stroke="var(--cyan)"
              strokeWidth={1}
              strokeLinecap="round"
              strokeDasharray="1 4"
              vectorEffect="non-scaling-stroke"
              opacity={0.7}
            />
          )}
        </svg>

        {NODES.map((node, i) => {
          const Icon = node.icon;
          const isGlowing = preview.glow === -1 || preview.glow === i;
          return (
            <div
              key={node.key}
              className="absolute flex -translate-y-1/2 items-center gap-1.5"
              style={{ left: `${ICON_LEFT}%`, top: `${node.top}%` }}
            >
              <div
                className="flex h-7 w-7 items-center justify-center rounded-md border bg-[var(--paper)] shadow-sm transition-all duration-300"
                style={{
                  borderColor: isGlowing ? "var(--accent)" : "var(--line)",
                  color: isGlowing ? "var(--accent)" : "var(--ink-soft)",
                }}
              >
                <Icon size={12} />
              </div>
            </div>
          );
        })}

        {/* Answer node */}
        <div
          className="absolute flex -translate-x-1/2 -translate-y-1/2 items-center justify-center"
          style={{ left: `${HUB_X}%`, top: `${HUB_Y}%` }}
        >
          <span className="absolute h-9 w-9 rounded-full bg-[var(--accent)]/25 [animation:pulseRing_2.4s_ease-in-out_infinite]" />
          <div className="relative flex h-8 w-8 items-center justify-center rounded-full bg-[var(--ink)] text-[var(--paper)] shadow-sm">
            <MessageSquare size={13} />
          </div>
        </div>
      </div>

      {/* Text */}
      <div className="min-w-0">
        <div className="flex items-center justify-between gap-4 sm:justify-start">
          <span className="font-[var(--font-mono)] text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--accent)]">
            {preview.eyebrow}
          </span>
          <span className="font-[var(--font-mono)] text-[10px] text-[var(--muted)] sm:ml-3">0{active + 1} / 03</span>
        </div>
        <h3 className="mt-4 font-[var(--font-display)] text-xl font-semibold tracking-[-0.02em] text-[var(--ink)] sm:text-2xl">
          {preview.title}
        </h3>
        <p className="mt-3 max-w-md text-sm leading-6 text-[var(--ink-soft)]">{preview.description}</p>

        <div className="mt-6 h-px w-full max-w-[220px] bg-[var(--line)]">
          <div
            className="h-px bg-[var(--accent)] transition-all duration-700"
            style={{ width: `${((active + 1) / 3) * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}

interface SourceBadgeProps {
  icon: ReactNode;
  label: string;
}

function SourceBadge({ icon, label }: SourceBadgeProps) {
  return (
    <div className="flex min-w-[110px] items-center gap-2 rounded-lg border border-dashed border-[var(--line)] bg-[var(--paper)] px-3 py-3 transition-all duration-200 hover:-translate-y-px hover:border-[var(--accent)]/50 hover:shadow-sm">
      <span className="text-[var(--ink-soft)]">{icon}</span>
      <span className="text-xs font-medium text-[var(--ink)]">{label}</span>
    </div>
  );
}