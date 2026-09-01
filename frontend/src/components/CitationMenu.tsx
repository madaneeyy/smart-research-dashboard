import { useState } from "react";
import { Check, Clipboard, X } from "lucide-react";
import {
  copyText,
  formatApaCitation,
  formatBibtexCitation,
  type ArxivCitation,
} from "../lib/citations";

export function CitationMenu({
  citation,
  onClose,
}: {
  citation: ArxivCitation;
  onClose: () => void;
}) {
  const [copiedFormat, setCopiedFormat] = useState<
    "apa" | "bibtex" | null
  >(null);

  const copyCitation = async (
    format: "apa" | "bibtex",
  ) => {
    const text =
      format === "apa"
        ? formatApaCitation(citation)
        : formatBibtexCitation(citation);

    try {
      await copyText(text);
      setCopiedFormat(format);
      window.setTimeout(() => setCopiedFormat(null), 1800);
    } catch {
      // Clipboard access can be unavailable in some browsers.
    }
  };

  return (
    <div
      className="absolute right-0 top-11 z-40 w-80 overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--paper)] p-3 shadow-xl sm:w-96"
      onClick={(event) => event.stopPropagation()}
    >
      <div className="flex items-start justify-between gap-3 px-1 pb-2">
        <div>
          <p className="text-xs font-semibold text-[var(--ink)]">
            Copy citation
          </p>
          <p className="mt-0.5 text-[9px] leading-4 text-[var(--muted)]">
            Ready to paste into your references.
          </p>
        </div>

        <button
          type="button"
          onClick={onClose}
          className="rounded-md p-1 text-[var(--muted)] transition-colors hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
          aria-label="Close citation menu"
        >
          <X size={13} />
        </button>
      </div>

      <CitationOption
        label="APA 7"
        preview={formatApaCitation(citation)}
        copied={copiedFormat === "apa"}
        onClick={() => void copyCitation("apa")}
      />

      <CitationOption
        label="BibTeX"
        preview={formatBibtexCitation(citation)}
        copied={copiedFormat === "bibtex"}
        onClick={() => void copyCitation("bibtex")}
      />
    </div>
  );
}

function CitationOption({
  label,
  preview,
  copied,
  onClick,
}: {
  label: string;
  preview: string;
  copied: boolean;
  onClick: () => void;
}) {
  return (
    <div className="mb-2 rounded-lg border border-[var(--line-soft)] bg-[var(--paper-dim)] p-2.5 last:mb-0">
      <div className="flex items-center justify-between gap-2">
        <span className="font-[var(--font-mono)] text-[9px] uppercase tracking-[0.08em] text-[var(--muted)]">
          {label}
        </span>

        <button
          type="button"
          onClick={onClick}
          className="inline-flex shrink-0 items-center gap-1 rounded-md border border-[var(--line)] bg-[var(--paper)] px-2 py-1 text-[10px] font-medium text-[var(--ink-soft)] transition-colors hover:bg-[var(--paper-dim)] hover:text-[var(--ink)]"
        >
          {copied ? <Check size={11} /> : <Clipboard size={11} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>

      <p className="mt-2 max-h-20 overflow-auto text-[10px] leading-4 text-[var(--ink-soft)]">
        {preview}
      </p>
    </div>
  );
}
