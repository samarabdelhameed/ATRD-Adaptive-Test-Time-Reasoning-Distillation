import React, { useState } from "react";
import { cn } from "@/lib/utils";
import { Check, Copy, FileCode } from "lucide-react";

interface CodeBlockProps {
  code: string;
  language?: string;
  filename?: string;
  className?: string;
}

export function CodeBlock({ code, language = "python", filename, className }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const lines = code.trim().split("\n");

  return (
    <div className={cn("glass-panel rounded-lg overflow-hidden border border-default flex flex-col w-full", className)}>
      <div className="flex items-center justify-between px-4 py-2 border-b border-default bg-surface/80">
        <div className="flex items-center gap-2">
          <FileCode className="h-4 w-4 text-text-muted" />
          {filename && (
            <span className="font-mono text-xs text-text-secondary">
              {filename}
            </span>
          )}
          <span className="font-display text-[10px] font-semibold text-text-muted uppercase tracking-wider bg-void px-1.5 py-0.5 rounded border border-default">
            {language}
          </span>
        </div>

        <button
          onClick={handleCopy}
          className="p-1 rounded hover:bg-elevated text-text-muted hover:text-text-primary transition-all duration-200"
          title="Copy Code"
        >
          {copied ? <Check className="h-3.5 w-3.5 text-nvidia" /> : <Copy className="h-3.5 w-3.5" />}
        </button>
      </div>

      <div className="flex font-mono text-xs overflow-x-auto p-4 bg-void/50 leading-relaxed scrollbar-thin">
        {/* Line numbers */}
        <div className="flex flex-col text-right text-text-muted select-none pr-4 border-r border-default/20">
          {lines.map((_, i) => (
            <span key={i} className="min-w-[20px]">
              {i + 1}
            </span>
          ))}
        </div>

        {/* Code Content */}
        <pre className="flex flex-col pl-4 text-text-primary overflow-x-visible whitespace-pre text-left">
          {lines.map((line, i) => (
            <span key={i} className="block">
              {line || " "}
            </span>
          ))}
        </pre>
      </div>
    </div>
  );
}
