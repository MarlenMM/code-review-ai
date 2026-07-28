% ctexart.cls -- minimal local shim, NOT a CJK-typesetting implementation.
%
% reports/template/experimentreport.cls does \LoadClass[12pt,a4paper]{ctexart},
% but every report in this project is written entirely in English. Rather than
% install real CJK support (ctexart pulls in the full ctex bundle) for a
% report with no Chinese text, this defines only the handful of ctex-specific
% commands experimentreport.cls actually calls, on top of plain article.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ctexart}[2026/07/23 minimal ctexart shim for English-only reports]

\LoadClassWithOptions{article}

% ctex's "hao" (\C{号}) font-size system -- only the sizes
% experimentreport.cls actually uses (2, 3, 4) are mapped.
\newcommand{\zihao}[1]{%
  \ifnum#1=2
    \fontsize{22pt}{27pt}\selectfont
  \else\ifnum#1=3
    \fontsize{16pt}{20pt}\selectfont
  \else\ifnum#1=4
    \fontsize{14pt}{18pt}\selectfont
  \else
    \normalsize
  \fi\fi\fi
}

\newcommand{\heiti}{\bfseries}
\newcommand{\songti}{\normalfont}
\newcommand{\kaishu}{\normalfont}
\newcommand{\fangsong}{\normalfont}

% Section-counter-to-Chinese-numeral command; mapped to plain Arabic numerals
% since this report has no Chinese text.
\newcommand{\chinese}[1]{\arabic{#1}}

\endinput
