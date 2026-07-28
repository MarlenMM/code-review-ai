% experimentreport.cls
% LaTeX class file for experiment reports

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{experimentreport}[2026/06/11 Course Experiment Report Class]

\LoadClass[12pt,a4paper]{ctexart}

% Page layout
\RequirePackage{geometry}
\geometry{
left=2.6cm,
right=2.6cm,
top=2.2cm,
bottom=2.2cm
}

% Common packages
\RequirePackage{array}
\RequirePackage{setspace}
\RequirePackage{tabularx}
\RequirePackage{titlesec}

% Basic formatting
\pagestyle{empty}
\setlength{\parindent}{2em}
\setstretch{1.35}
\renewcommand{\arraystretch}{1.45}

\newcommand{\placeholder}[2]{
\begin{center}
\fbox{
\parbox[t][#1][t]{0.9\textwidth}{
#2
}
}
\end{center}
}

\usepackage[normalem]{ulem}
\newcommand{\hintholder}[1]{%
\uline{\hfill\textnormal{\small #1}\hfill}
}

% Section title formatting
\titleformat{\section}
{\zihao{4}\heiti}
{\chinese{section}、}
{0em}
{}

% Default values ​​for information fields
\newcommand{\coursename}{Practice of Basic Software Development Technologies}
\newcommand{\reportname}{Intelligent Software Engineering Experiment Report}
\newcommand{\experimenttitle}{Experiment \_\_：\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_}

\newcommand{\studentname}{}
\newcommand{\department}{}
\newcommand{\studentid}{}
\newcommand{\courseteacher}{}
\newcommand{\instructor}{}
\newcommand{\lablocation}{}
\newcommand{\labtime}{}
\newcommand{\performance}{}
\newcommand{\reportscore}{}
\newcommand{\totalscore}{}
\newcommand{\operationscore}{}

% User-editable commands
\newcommand{\setcoursename}[1]{\renewcommand{\coursename}{#1}}
\newcommand{\setreportname}[1]{\renewcommand{\reportname}{#1}}
\newcommand{\setexperimenttitle}[1]{\renewcommand{\experimenttitle}{#1}}

\newcommand{\setstudentname}[1]{\renewcommand{\studentname}{#1}}
\newcommand{\setdepartment}[1]{\renewcommand{\department}{#1}}
\newcommand{\setstudentid}[1]{\renewcommand{\studentid}{#1}}
\newcommand{\setcourseteacher}[1]{\renewcommand{\courseteacher}{#1}}
\newcommand{\setinstructor}[1]{\renewcommand{\instructor}{#1}}
\newcommand{\setlablocation}[1]{\renewcommand{\lablocation}{#1}}
\newcommand{\setlabtime}[1]{\renewcommand{\labtime}{#1}}
\newcommand{\setperformance}[1]{\renewcommand{\performance}{#1}}
\newcommand{\setreportscore}[1]{\renewcommand{\reportscore}{#1}}
\newcommand{\settotalscore}[1]{\renewcommand{\totalscore}{#1}}
\newcommand{\setoperationscore}[1]{\renewcommand{\operationscore}{#1}}

% Cover page title
\newcommand{\makereporttitle}{
\begin{center}
{\zihao{2}\heiti \coursename}

\vspace{0.6cm}

{\zihao{2}\heiti \reportname}

\vspace{0.6cm}

{\zihao{3}\heiti \experimenttitle}
\end{center}

\vspace{0.4cm}
}

% Information table
\newcommand{\makereportinfo}{
\begin{center}
\begin{tabular}{|p{2.8cm}|p{4.2cm}|p{2.8cm}|p{4.2cm}|}
\hline
Name & \studentname & Department & \department \\ \hline
Student ID & \studentid & Course Teacher & \courseteacher \\ \hline
Instructor & \instructor & Lab Location & \lablocation \\ \hline
Lab Time & \labtime &  &  \\ \hline
Lab Score & \reportscore & Discussion Score & \performance \\ \hline
Report Score & \operationscore & Total Experiment Score & \totalscore \\ \hline
\end{tabular}
\end{center}

\vspace{0.8cm}
}

% Full report header
\newcommand{\makereporthead}{
\makereporttitle
\makereportinfo
}

% Standard section commands
\newcommand{\experimentpurpose}[1]{
\section{Experiment Objectives}
#1
}

\newcommand{\experimentcontent}[1]{
\section{Experiment Content}
#1
}

\newcommand{\experimentresult}[1]{
\section{Experimental Results}
Requirement: Describe the results obtained from the experiment. Basic content should include:

\vspace{0.3cm}

#1
}

\newcommand{\experimentproblems}[1]{
\section{Summary of Problems Encountered}
Requirement: Primarily explain how the problems encountered during the experiment were resolved.

\vspace{0.3cm}

#1
}

\newcommand{\experimentexperience}[1]{
\section{Reflections on the Experiment}
#1
}

% Instructor's comments
\newcommand{\teachercomment}{
\vspace{1cm}

\noindent{\heiti Instructor's Comments:}

\vspace{4cm}

\noindent Date: \hspace{4cm}
}

\endinput