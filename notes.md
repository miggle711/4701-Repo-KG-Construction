# Possible Code Benchmarks
SWE-bench: Evaluates resolving real-world GitHub issues. Variants include SWE-bench Lite (300 issues), SWE-bench Verified (500 verified issues), SWE-bench-java Verified (91 Java issues), and Multi-SWE-bench
.
EvoCodeBench (Evolutionary Code Benchmark): Specifically designed for repository-level tasks, focusing on generating function bodies based on repository context
.
REPOKG-50: A curated benchmark for repository-level code generation consisting of 4,250 tasks across 50 Python projects with aligned static and dynamic graphs
.
CrossCodeEval: A diverse, multi-lingual benchmark (Python, Java, TypeScript, C#) focusing on cross-file code completion
.
DevEval: Aligned with real-world code repositories, featuring 117 repositories with manual annotations
.
RepoEval (and RepoEval-Updated): A collection of repository-level code completion tasks derived from GitHub repositories
.
CoderEval: A latest benchmark that mirrors real-world development using data from actual GitHub repositories
.
ComplexCodeEval: Evaluates performance in complex scenarios like API recommendations and test case generation
.
2. Code Migration, Evolution, and Environment Awareness
These benchmarks test how well models adapt to library version changes or specific software environments.
VersiCode: The first large-scale benchmark for version-controllable code generation, covering version-specific completion (VSCC) and version-aware migration (VACM) across 300+ libraries
.
VersiBCB: Focused on Environment-Aware Code Generation (EACG), requiring code to be functionally correct and executable under arbitrary software configurations
.
LibEvolutionEval: A benchmark for version-specific code generation highlighting the impact of rapidly-evolving public libraries
.
GitChameleon: Designed to unmask version-switching capabilities of code generation models
.
CodeUpdateArena: Benchmarking knowledge editing specifically on API updates
.
3. Program Repair, Debugging, and Robustness
DebugBench: A comprehensive benchmark for evaluating debugging capability across syntax, reference, and logic errors
.
RepoBugs & RepoDebug: Focus on repository-level automatic program repair and multi-task/multi-language debugging
.
GHRB (GitHub Recent Bugs): Collects real-world Java bugs from highly-starred repositories
.
Breakpoint: Stress-tests systems-level reasoning in LLM agents through adversarial errors
.
ReCode: A robustness evaluation framework that applies natural transformations to docstrings, variable names, and syntax
.
Refactory: Used for evaluating program repair on student-submitted programming assignments
.
4. Code Reasoning, Understanding, and Formal Verification
CoRe: Human-verified benchmark evaluating code reasoning through fundamental static analysis tasks like data and control dependency
.
CRUXEval (and CRUXEval-X): Assesses code reasoning, understanding, and execution via input/output prediction
.
LeetCode-C-Spec-200: Designed for generating memory-aware formal function specifications using separation logic predicates
.
VIFBENCH: Rigorously evaluates LLM adherence to abstract and concrete instruction-following constraints
.
LONGCODEU: Tests multi-aspect comprehension of long-context codebases
.
5. Standard and Function-Level Generation
These are often used as baselines for general coding proficiency.
HumanEval (and variants: +, ET, Pro, Comm, FIM): The seminal hand-written Python benchmark for function-level synthesis
.
MBPP (and variants: +, ET, Pro): Crowd-sourced Python problems covering entry-level programming fundamentals
.
CodeNet: A large-scale dataset covering 55 programming languages for various tasks
.
BigCodeBench: Evaluates handling of complex instructions and diverse library function calls
.
DS-1000: A natural and reliable benchmark for data science-related code generation
.
6. Security, Vulnerability, and Other Specialized Tasks
CVE-Bench & SEC-Bench: Benchmarks for evaluating repair of real-world vulnerabilities and security engineering tasks like PoC generation
.
SECCODEPLT: A unified platform for Evaluating secure code generation across 44 CWE categories
.
SWT-bench & TestGenEval: Focused on automated test case generation and issue-reproducing tests
.
GSO: Challenging software optimization tasks requiring patches that improve runtime efficiency
.
DI-BENCH: Evaluates the dependency inference ability of LLMs at scale
.
CodeHalu & Collu-Bench: Specifically designed to predict and detect code-related hallucinations
.


## Todo
![alt text](image.png)
### Near Term (to be written down explicitly in the g drive by Friday 11:55pm OZ time innit)
- Determine Datasets to use for KG Construction (Wan Mun)
- Determine Baseline methods to compare our method against (Joe SIahhhh)
    - RAG
    - Maybe another KG augmented method (chicken jockey)
- CODE (Miguel)
    - LLM enrichment (for docs)
    - add more DATA (method signatures)
    - Make KGs using Docs?
- Filtering Methodolgy (Sheryl)

use swebench but filter it ;)
use a dataset from a different paper
do other datasets have a "golden patch"??
- source Aaron

### W8
project problem
initial scope
early design/data collection plan
key risks
OSHA and Ethics

### W12 draft of paper
lit review
methodology
expected outcomes 
preliminary results (maybe)