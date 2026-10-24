This .md file provides a full overview of our usage of LLMs. Claude Sonnet 5.5 Medium was used in formatting this file.

Guide for labeling as provided in MachineLearningUiO:

| Level | Label       | Meaning                                                                                    |
| ----- | ----------- | ------------------------------------------------------------------------------------------ |
| 0     | None        | No LLM assistance                                                                          |
| 1     | Debugging   | LLM helped identify a bug; fix implemented by you                                          |
| 2     | Snippet     | LLM provided a function, loop, or short block; you integrated and tested it                |
| 3     | Skeleton    | LLM generated the overall structure of a script/class; you filled in domain-specific logic |
| 4     | Substantial | LLM wrote the majority of a file; you adapted, tested, and commented it                    |

**Models used.**

**Text writing and editing.**

| Section      | Level | Notes |
| ------------ | ----- | ----- |
| Abstract     |       |       |
| Introduction |       |       |
| Methods      |       |       |
| Results      |       |       |
| Discussion   |       |       |
| Conclusion   |       |       |

**Code generation and assistance.**

| File / notebook | LLM level | Description                            |
| --------------- | --------- | -------------------------------------- |
| `__init__.py` | 1         | How to access modules from other files |
|                 'general_functions.py'|          2 |Helper class for passing arguments                                        |
