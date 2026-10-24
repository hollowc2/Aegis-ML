# Targeted-data provenance and licences

Source revisions were verified before data use on 2026-10-22. Downloads are
pinned in `training/data/targeted_sources.py`; the frozen evaluation manifest
records source-file SHA256 values.

- [BIPIA](https://github.com/microsoft/BIPIA/tree/a004b69ec0dd446e0afd461d98cb5e96e120a5d0):
  repository and text attacks are MIT. Only `benchmark/email/{train,test}.jsonl`
  (OpenAI Evals invoices, MIT) and `benchmark/text_attack_{train,test}.json`
  are used. JSONL contexts have `context`, `question`, `ideal`; attack JSON
  maps attack families to lists of strings. Training uses train attacks with
  train contexts; evaluation uses test attacks with test contexts. These are
  detector examples, not a reproduction of the BIPIA agent-task benchmark.
  Code contexts (Stack Exchange) and tables (WikiTableQuestions) are CC BY-SA
  4.0 and excluded. WebQA/NewsQA and summarization/XSum require separate data
  acquisition/licence review and are excluded. See the upstream
  [licence exceptions](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/LICENSE)
  and [format description](https://github.com/microsoft/BIPIA/blob/a004b69ec0dd446e0afd461d98cb5e96e120a5d0/benchmark/README.md).
- [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent/tree/f19c9f2c79a41046eb13c03c51a24c567a8ffa07):
  MIT, copyright 2023 Qiusi Zhan. JSON arrays in
  `data/test_cases_{dh,ds}_base.json` contain `Tool Response` and
  `Attacker Instruction`. All 510 direct-harm and 544 data-stealing cases
  are reserved for evaluation; data-stealing maps to `data_exfiltration`.
  There is no official training split. Enhanced cases reuse base attacks,
  so they are not separately counted. No InjecAgent-derived training rows
  are generated. See the upstream [licence](https://github.com/uiuc-kang-lab/InjecAgent/blob/f19c9f2c79a41046eb13c03c51a24c567a8ffa07/LICENCE).
- [NotInject](https://huggingface.co/datasets/leolee99/NotInject/tree/847ae76cf8fea5ed325429e569ae8cfef022d2e0):
  MIT according to the dataset card. Parquet fields are `prompt`, `word_list`,
  `category`. `NotInject_one`, `NotInject_two`, `NotInject_three` each have
  113 benign examples. The additional train/validation/test files duplicate
  these respective difficulty groups; they are not independent split families.
  All 339 cases are reserved for evaluation to avoid leaking trigger families.
  No NotInject cases are used for training. Its general benign trigger-word
  FPR is reported separately from the authored security-specific benign FPR.
- Original authored carriers, attack scenarios and hard negatives are under
  this repository's MIT licence. Long evaluation carriers repeat original
  archive prose; they do not establish generalization to diverse long documents.
  Indirect and buried evaluation cases are paired versions of the same 100
  instructions. The authored security examples are not an independent human
  audit and describe only the sampled styles.
- AgentDojo was not used: optional source omitted to keep the new data's
  licence and split boundaries narrow and reproducible.

No new non-commercial sources are added. Existing toxic-chat (CC BY-NC 4.0),
Dolly (CC BY-SA 3.0) and other legacy sources retain their original terms;
this work does not remove those existing restrictions. Copies of the new
MIT notices are in `docs/data_licenses/`.

## Round 2 additions (2026-10-23)

Only **benign** material from these sources is used, in the frozen
false-positive set `data/round2/final_fp_eval.csv`. Nothing from them is
used for training.

- [LLMail-Inject](https://huggingface.co/datasets/microsoft/llmail-inject-challenge/tree/1063bdf01ec8762b812d5e06ee768a06faa5a6f7):
  MIT (dataset card; repository licence at commit
  `3da2399839b285e129888161fcc8cb03169d1b43`, copied to
  `docs/data_licenses/LLMail-Inject.txt`). Used: `data/emails_for_fp_tests.json`
  (160 unique benign emails). Challenge attack submissions are not used.
- [AgentDojo](https://github.com/ethz-spylab/agentdojo/tree/089ed468cf3ed0322acc66b0211f26d9d90dbf60):
  MIT (`docs/data_licenses/AgentDojo.txt`). Used: 41 environment records from
  the four v1 suites, with every injection vector set to its benign default
  value. Injection tasks and attack strings are not used. The YAML is read as
  data; no downloaded code is executed.
- Authored benign cases (266 rows) were written by a separate agent instance
  under this repository's MIT licence.
