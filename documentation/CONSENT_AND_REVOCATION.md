# Contributor Consent, Privacy, and Revocation Policy (SLAI-025)

This document defines how contributor consent is recorded, separated from personally identifiable information (PII), scoped by task, and revoked across the Sierra Leone AI Foundation datasets.

---

## 1. Separation of Identifiable Consent Forms (Restricted Store)

- **No PII in Dataset JSONL Files:** Identifiable consent artefacts—including contributor full names, phone numbers, email addresses, national ID numbers, or physical/digital signatures—must **never** be committed to git or stored inside `.jsonl` dataset files.
- **Restricted Consent Store:** Signed paper or digital consent forms are stored in an encrypted, access-controlled external repository managed by the project's data governance lead.
- **Pseudonymous Linking (`consent_record_id`):** Dataset records store only an opaque `consent_record_id` (e.g., `"CONSENT-SL-2026-00042"`), `contributor_consent: bool`, and `consent_scope: list[str]`.
- **Automated PII Lint:** The dataset loader provides `scan_jsonl_for_pii(path)` which fails validation if email addresses or phone number patterns appear in any `.jsonl` file.

---

## 2. Granular Consent Scopes

Consent is never a blanket boolean. Every record where `contributor_consent=True` must declare explicit scopes in `metadata.consent_scope`:

| Scope | Meaning |
|---|---|
| `text_training` | Permitted for language model / translation text training |
| `speech_recognition` | Permitted for Automatic Speech Recognition (ASR) training and evaluation |
| `voice_synthesis` | **Separate, explicit opt-in** for Text-to-Speech (TTS) / voice synthesis |
| `public_release` | Permitted for redistribution in an open dataset release |
| `commercial_use` | Permitted for commercial applications |

> **Critical Rule for Speech Data:** Consent for `voice_synthesis` (TTS / voice cloning) is **strictly separate and opt-in**. It is **never** inferred from `speech_recognition` (ASR) consent or general audio recording consent. `check_tts_recording_quality()` rejects any recording lacking `"voice_synthesis"` in `consent_scope`.

---

## 3. Contributor Consent Withdrawal (Revocation Process)

Contributors may withdraw their consent at any time without penalty:

1. **Submission of Withdrawal Request:** The contributor (or community liaison) submits a withdrawal request referencing their name, contact details, or `consent_record_id` to the data governance team.
2. **Lookup in Restricted Store:** The governance lead resolves the contributor's identity to their `consent_record_id` and associated `record_id` / `speaker_id` entries.
3. **Immediate Dataset Tombstoning & Removal:**
   - All matching records in `datasets/*/raw/`, `datasets/*/cleaned/`, `datasets/*/translations/`, `datasets/*/conversations/`, `datasets/*/audio/`, and `benchmarks/` have `metadata.consent_withdrawn = True` set immediately and are purged from active JSONL and audio directories in the next dataset version bump.
   - The dataset loader (`_is_eligible`) immediately blocks any record with `consent_withdrawn=True` (`skip_reason="consent_withdrawn"`).
4. **Honest Statement on Already-Trained Checkpoints:**
   - **What is removed:** All raw audio/text files, normalized/derived records, benchmark items, and inclusion in any future training runs or public dataset exports.
   - **What cannot be removed immediately:** Once a neural network checkpoint has been trained, individual training examples cannot be mathematically extracted or "unlearned" from the frozen model weights without retraining. If a contributor withdraws consent after a checkpoint was trained on their data, that checkpoint will be scheduled for deprecation and excluded from future releases once a replacement checkpoint is trained on the updated, post-revocation corpus.

---

## 4. Third-Party Copyrighted Material & Legal Review

- **Legal Review Requirement:** Formal legal review and written permission from rights holders are required before **commercial training** or **public release** of any corpus containing third-party copyrighted text (including Sierra Leonean textbooks, newspapers, broadcast transcripts, or scripture translations).
- Records with `licence="unknown"`, `training_permitted=False`, or `UNREVIEWED` status in `datasets/source_register.csv` are automatically blocked from model training and release exports.
