# External-Data Rights Audit Template (SLAI-031)

Before any third-party or externally published text/speech dataset is ingested into the Sierra Leone AI Foundation training or benchmark pipelines, a formal **Source Rights Audit** must be completed and recorded in [`datasets/source_register.csv`](../datasets/source_register.csv).

---

## Core Principle

> **A repository-level or Hugging Face dataset card licence tag (such as `Apache-2.0` or `MIT`) does NOT automatically clear the copyright of the underlying books, articles, recordings, or publications contained within that dataset.**

For example:
- `Jaward/Krio-Corpus` on Hugging Face carries an `apache-2.0` tag on its dataset card, but its `meta_data` field cites published Sierra Leone junior secondary school textbooks (*Lɛ Wi Lan Krio JSS 1* by Pearce & Pearce, 1996; *Lɛ Wi Lan Krio JSS 3* by Cline-Cole, 1996, published by Lekon Publishing, Freetown), the TypeCraft Krio Corpus (*Beatrice Owusua Nyampong*, 2015), and commercial YouTube animations.
- Unless written permission from the underlying copyright holder (or proof of public-domain status, such as the Universal Declaration of Human Rights) is obtained and archived, those records **cannot** be redistributed or used for commercial model training.

---

## Loader Enforcement Rule

The dataset loader (`src/sierraleone_ai/datasets/loader.py`) reads `datasets/source_register.csv` on load:
- Any source listed in `datasets/source_register.csv` with `decision` set to `UNREVIEWED` or `REJECTED` is **automatically blocked** from training loads (`skip_reason = "unreviewed_source_rights"`).
- Only sources whose `decision` is explicitly set to `APPROVED` (with a named `reviewer`, `review_date`, and `evidence_or_permission_link`) are eligible.

---

## Per-Source Audit Checklist Template

Copy and complete this template for each external data source before changing its `decision` in `datasets/source_register.csv`:

```markdown
### Source Rights Audit Record

- **Source Identifier (`source_name`):** 
- **Canonical URL / Bibliographic Citation:** 
- **Primary Rights Holder(s):** 
- **Wrapper / Repository Licence Claimed:** 
- **Underlying Works Inspected:** 
  - [ ] Checked per-record metadata / citations for third-party books, news articles, or scripts
  - [ ] Verified whether underlying works are in the public domain or under copyright
- **Permissions Verified:**
  - **Redistribution Permitted?** (yes / no / restricted): 
  - **Model Training Permitted?** (yes / no / non-commercial only): 
  - **Commercial Use Permitted?** (yes / no): 
- **Written Permission / Evidence Link:** 
- **Contributor Informed Consent Verified?** (yes / no / N/A for public domain): 
- **Legal Review Completed By:** 
- **Audit Date (YYYY-MM-DD):** 
- **Final Decision (`UNREVIEWED` | `APPROVED` | `REJECTED`):** 
```
