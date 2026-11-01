# OCR Similarity Analysis Page + Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate per-batch page that analyzes OCR similarity (how wrong wrong results are, and whether similarity remains) plus CSV/Excel export for the OCR report page data.

**Architecture:** A new pure-analysis service computes per-engine and per-batch similarity statistics from existing `EngineJudgment` rows. A new view renders a dedicated report template with summary cards, tables, and Chart.js visualizations. Export functions reuse the same data assembly to produce CSV/XLSX files matching the OCR report columns.

**Tech Stack:** Django templates, Chart.js (already loaded in base), openpyxl for Excel export, existing `testing` app models.

**Spec:** `docs/superpowers/specs/2026-10-31-anpr-testing-webapp-design.md`

## Global Constraints

- Keep changes inside the existing `testing` Django app; no new dependencies.
- Reuse existing `EngineJudgment` similarity fields (`sim_nospasi`, `sim_spasi`, `ratio_nospasi`, `ratio_spasi`) and existing metrics utilities.
- Follow existing URL pattern `/batch/<id>/report/ocr/...`.
- All report numbers must come from stored judgments; missing data yields `N/A`, never misleading zeros.
- Add tests for the new service and view; keep full suite green.

## Review Focus

- Engine with all `BENAR` judgments should show 100% exact match and mean similarity 100.
- Batch with no judgments should render gracefully (cards show N/A, tables empty, charts hidden).
- Export CSV and XLSX must contain the same columns and row count as the OCR report page table.
- Similarity bins must be inclusive on the lower bound and exclusive on the upper bound (e.g. 75-100 means >=75 and <100, with 100 in its own bucket).

---

### Task 1: Build similarity analysis service

**Files:**
- Create: `testing/services/ocr_analysis.py`
- Test: `testing/tests/test_ocr_analysis.py`

**Interfaces:**
- Consumes: `TestBatch`, `EngineJudgment` rows via `annotation__photo__batch`.
- Produces: `analyze_ocr_similarity(batch) -> dict` with keys `engines`, `overall`, `bins`, `wrong_but_similar`, `rows`.

- [ ] **Step 1: Write failing test**

```python
def test_analyze_empty_batch():
    batch = TestBatch.objects.create(name='E')
    result = ocr_analysis.analyze_ocr_similarity(batch)
    assert result['engines'] == {}
    assert result['overall']['n_judged'] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd webapp && python manage.py test testing.tests.test_ocr_analysis -v 2`
Expected: FAIL — `analyze_ocr_similarity` not defined.

- [ ] **Step 3: Implement `analyze_ocr_similarity(batch)`**

Compute per-engine:
- `n`, `exact`, `mean_sim_nospasi`, `mean_sim_spasi`, `mean_ratio_nospasi`, `mean_ratio_spasi`, `wrong_count`.
- Similarity bins: `[0,25)`, `[25,50)`, `[50,75)`, `[75,100)`, `100` for both nospasi/spasi.
- `wrong_but_similar`: count of judgments with `ok='SALAH'` and `sim_nospasi >= 80`.
- Overall: totals across engines.

- [ ] **Step 4: Run tests to verify it passes**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add testing/services/ocr_analysis.py testing/tests/test_ocr_analysis.py
git commit -m "feat: OCR similarity analysis service"
```

---

### Task 2: Build analysis report page

**Files:**
- Create: `testing/templates/testing/ocr_analysis.html`
- Modify: `testing/views.py`, `testing/urls.py`
- Test: `testing/tests/test_views_ocr_analysis.py`

**Interfaces:**
- Consumes: `ocr_analysis.analyze_ocr_similarity`.
- Produces: `batch_ocr_analysis_view(request, batch_id)` rendered with context `batch`, `analysis`, `chart_data_json`.

- [ ] **Step 1: Write failing view test**

```python
def test_analysis_page_renders_for_batch(self):
    resp = self.client.get(f'/batch/{self.batch.id}/report/ocr/analysis/')
    self.assertEqual(resp.status_code, 200)
    self.assertIn('Analisis Similarity OCR', resp.content.decode())
```

- [ ] **Step 2: Add URL path in `testing/urls.py`**

`path('batch/<int:batch_id>/report/ocr/analysis/', views.batch_ocr_analysis_view, name='batch_ocr_analysis')`

- [ ] **Step 3: Implement view `batch_ocr_analysis_view`**

Call `ocr_analysis.analyze_ocr_similarity(batch)` and serialize chart data as JSON.

- [ ] **Step 4: Create `ocr_analysis.html` template**

Include:
- Summary cards: total judgments, exact match rate, mean similarity, wrong-but-similar count.
- Per-engine table: n, exact %, mean sim %, mean ratio, wrong count.
- Similarity distribution table: counts per bin per engine.
- Chart.js bar chart for bin distribution (categorical, one dataset per engine).
- Section explaining why similarity matters (1-2 sentences).

- [ ] **Step 5: Run tests**

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add testing/templates/testing/ocr_analysis.html testing/views.py testing/urls.py testing/tests/test_views_ocr_analysis.py
git commit -m "feat: OCR similarity analysis page"
```

---

### Task 3: Add OCR report export (CSV/XLSX)

**Files:**
- Modify: `testing/services/export.py`, `testing/urls.py`, `testing/views.py`
- Test: `testing/tests/test_export.py` (extend existing)

**Interfaces:**
- Consumes: `TestBatch` and `EngineJudgment` rows.
- Produces: `export_ocr_rows(batch) -> list`, `export_ocr_csv(batch) -> str`, `export_ocr_xlsx(batch) -> bytes`.

- [ ] **Step 1: Write failing export test**

```python
def test_ocr_csv_has_engine_columns_and_similarity(self):
    rows = export.export_ocr_rows(self.batch)
    header = rows[0]
    self.assertIn('sim_tanpa_spasi_easyocr', header)
    self.assertIn('ratio_tanpa_spasi_easyocr', header)
```

- [ ] **Step 2: Implement `export_ocr_rows` / `export_ocr_csv` / `export_ocr_xlsx`**

Columns: per-photo metadata + per-engine columns: `bacaan`, `nilai`, `slh_karakter`, `catatan`, `sim_tanpa_spasi`, `sim_dengan_spasi`, `ratio_tanpa_spasi`, `ratio_dengan_spasi`.

- [ ] **Step 3: Add URL paths and views**

`/batch/<id>/report/ocr/export.csv` and `/batch/<id>/report/ocr/export.xlsx`.

- [ ] **Step 4: Run tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add testing/services/export.py testing/urls.py testing/views.py testing/tests/test_export.py
git commit -m "feat: OCR report CSV/XLSX export"
```

---

### Task 4: Wire export buttons into OCR report page

**Files:**
- Modify: `testing/templates/testing/ocr_report.html`

- [ ] **Step 1: Add export buttons next to the "← Batch" button**

Two links: `Export CSV` and `Export Excel` pointing to the new URL endpoints.

- [ ] **Step 2: Verify buttons appear and links resolve**

Run report view test and check rendered HTML contains the new URLs.

- [ ] **Step 3: Commit**

```bash
git add testing/templates/testing/ocr_report.html
git commit -m "ui: export buttons on OCR report page"
```

---

### Task 5: Run full test suite and verify UI

- [ ] **Step 1: Run full suite**

```bash
cd webapp && python manage.py test testing -v 2
```
Expected: all tests pass (86 + new tests).

- [ ] **Step 2: Open `/batch/6/report/ocr/analysis/` in browser and screenshot**

Verify cards, tables, and chart render with real data.

- [ ] **Step 3: Commit final fixes**

---

## Self-Review

- Spec coverage: similarity analysis (§7.3 OCR metrics) covered by Task 1-2; export covered by Task 3-4.
- All steps are single, checkable actions.
- Types consistent: `analyze_ocr_similarity(batch) -> dict` consumed by view and export.
- Review focus items each get a test or render guard.
- Plan length proportional to spec; no over-transcription.
