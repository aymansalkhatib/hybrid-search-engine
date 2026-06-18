# البحث بالفهرس المقلوب فقط — الاسترجاع البولياني (Boolean Retrieval)

> بحث يطابق المستندات باستخدام **الفهرس المقلوب وحده** (قوائم الـ postings)، **دون أي تهديف**
> (لا TF‑IDF ولا BM25). هو أقدم نماذج الاسترجاع وأكثرها مباشرةً، وقيمته تعليمية: يُظهر ما يفعله
> الـ inverted index فعلياً.

---

## 1. ما هو؟ ولماذا منطقي؟

الفهرس المقلوب يربط كل **مصطلح → قائمة المستندات التي تحتويه** (postings). البحث البولياني يستعمل هذه
القوائم مباشرةً عبر جبر المجموعات:

- **AND** — المستندات التي تحتوي **كل** مصطلحات الاستعلام (تقاطع قوائم الـ postings).
- **OR** — المستندات التي تحتوي **أي** مصطلح (اتحاد القوائم).

لا توجد «درجة تشابه» (relevance score). الناتج **مجموعة** من المستندات. نرتّبها ترتيباً
حتمياً بسيطاً حسب **عدد المصطلحات المطابقة** (coordination) ثم حسب `doc_id` — وهذا **عدّ** لا
خوارزمية تهديف. (في وضع AND يكون عدد المطابقات ثابتاً = كل المصطلحات، فيؤول الترتيب إلى `doc_id`.)

هذا يكمّل — ولا يستبدل — البحث المعتمد على النماذج (TF‑IDF/BM25/Embeddings/Hybrid)، ويتيح في التقرير
مقارنة «المطابقة فقط» مقابل «المطابقة + التهديف».

## 2. أين يعيش في المعمارية؟ (نفس نمط `representation /rank`)

نفس مبدأ الفصل الذي اعتمدناه في خدمة البحث: **المالك يحسب حيث تقيم البيانات، وخدمة البحث تنسّق**.

| الطبقة | المسؤولية |
|--------|-----------|
| **`indexing-service`** (مالك الفهرس) | بدائية `POST /match` (داخلية): يطبّع الاستعلام بخيارات بناء الفهرس نفسها (عبر `preprocessing-service`)، ويحسب تقاطع/اتحاد الـ postings داخل الذاكرة بمؤشرات الأعداد الصحيحة، ويعيد المستندات المطابقة + `df` كل مصطلح. **لا تهديف.** |
| **`retrieval-service`** (دماغ المطابقة) | نقطة `POST /boolean`: تستدعي `/match`، تجلب **النص الأصلي** بالـ ID من الـ doc-store، وتعيد رداً بنفس شكل البحث المُرتَّب. |
| **`api-gateway`** | `POST /retrieval/boolean` (الباب الخارجي الوحيد). البدائية `/match` تبقى **داخلية** (retrieval→indexing) ولا تُنشر عبر البوابة — تماماً كـ `/rank` و`/score`. |
| **الواجهة (Search tab)** | زر «Boolean · index» + مفتاح **AND/OR**؛ يرسل إلى `retrieval/boolean` ويعرض المستندات والنص الأصلي وعدد المصطلحات المطابقة. |

```
UI ─► gateway ─/retrieval/boolean─► retrieval-service ─/match─► indexing-service (postings, AND/OR)
                                          │
                                          └─/docs (بالـ ID)──► doc-store (النص الأصلي)
```

لماذا هكذا؟ تطبيع الاستعلام يحتاج **خيارات بناء الفهرس** التي يملكها `indexing-service` (كما يملك
`representation-service` خيارات بناء النموذج لـ `/rank`)، وجبر المجموعات على قوائم postings ضخمة
(قد تبلغ مئات الآلاف لمصطلح شائع) أسرع وأنظف أن يُنفَّذ **حيث الفهرس في الذاكرة** بدل نقل القوائم عبر
الشبكة. تبقى `retrieval-service` خفيفة (بلا artifacts) وتملك **استراتيجية** البحث فقط (المطابقة + العرض).

## 3. العقود (في `shared/contracts`)

- بدائية الفهرسة (`indexing.py`): `BooleanMatchRequest{dataset, query, operator, top_k}` →
  `BooleanMatchResponse{dataset_id, operator, terms:[{term, df}], total_matched, hits:[{doc_id, matched}]}`.
- واجهة البحث (`retrieval.py`): `BooleanSearchRequest{dataset, query, operator, top_k, with_text}` →
  `BooleanSearchResponse{…, hits:[{rank, doc_id, matched, text}]}`.

## 4. أمثلة (عبر البوابة)

```bash
# AND — مستندات تحتوي كل المصطلحات
curl -X POST http://localhost:8000/retrieval/boolean -H "Content-Type: application/json" \
  -d '{"dataset":"beir/quora/test","query":"learn python quickly","operator":"and","top_k":5}'

# OR — مستندات تحتوي أي مصطلح (مرتّبة بعدد المطابقات)
curl -X POST http://localhost:8000/retrieval/boolean -H "Content-Type: application/json" \
  -d '{"dataset":"beir/quora/test","query":"learn python quickly","operator":"or","top_k":5}'
```

> ملاحظة: يتطلب أن يكون **الفهرس المقلوب مبنيّاً** للداتاسيت (عبر `indexing` POST /build). إن لم يُبنَ
> ترجع 404 «build the index first». الاستعلام يُطبَّع بنفس خيارات بناء الفهرس ليبقى في فضاء المصطلحات ذاته.
