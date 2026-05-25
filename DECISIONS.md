# 🌳 Neural Forest v4.0 — وثيقة القرارات النهائية

**التاريخ:** 25 مايو 2026
**المهندس:** Naoufel
**المرحلة:** Phase 4 — Proof of Concept

---

## 🎯 الرؤية الاستراتيجية

### الهدف الحالي (PoC للممول)
بناء سلسلة نماذج (50M → 125M → 500M) تُثبت أن المعمارية scaleable.

### الهدف النهائي (مع التمويل)
نموذج 30B params يعمل محلياً على 16GB VRAM.

### الفكرة المميزة
نموذج محلي يتذكر المستخدم — **"ChatGPT يعرفه الجميع وهو لا يعرف أحداً، Forest يعرفك"**.

---

## 📊 الأرقام الواقعية للنجاح

| الهدف | الحد الأدنى | المتوقع الواقعي | للأحجام الكبيرة (3B+) |
|---|---:|---:|---:|
| توفير VRAM | 50% | 55-65% | 70-80% |
| Zero Forgetting | 90% | 95-97% | 95-97% |
| توفير FLOPs | 50% | 55-65% | 60-70% |
| Memory + Time | 100% | 100% | 100% |

**ملاحظة:** الأرقام أعلاه واقعية وقابلة للإثبات تجريبياً. تجنبنا الادعاءات المبالغ فيها.

---

## 🏗️ القرارات المعمارية النهائية

### 1. Shared Spine + Specialty Zones (Mixtral-inspired)
- **Spine:** 6 طبقات Attention مشتركة (~95M params) — دائماً في VRAM
- **Zones:** 10 مناطق Multi-layer FFN/GLU متخصصة — تُحمَّل من RAM عند الطلب
- **KV Cache:** في الـ Spine فقط (يحل Context Fragmentation)

### 2. الـ 10 مناطق النهائية
| # | Zone | المحتوى |
|---|------|---------|
| 0 | **Skip Zone** | الكلمات الوظيفية وعلامات الترقيم (لا حوسبة) |
| 1 | الرياضيات | جبر، حساب، تفاضل، إحصاء |
| 2 | البرمجة | Python, JS, C++, Rust, خوارزميات |
| 3 | الفيزياء والكيمياء | علوم صلبة، مواد، طاقة |
| 4 | الأحياء والطب | طب، أحياء، تشريح، صيدلة |
| 5 | التاريخ والجغرافيا | (مدمجة) حضارات، خرائط، مناخ |
| 6 | الفلسفة وعلم الاجتماع | فلسفة، اجتماع، علم نفس |
| 7 | الاقتصاد والمال | تجارة، بورصة، تحليل |
| 8 | الفنون والتصميم | تصميم، فنون بصرية |
| 9 | الأدب والكتابة | شعر، رواية، نقد |
| 10 | الترجمة | عربي/إنجليزي/فرنسي |

**ملاحظة:** Zone 0 (Skip) هو زون "وهمي" - لا يحوي معاملات، فقط مسار مباشر للـ output.

### 3. شكل كل Zone (Multi-layer FFN / GLU)
```python
class Zone(nn.Module):
    """
    لا attention - الـ Spine يهتم بالسياق
    GLU activation (SwiGLU)
    2-3 طبقات FFN متتالية
    """
    layers:
        Linear(embed_dim, hidden) + SiLU
        Linear(hidden, hidden) + SiLU  # طبقة وسط
        Linear(hidden, embed_dim)
```

### 4. Router الذكي
- **التدريب:** Top-1 hard routing (Gumbel-Softmax)
- **النشر:** Dynamic Top-1/Top-2 (إذا الاحتمالات قريبة)
- **Skip Detection:** إذا أعلى احتمال = Zone 0 → bypass كل الـ FFN

### 5. HOT/WARM/COLD Memory Tiers
```
🔥 HOT (VRAM دائماً):
   Embedding + Spine (6 layers) + Router + LM Head
   ~95M params في 50M model
   
🟡 WARM (RAM):
   10 Specialty Zones
   تنتقل لـ VRAM في 0.5-3ms عند الطلب
   
❄️ COLD (SSD):
   Checkpoints + Backups
```

### 6. Memory Module (خارج العصبي)
- **SQLite database** على جهاز المستخدم
- يحوي: المحادثات، المشاريع، التفضيلات
- يُحقن في System Prompt كنص قبل forward pass
- **يتذكر بدون إعادة تدريب**

### 7. Temporal Service
- خدمة خارجية تقرأ ساعة الجهاز
- تُحقن "اليوم 25 مايو 2026" في System Prompt
- لا تستهلك معاملات عصبية

---

## 📈 خطة الـ Scale Ladder

| المرحلة | الحجم | tokens | الوقت | التكلفة | الهدف |
|---------|------:|-------:|-------|--------|--------|
| **4.1** | 50M | 0.5B | 15 دقيقة | $1 | Smoke test |
| **4.2** | 125M | 2.5B | 1.5 ساعة | $2 | إثبات المعمارية |
| **4.3** | 500M | 10B | 1 يوم | $35 | Scaling law |
| **مجموع PoC** | | | **~3 أيام** | **~$40** | عرض الممول |

**ملاحظة:** هذه أرقام التدريب فقط. أضف 20% للـ debugging والـ tuning.

---

## 💾 البيانات (للتدريب)

### المرحلة 4.1-4.2 (Smoke test + Proof):
- **FineWeb-Edu** (English) - 5B tokens sample
- **FineWeb-Edu-Ar** (Arabic) - 5B tokens sample

### المرحلة 4.3 (Scaling):
| Domain | Dataset | الحجم |
|--------|---------|------:|
| اللغة العامة | FineWeb-Edu | 3B |
| العربية | FineWeb-Edu-Ar | 2B |
| الرياضيات | OpenWebMath + FineMath | 1B |
| البرمجة | The Stack v2 (subset) | 2B |
| العلوم | arXiv abstracts + Wiki | 1B |
| التاريخ والجغرافيا | Wikipedia | 1B |

---

## 🛠️ القرارات التقنية

| القرار | الاختيار |
|--------|----------|
| **اللغة** | Python 3.11+ |
| **الإطار** | PyTorch 2.5+ |
| **التدريب** | bf16 + Adam |
| **النشر** | 4-bit (GPTQ) |
| **البنية التحتية** | Colab Pro A100 أو Thunder Compute |
| **التخزين** | Google Drive للـ checkpoints |
| **المراقبة** | Weights & Biases (مجاني) |
| **الكود** | GitHub (Public) |
| **استراتيجية البرمجة** | Naoufel يكتب prompts → Claude Code ينفذ |

---

## 📁 بنية المشروع

```
neural-forest/
├── README.md
├── DECISIONS.md              # هذا الملف
├── ARCHITECTURE.md           # شرح تقني عميق
├── LICENSE
├── .gitignore
├── pyproject.toml
├── requirements.txt
│
├── forest/                   # الكود الأساسي
│   ├── __init__.py
│   ├── config.py             # ForestConfig
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── spine.py          # Shared Spine
│   │   ├── router.py         # Router + Skip detection
│   │   ├── zone.py           # Specialty Zone (FFN/GLU)
│   │   └── model.py          # NeuralForest (main)
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── sqlite_store.py
│   │   └── temporal.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── trainer.py
│   │   └── losses.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── vram_monitor.py   # لإثبات التوفير
│       └── benchmarks.py
│
├── tests/
│   ├── test_spine.py
│   ├── test_router.py
│   ├── test_zone.py
│   └── test_model.py
│
├── scripts/
│   ├── train_50m.py
│   ├── train_125m.py
│   ├── train_500m.py
│   └── benchmark.py
│
└── notebooks/
    └── colab_training.ipynb
```

---

## ✅ Definition of Done لكل مرحلة

### المرحلة 4.1 (50M Smoke Test):
- [ ] forest/ tree built and importable
- [ ] Model forward pass works
- [ ] Backward pass produces gradients in all components
- [ ] Router distributes tokens across zones
- [ ] Trains for 100 steps without crashing
- [ ] Loss decreases from random initialization

### المرحلة 4.2 (125M Proof):
- [ ] Full training run on 2.5B tokens
- [ ] VRAM usage measured and documented
- [ ] Zone specialization measured (entropy < 1.5)
- [ ] Comparison with Dense baseline shows savings
- [ ] Inference benchmark on consumer GPU

### المرحلة 4.3 (500M Scale):
- [ ] Scaling curve plotted (50M, 125M, 500M)
- [ ] MMLU score measured
- [ ] HumanEval score measured
- [ ] Memory module integration demo
- [ ] Investor presentation ready

---

**الحالة:** ✅ متفق عليه، جاهز للتنفيذ
**التوقيع:** Naoufel + Claude (Engineering Partner)
