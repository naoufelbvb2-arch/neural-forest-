# 🚀 PROMPT 1: Initial Project Setup

> **اشرح لـ Claude Code:** هذا أول prompt في مشروع Neural Forest v4.0.
> الهدف من هذا الـ prompt: إنشاء **البنية الكاملة للمشروع** مع ملفات placeholder + بعض الكود الأساسي.
> لا نريد المعمارية الكاملة الآن — نريد الهيكل أولاً.

---

## انسخ هذا الـ Prompt لـ Claude Code:

```
أنا أبني نموذج لغوي اسمه Neural Forest v4.0 — معمارية MoE sparse للنشر المحلي.

سأشاركك ملف DECISIONS.md و README.md منفصلين. اقرأهما أولاً ثم نفّذ المهمة التالية.

═══════════════════════════════════════════════════════════════════
المهمة: إنشاء بنية المشروع الكاملة (الهيكل + placeholders)
═══════════════════════════════════════════════════════════════════

أنشئ المشروع في مجلد neural-forest/ بالبنية التالية:

neural-forest/
├── README.md                  ← (سأنسخه لك)
├── DECISIONS.md               ← (سأنسخه لك)
├── ARCHITECTURE.md            ← اتركه فارغ، سنملأه لاحقاً
├── LICENSE                    ← Apache 2.0
├── .gitignore                 ← Python + ML standards
├── pyproject.toml             ← أنشئه
├── requirements.txt           ← قائمة بسيطة
│
├── forest/
│   ├── __init__.py           ← __version__ = "0.1.0"
│   ├── config.py             ← ForestConfig (شغّاله، انظر أدناه)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── spine.py          ← placeholder: class Spine
│   │   ├── router.py         ← placeholder: class Router
│   │   ├── zone.py           ← placeholder: class Zone
│   │   └── model.py          ← placeholder: class NeuralForest
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── sqlite_store.py   ← placeholder
│   │   └── temporal.py       ← placeholder
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── trainer.py
│   │   └── losses.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── vram_monitor.py   ← shaghal: get_vram_usage()
│       └── benchmarks.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config.py        ← shaghal: 3 tests
│   ├── test_spine.py         ← placeholder
│   ├── test_router.py        ← placeholder
│   ├── test_zone.py          ← placeholder
│   └── test_model.py         ← placeholder
│
├── scripts/
│   ├── train_50m.py          ← placeholder مع TODO
│   ├── train_125m.py         ← placeholder
│   ├── train_500m.py         ← placeholder
│   └── benchmark.py          ← placeholder
│
└── notebooks/
    └── colab_training.ipynb  ← فارغ

═══════════════════════════════════════════════════════════════════
متطلبات الكود
═══════════════════════════════════════════════════════════════════

1. forest/config.py - يجب أن يكون شغّالاً:
   - dataclass ForestConfig
   - الحقول:
     * vocab_size: int = 50000
     * embed_dim: int = 512
     * spine_layers: int = 6
     * spine_heads: int = 8
     * spine_head_dim: int = 64
     * num_zones: int = 11   # 0 = Skip Zone, 1-10 = specialty
     * zone_hidden_dim: int = 1024
     * zone_ffn_layers: int = 2
     * max_seq_len: int = 2048
     * router_top_k: int = 1  # default Top-1
     * dropout: float = 0.0
     * use_skip_zone: bool = True
   
   - presets:
     * @classmethod tiny() → 50M params total
     * @classmethod small() → 125M params total
     * @classmethod base() → 500M params total
     * @classmethod large() → 1B params total
   
   - methods:
     * estimate_params() → dict مع تفاصيل
     * estimate_vram_gb(batch_size, seq_len) → float
     * __repr__ يطبع ملخصاً مرتباً

2. forest/utils/vram_monitor.py - شغّال:
   - get_vram_usage() → dict (allocated_gb, reserved_gb, free_gb)
   - يدعم CUDA إذا متاح، يعطي 0.0 إذا CPU only
   - context manager: VRAMTracker() لقياس قبل/بعد

3. tests/test_config.py - 3 tests:
   - test_default_config_is_valid()
   - test_tiny_preset_is_around_50m()  # tolerance ±5M
   - test_small_preset_is_around_125m()

4. كل placeholders يجب أن تحوي:
   - docstring يشرح الغرض
   - """ TODO: implement in PROMPT 2 """ comment
   - class skeleton فارغ
   - import statements الأساسية

5. .gitignore - شامل:
   *.pyc, __pycache__/, .pytest_cache/, .vscode/, .idea/
   *.pt, *.safetensors, *.bin, *.gguf, checkpoints/
   wandb/, runs/, .env, venv/, .venv/
   *.egg-info/, dist/, build/
   datasets/, *.parquet, *.arrow, .cache/

6. pyproject.toml:
   - name = "neural-forest"
   - version = "0.1.0"
   - python_requires = ">=3.11"
   - dependencies: torch>=2.5, einops, numpy, pyyaml
   - dev dependencies: pytest, black, ruff, mypy

7. requirements.txt - مبسط من pyproject.toml

═══════════════════════════════════════════════════════════════════
بعد إنشاء البنية، نفّذ:
═══════════════════════════════════════════════════════════════════

1. cd neural-forest && git init
2. git add -A && git commit -m "Initial commit: project structure"
3. pip install -e . (للتأكد أن الحزمة تعمل)
4. pytest tests/test_config.py -v (للتأكد أن tests تعمل)

═══════════════════════════════════════════════════════════════════
معايير القبول (Definition of Done)
═══════════════════════════════════════════════════════════════════

✅ بنية المجلدات مطابقة 100%
✅ pip install -e . ينجح بدون أخطاء
✅ python -c "from forest.config import ForestConfig; print(ForestConfig.tiny())" يعمل
✅ pytest tests/test_config.py ينجح
✅ git log يظهر commit واحد نظيف

═══════════════════════════════════════════════════════════════════
ملاحظات نهائية
═══════════════════════════════════════════════════════════════════

- لا تكتب أي كود معماري (Spine, Zone, Router) في هذا الـ prompt — فقط placeholders
- ركّز على البنية، الإعدادات، والاختبارات الأولية
- استخدم Python type hints في كل مكان
- استخدم docstrings بأسلوب Google
- لا تستخدم emojis في الكود (في README.md فقط)
- بعد الانتهاء، أعطني تقريراً مختصراً: ماذا أنشأت، كم ملف، نتائج الاختبارات
```

---

## ملاحظات لـ Naoufel:

### قبل تشغيل الـ prompt:
1. افتح Claude Code في VS Code
2. افتح مجلد فارغ (سيُنشأ المشروع داخله)
3. الصق الـ prompt
4. **أرفق ملفي DECISIONS.md و README.md** لـ Claude Code

### بعد التنفيذ:
- تحقق من المخرجات
- شغّل الأوامر التحققية يدوياً
- إذا كل شيء OK → ارجع لي للـ PROMPT 2

### PROMPT 2 سيكون:
بناء `forest/core/spine.py` — أول component معماري حقيقي.
