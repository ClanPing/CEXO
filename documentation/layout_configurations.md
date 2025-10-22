## 🏗️ Layout configurations

<p align="center">
<img src="assets/config.png" alt="Facility types">
</p>

### Facility selection range:

- **Minimum Facilities**: 3 (minimal operational site)
  - Always includes: `core`, `crane`, `storage`

- **Maximum Facilities**: 8 (complex multi-function site)
  - Includes: All 5 types + additional operational facilities

- **Default Configuration**: 6 facilities
  - Typical mix: `core`, `crane`, `storage`, `office`, `rest_area`, + 1 operational facility

```
Facility combination:

- If count ≥ 3:  Add [core, crane, storage]                     # Operational facilities
- If count ≥ 5:  Add [office, rest_area]                        # Worker facilities
- If count > 5:  Fill remaining with [core, storage, crane]     # Additional operational
- Finally: Shuffle order randomly (seed-controlled)
```

**Example Generations** (default seed=42):

| Count | Facility Mix | Breakdown |
|-------|--------------|-----------|
| 3 | `['storage', 'core', 'crane']` | 3 operational only |
| 4 | `['storage', 'core', 'crane', 'crane']` | 3 operational + 1 extra operational |
| 5 | `['storage', 'core', 'crane', 'rest_area', 'office']` | 3 operational + 2 worker |
| 6 | `['storage', 'core', 'crane', 'rest_area', 'office', 'storage']` | Balanced + 1 extra operational |
| 7 | `['storage', 'core', 'crane', 'rest_area', 'office', 'storage', 'crane']` | Balanced + 2 extra operational |
| 8 | `['storage', 'core', 'crane', 'rest_area', 'office', 'storage', 'crane', 'storage', 'core', 'crane']` | Full site |
