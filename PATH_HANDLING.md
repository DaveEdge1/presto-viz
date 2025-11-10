# Path Handling in Cross-Repository Workflows

## The Problem

When using presto-viz as a tool in another repository (like LMR2), path handling can be tricky because:
- Scripts may be called from different working directories
- Relative paths resolve differently depending on where you run from
- GitHub Actions may checkout repositories to different locations

## Solution: Script 3 Smart Path Resolution

Script 3 now intelligently handles the `web_data_dir` parameter:

**If you pass a relative path** (e.g., `viz/web_assets/`):
- It will be resolved **relative to the script's location** (presto-viz directory)
- This works correctly regardless of your current working directory

**If you pass an absolute path** (e.g., `/home/runner/work/LMR2/LMR2/presto-viz/viz/web_assets/`):
- It will be used as-is

## Recommended Workflow Patterns

### Pattern 1: Use Relative Path (Simplest)

```yaml
- name: Run presto-viz scripts
  run: |
    DATA_DIR="${{ github.workspace }}/data/CFR_Run_17_Reviz_22"
    OUTPUT_DIR="${{ github.workspace }}/output/CFR_Run_17_Reviz_22"

    cd presto-viz
    python 1_format_data_daholocene_graphem.py "$DATA_DIR"
    python 2_make_maps_and_ts.py "$DATA_DIR" "$OUTPUT_DIR"
    python 3_make_html_file.py "$DATA_DIR" "$OUTPUT_DIR" "viz/web_assets"
```

**Pros:** Simple, works from any directory
**Why it works:** Script 3 resolves `viz/web_assets` relative to its own location

### Pattern 2: Use Absolute Paths (Most Explicit)

```yaml
- name: Run presto-viz scripts
  run: |
    DATA_DIR="${{ github.workspace }}/data/CFR_Run_17_Reviz_22"
    OUTPUT_DIR="${{ github.workspace }}/output/CFR_Run_17_Reviz_22"
    WEB_DATA_DIR="${{ github.workspace }}/presto-viz/viz/web_assets"

    python presto-viz/1_format_data_daholocene_graphem.py "$DATA_DIR"
    python presto-viz/2_make_maps_and_ts.py "$DATA_DIR" "$OUTPUT_DIR"
    python presto-viz/3_make_html_file.py "$DATA_DIR" "$OUTPUT_DIR" "$WEB_DATA_DIR"
```

**Pros:** Explicit, works from any directory, no `cd` needed
**Why it works:** All paths are absolute

## Common Pitfalls to Avoid

### ❌ DON'T: Use relative path from wrong directory

```yaml
# Running from LMR2 root, passing relative path
- run: python presto-viz/3_make_html_file.py "$DATA_DIR" "$OUTPUT_DIR" "viz/web_assets"
```

**Issue:** OLD behavior would resolve `viz/web_assets` to `/home/runner/work/LMR2/LMR2/viz/web_assets` ❌
**Fixed:** NEW behavior resolves it relative to script location: `/home/runner/work/LMR2/LMR2/presto-viz/viz/web_assets` ✅

### ❌ DON'T: Mix working directories

```yaml
- run: |
    cd presto-viz
    python 1_format_data_daholocene_graphem.py "../data/CFR_Run_17_Reviz_22"  # Fragile!
```

**Issue:** Relative paths in arguments depend on current directory
**Fix:** Always use absolute paths for data_dir and output_dir arguments

## Diagnostic Output

Script 3 now prints resolved paths, making it easy to verify:

```
Note: web_data_dir was relative ("viz/web_assets"), resolving relative to script location
  Script location: /home/runner/work/LMR2/LMR2/presto-viz
  Resolved to: /home/runner/work/LMR2/LMR2/presto-viz/viz/web_assets
Data directory: /home/runner/work/LMR2/LMR2/data/CFR_Run_17_Reviz_22
Output directory: /home/runner/work/LMR2/LMR2/output/CFR_Run_17_Reviz_22
Web assets directory: /home/runner/work/LMR2/LMR2/presto-viz/viz/web_assets
```

If you see unexpected paths, check your workflow configuration.

## Path Requirements

| Script | Argument | Must Be Absolute? | Notes |
|--------|----------|-------------------|-------|
| Script 1 | `data_dir` | Recommended | Converted to absolute internally |
| Script 2 | `data_dir` | Recommended | Converted to absolute internally |
| Script 2 | `output_dir` | Recommended | Converted to absolute internally |
| Script 3 | `data_dir` | Recommended | Converted to absolute internally |
| Script 3 | `output_dir` | Recommended | Converted to absolute internally |
| Script 3 | `web_data_dir` | **No** | If relative, resolved to script location |

## Best Practice Summary

✅ **DO:**
- Use `${{ github.workspace }}` for absolute paths in GitHub Actions
- Pass relative `web_data_dir` as simply `viz/web_assets` (script handles it)
- Use absolute paths for `data_dir` and `output_dir`
- Verify paths in script output if issues occur

❌ **DON'T:**
- Rely on current working directory for path resolution
- Use `..` in paths (use absolute paths instead)
- Assume scripts are run from a specific directory

## Troubleshooting

**Error:** `FileNotFoundError: Template file not found: /wrong/path/viz/web_assets/visualizer_template.html`

**Solution:** Check that:
1. presto-viz repository is checked out correctly
2. You're passing `viz/web_assets` (relative) or the correct absolute path
3. Check the diagnostic output showing resolved paths

**Error:** `FileNotFoundError: Script 2 output not found`

**Solution:** Check that:
1. Script 2 completed successfully
2. `output_dir` is the same for both scripts 2 and 3
3. Check script 2's final output message for the actual output location
